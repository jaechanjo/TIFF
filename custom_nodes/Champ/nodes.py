import os
import os.path as osp
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision.transforms import ToPILImage
from contextlib import nullcontext
from omegaconf import OmegaConf

from transformers import CLIPVisionModelWithProjection

try:
    from diffusers import (
        DDIMScheduler,
        DPMSolverMultistepScheduler, 
        AutoencoderKL, 
        UNet2DConditionModel, 
        LCMScheduler, 
        DDPMScheduler, 
        DEISMultistepScheduler, 
        PNDMScheduler,
        UniPCMultistepScheduler
    )
    from diffusers.loaders.single_file_utils import (
        convert_ldm_vae_checkpoint, 
        convert_ldm_unet_checkpoint, 
        create_vae_diffusers_config, 
        create_unet_diffusers_config,
    )     
except:
    raise ImportError("Diffusers version too old. Please update to 0.26.0 minimum.")
from diffusers.utils import is_accelerate_available
if is_accelerate_available():
    from accelerate import init_empty_weights
    from accelerate.utils import set_module_tensor_to_device

from .scheduling_tcd import TCDScheduler

from .models.unet_2d_condition import UNet2DConditionModel
from .models.unet_3d import UNet3DConditionModel
from .models.mutual_self_attention import ReferenceAttentionControl
from .models.guidance_encoder import GuidanceEncoder
from .models.champ_model import ChampModel
from .pipelines.pipeline_aggregation import MultiGuidance2LongVideoPipeline

import comfy.model_management as mm
import comfy.utils

def convert_dtype(dtype_str):
    if dtype_str == 'fp32':
        return torch.float32
    elif dtype_str == 'fp16':
        return torch.float16
    elif dtype_str == 'bf16':
        return torch.bfloat16
    else:
        raise NotImplementedError

def setup_guidance_encoder(cfg):
    guidance_encoder_group = dict()
    
    if cfg.weight_dtype == "fp16":
        weight_dtype = torch.float16
    else:
        weight_dtype = torch.float32
    
    for guidance_type in cfg.guidance_types:
        guidance_encoder_group[guidance_type] = GuidanceEncoder(
            guidance_embedding_channels=cfg.guidance_encoder_kwargs.guidance_embedding_channels,
            guidance_input_channels=cfg.guidance_encoder_kwargs.guidance_input_channels,
            block_out_channels=cfg.guidance_encoder_kwargs.block_out_channels,
        ).to(device="cuda", dtype=weight_dtype)
    
    return guidance_encoder_group

def process_semantic_map(semantic_map_path: Path):
    image_name = semantic_map_path.name
    mask_path = semantic_map_path.parent.parent / "mask" / image_name
    semantic_array = np.array(Image.open(semantic_map_path))
    mask_array = np.array(Image.open(mask_path).convert("RGB"))
    semantic_pil = Image.fromarray(np.where(mask_array > 0, semantic_array, 0))
    
    return semantic_pil

script_directory = os.path.dirname(os.path.abspath(__file__))

def combine_guidance_data_from_tensors(guidance_tensor_batches):
    guidance_pil_group = {}
    to_pil = ToPILImage()

    for guidance_type, tensor_batch in guidance_tensor_batches.items():
        guidance_pil_group[guidance_type] = []
        for i in range(tensor_batch.size(0)):  # Iterate over the batch
            tensor = tensor_batch[i]
            # Permute the tensor from B, H, W, C to B, C, H, W
            tensor = tensor.permute(2, 0, 1)
            # Convert tensor to PIL Image
            pil_image = to_pil(tensor)
            # Add the PIL Image to the group
            guidance_pil_group[guidance_type].append(pil_image)

    # Get video length from the first guidance sequence
    first_guidance_length = len(guidance_pil_group[next(iter(guidance_tensor_batches.keys()))])
    # Ensure all guidance sequences are of equal length
    assert all(len(sublist) == first_guidance_length for sublist in guidance_pil_group.values())

    return guidance_pil_group, first_guidance_length

class champ_model_loader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "model": ("MODEL",),
            
            "vae": ("VAE",),
            "diffusion_dtype": (
                    [
                        'fp32',
                        'fp16',
                        'bf16',
                        'auto',
                    ], {
                        "default": 'auto'
                    }),
            "vae_dtype": (
                    [
                        'fp32',
                        'fp16',
                        'bf16',
                        'auto'
                    ], {
                        "default": 'auto'
                    }),
            },
            "optional": {
                "motion_model":("MOTION_MODEL_ADE",),
            }
        }

    RETURN_TYPES = ("CHAMPMODEL", "CHAMPVAE", "CHAMPENCODER")
    RETURN_NAMES = ("champ_model", "champ_vae", "champ_encoder",)
    FUNCTION = "loadmodel"
    CATEGORY = "champWrapper"

    def loadmodel(self, model, vae, diffusion_dtype, vae_dtype, motion_model=None):
        mm.soft_empty_cache()
        device = mm.get_torch_device()
        config_path = os.path.join(script_directory, "configs/inference.yaml")
        cfg = OmegaConf.load(config_path)

        custom_config = {
            'diffusion_dtype': diffusion_dtype,
            'vae_dtype': vae_dtype,
            'model': model,
            'vae': vae,
            'motion_model' : motion_model
        }
        if not hasattr(self, 'model') or self.model == None or custom_config != self.current_config:
            pbar = comfy.utils.ProgressBar(7)
            self.current_config = custom_config
            # setup pretrained models
            original_config = OmegaConf.load(os.path.join(script_directory, f"configs/v1-inference.yaml"))
            ad_unet_config = OmegaConf.load(os.path.join(script_directory, f"configs/ad_unet_config.yaml"))
            if diffusion_dtype == 'auto':
                try:
                    if mm.should_use_fp16():
                        print("Diffusion using fp16")
                        dtype = torch.float16
                    elif mm.should_use_bf16():
                        print("Diffusion using bf16")
                        dtype = torch.bfloat16
                    else:
                        print("Diffusion using fp32")
                        dtype = torch.float32
                except:
                    raise AttributeError("ComfyUI version too old, can't autodecet properly. Set your dtypes manually.")
            else:
                print(f"Diffusion using {diffusion_dtype}")
                dtype = convert_dtype(diffusion_dtype)

            denoising_unet_path = os.path.join(script_directory,"checkpoints", "denoising_unet.pth")
            reference_unet_path = os.path.join(script_directory,"checkpoints", "reference_unet.pth")
            motion_module_path = os.path.join(script_directory,"checkpoints", "motion_module.pth")
             
            mm.load_model_gpu(model)
            sd = model.model.state_dict_for_saving(None, vae.get_sd(), None)

            # 1. vae
            converted_vae_config = create_vae_diffusers_config(original_config, image_size=512)
            converted_vae = convert_ldm_vae_checkpoint(sd, converted_vae_config)
           
            with (init_empty_weights() if is_accelerate_available() else nullcontext()):
                self.vae = AutoencoderKL(**converted_vae_config)
            if is_accelerate_available():
                for key in converted_vae:
                    set_module_tensor_to_device(self.vae, key, device=device, dtype=dtype, value=converted_vae[key])
            else:
                self.vae.load_state_dict(converted_vae, strict=False)

            if vae_dtype == "auto":
                try:
                    if mm.should_use_bf16():
                        self.vae.to(convert_dtype('bf16'))
                    else:
                        self.vae.to(convert_dtype('fp32'))
                except:
                    raise AttributeError("ComfyUI version too old, can't autodetect properly. Set your dtype manually.")
            else:
                self.vae.to(convert_dtype(vae_dtype))
            print(f"VAE using dtype: {self.vae.dtype}")
            pbar.update(1)
            # 2. unet
            converted_unet_config = create_unet_diffusers_config(original_config, image_size=512)
            converted_unet = convert_ldm_unet_checkpoint(sd, converted_unet_config)
            del sd
            reference_unet = UNet2DConditionModel(**converted_unet_config)
            reference_unet.load_state_dict(converted_unet, strict=False)
            pbar.update(1)
            denoising_unet = UNet3DConditionModel(**ad_unet_config)
            denoising_unet.load_state_dict(converted_unet, strict=False)
            pbar.update(1)

            if motion_model is not None:
                motion_state_dict = motion_model.model.state_dict()
                if motion_model.model.mm_info.mm_format == "AnimateLCM":
                    motion_state_dict = {k: v for k, v in motion_state_dict.items() if "pos_encoder" not in k}
            else:
                motion_state_dict = torch.load(motion_module_path, map_location="cpu", weights_only=True)
            pbar.update(1)
            denoising_unet.load_state_dict(motion_state_dict, strict=False)
            del motion_state_dict
            
            guidance_encoder_group = setup_guidance_encoder(cfg)
            
            denoising_unet.load_state_dict(torch.load(denoising_unet_path, map_location="cpu"), strict=False)
            reference_unet.load_state_dict(torch.load(reference_unet_path, map_location="cpu"), strict=False)

            denoising_unet.to(dtype).to(device)
            reference_unet.to(dtype).to(device)
            pbar.update(1)
            for guidance_type, guidance_encoder_module in guidance_encoder_group.items():
                guidance_encoder_module.load_state_dict(
                    torch.load(
                        osp.join(script_directory,"checkpoints", f"guidance_encoder_{guidance_type}.pth"),
                        map_location="cpu",
                    ),
                    strict=False,
                )
            pbar.update(1)
            reference_control_writer = ReferenceAttentionControl(
                reference_unet,
                do_classifier_free_guidance=False,
                mode="write",
                fusion_blocks="full",
            )
            reference_control_reader = ReferenceAttentionControl(
                denoising_unet,
                do_classifier_free_guidance=False,
                mode="read",
                fusion_blocks="full",
            )
                
            self.model = ChampModel(
                reference_unet=reference_unet,
                denoising_unet=denoising_unet,
                reference_control_writer=reference_control_writer,
                reference_control_reader=reference_control_reader,
                guidance_encoder_group=guidance_encoder_group,
            ).to(device, dtype=dtype)
            pbar.update(1)
            if mm.XFORMERS_IS_AVAILABLE:
                reference_unet.enable_xformers_memory_efficient_attention()
                denoising_unet.enable_xformers_memory_efficient_attention()

        if not hasattr(self, 'image_enc') or self.image_enc == None:
            self.image_enc = CLIPVisionModelWithProjection.from_pretrained(os.path.join(script_directory,"checkpoints", "image_encoder"))
            self.image_enc.to(dtype).to(device)
        
   
        return (self.model, self.vae, self.image_enc)
    
class champ_sampler:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "champ_model": ("CHAMPMODEL",),
            "champ_vae": ("CHAMPVAE",),
            "champ_encoder": ("CHAMPENCODER",),
            "image": ("IMAGE",),
            "width": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 64}),
            "height": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 64}),
            "steps": ("INT", {"default": 20, "min": 1, "max": 200, "step": 1}),
            "guidance_scale": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 20.0, "step": 0.01}),
            "frames": ("INT", {"default": 16, "min": 1, "max": 100, "step": 1}),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            "keep_model_loaded": ("BOOLEAN", {"default": True}),
            "latent_image": ("LATENT", {"default": None}),
            "start_at_step": ("INT", {"default": 0, "min": 0, "max": 10000}),
        },
            "optional":{
            "depth_tensors": ("IMAGE",),
            "normal_tensors": ("IMAGE",),
            "semantic_tensors": ("IMAGE",),
            "dwpose_tensors": ("IMAGE",),
            "scheduler": (
                [
                    'DDIMScheduler',
                    'DDPMScheduler',
                    'LCMScheduler',
                    'PNDMScheduler',
                    'DEISMultistepScheduler',
                    'DPMSolverMultistepScheduler',
                    'UniPCMultistepScheduler',
                    'TCDScheduler'
                ], {
                    "default": 'DDIMScheduler'
                }),
            "style_fidelity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
            }
    
        }

    RETURN_TYPES = ("IMAGE", "IMAGE",)
    RETURN_NAMES = ("images", "last_image",)
    FUNCTION = "process"
    CATEGORY = "champWrapper"

    def process(self, champ_model, champ_vae, champ_encoder, image, width, height, 
                guidance_scale, steps, seed, keep_model_loaded, frames, latent_image, start_at_step, style_fidelity=1.0, depth_tensors=None, normal_tensors=None, semantic_tensors=None, dwpose_tensors=None, scheduler='DDIMScheduler'):
        device = mm.get_torch_device()
        mm.unload_all_models()
        mm.soft_empty_cache()
        model = champ_model
        vae = champ_vae
        image_enc = champ_encoder
        torch.manual_seed(seed)
        dtype = model.reference_unet.dtype
        print(dtype)
        

        config_path = os.path.join(script_directory, "configs/inference.yaml")
        cfg = OmegaConf.load(config_path)

        sched_kwargs = OmegaConf.to_container(cfg.noise_scheduler_kwargs)
        if cfg.enable_zero_snr:
            sched_kwargs.update( 
                rescale_betas_zero_snr=True,
                timestep_spacing="trailing",
                prediction_type="v_prediction",
            )
        sched_kwargs.update({"beta_schedule": "scaled_linear"})

        if scheduler == 'DDIMScheduler':
            noise_scheduler = DDIMScheduler(**sched_kwargs)
        elif scheduler == 'DDPMScheduler':
            noise_scheduler = DDPMScheduler(**sched_kwargs)
        elif scheduler == 'LCMScheduler':
            noise_scheduler = LCMScheduler(**sched_kwargs)
        elif scheduler == 'PNDMScheduler':
            sched_kwargs.pop("clip_sample", None)
            sched_kwargs.pop("rescale_betas_zero_snr", None)
            noise_scheduler = PNDMScheduler(**sched_kwargs)
        elif scheduler == 'DEISMultistepScheduler':
            sched_kwargs.pop("clip_sample", None)
            sched_kwargs.pop("rescale_betas_zero_snr", None)
            noise_scheduler = DEISMultistepScheduler(**sched_kwargs)
        elif scheduler == 'DPMSolverMultistepScheduler':
            sched_kwargs.pop("clip_sample", None)
            sched_kwargs.pop("rescale_betas_zero_snr", None)
            sched_kwargs.update({"algorithm_type": "sde-dpmsolver++"})
            sched_kwargs.update({"use_karras_sigmas": "True"})
            noise_scheduler = DPMSolverMultistepScheduler(**sched_kwargs)
        elif scheduler == 'UniPCMultistepScheduler':
            sched_kwargs.pop("clip_sample", None)
            sched_kwargs.pop("rescale_betas_zero_snr", None)
            noise_scheduler = UniPCMultistepScheduler(**sched_kwargs)
        elif scheduler == 'TCDScheduler':
            noise_scheduler = TCDScheduler(**sched_kwargs)        
        
        model.to(device)

        autocast_condition = (dtype != torch.float32) and not mm.is_device_mps(device)
        with torch.autocast(mm.get_autocast_device(device), dtype=dtype) if autocast_condition else nullcontext():
            image = image.permute(0, 3, 1, 2).to(dtype).to(device)

            B, C, H, W = image.shape
            orig_H, orig_W = H, W
            if W % 64 != 0:
                W = W - (W % 64)
            if H % 64 != 0:
                H = H - (H % 64)
            if orig_H % 64 != 0 or orig_W % 64 != 0:
                image = F.interpolate(image, size=(H, W), mode="bicubic")
           
            B, C, H, W = image.shape
            
            to_pil = ToPILImage()
            ref_image_pil = to_pil(image[0])

            guidance_tensor_batches = {}
            if depth_tensors is not None:
                guidance_tensor_batches["depth"] = depth_tensors
            if normal_tensors is not None:
                guidance_tensor_batches["normal"] = normal_tensors
            if semantic_tensors is not None:
                guidance_tensor_batches["semantic_map"] = semantic_tensors
            if dwpose_tensors is not None:
                guidance_tensor_batches["dwpose"] = dwpose_tensors
                
            guidance_pil_group, video_length = combine_guidance_data_from_tensors(guidance_tensor_batches)

            result_video_tensor = inference(
                cfg=cfg,
                vae=vae,
                image_enc=image_enc,
                model=model,
                scheduler=noise_scheduler,
                ref_image_pil=ref_image_pil,
                guidance_pil_group=guidance_pil_group,
                video_length=frames,
                width=width, height=height,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                start_at_step=start_at_step,
                latent_image=latent_image,
                style_fidelity=style_fidelity,
                device=device, dtype=dtype
            )  # (1, c, f, h, w)
            
            result_video_tensor = result_video_tensor.squeeze(0)
            result_video_tensor = result_video_tensor.permute(1, 2, 3, 0).cpu()
            if not keep_model_loaded:
                model.to('cpu')
            return (result_video_tensor,)
def inference(
    cfg,
    vae,
    image_enc,
    model,
    scheduler,
    ref_image_pil,
    guidance_pil_group,
    video_length,
    width,
    height,
    num_inference_steps,
    guidance_scale,
    start_at_step,
    latent_image,
    style_fidelity,
    dtype,
    device,
):
    reference_unet = model.reference_unet
    denoising_unet = model.denoising_unet
    guidance_types = cfg.guidance_types
    guidance_encoder_group = {f"guidance_encoder_{g}": getattr(model, f"guidance_encoder_{g}") for g in guidance_types}
    
    generator = torch.Generator(device=device)
    generator.manual_seed(cfg.seed)
    pipeline = MultiGuidance2LongVideoPipeline(
        vae=vae,
        image_encoder=image_enc,
        reference_unet=reference_unet,
        denoising_unet=denoising_unet,
        **guidance_encoder_group,
        scheduler=scheduler,
    )
    pipeline = pipeline.to(device, dtype)
    
    video = pipeline(
        ref_image_pil,
        guidance_pil_group,
        width,
        height,
        video_length,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
        start_at_step=start_at_step,
        latent_image=latent_image,
        style_fidelity=style_fidelity,
    ).videos
    
    del pipeline
    mm.soft_empty_cache()
    
    return video       

NODE_CLASS_MAPPINGS = {
    "champ_model_loader": champ_model_loader,
    "champ_sampler": champ_sampler,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "champ_model_loader": "champ_model_loader",
    "champ_sampler": "champ_sampler",
}
