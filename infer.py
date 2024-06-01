# This script is the inference code with BiRefNet removed, as proposed in the paper.

import os
import random
import sys
import json
import argparse
import contextlib
from typing import Sequence, Mapping, Any, Union
import torch
import subprocess
import numpy as np
from PIL import Image, PngImagePlugin, ExifTags
from datetime import datetime


def get_value_at_index(obj: Union[Sequence, Mapping], index: int) -> Any:
    """Returns the value at the given index of a sequence or mapping.

    If the object is a sequence (like list or string), returns the value at the given index.
    If the object is a mapping (like a dictionary), returns the value at the index-th key.

    Some return a dictionary, in these cases, we look for the "results" key

    Args:
        obj (Union[Sequence, Mapping]): The object to retrieve the value from.
        index (int): The index of the value to retrieve.

    Returns:
        Any: The value at the given index.

    Raises:
        IndexError: If the index is out of bounds for the object and the object is not a mapping.
    """
    try:
        return obj[index]
    except KeyError:
        return obj["result"][index]


def find_path(name: str, path: str = None) -> str:
    """
    Recursively looks at parent folders starting from the given path until it finds the given name.
    Returns the path as a Path object if found, or None otherwise.
    """
    # If no path is given, use the current working directory
    if path is None:
        if args is None or args.comfyui_directory is None:
            path = os.getcwd()
        else:
            path = args.comfyui_directory

    # Check if the current directory contains the name
    if name in os.listdir(path):
        path_name = os.path.join(path, name)
        print(f"{name} found: {path_name}")
        return path_name

    # Get the parent directory
    parent_directory = os.path.dirname(path)

    # If the parent directory is the same as the current directory, we've reached the root and stop the search
    if parent_directory == path:
        return None

    # Recursively call the function with the parent directory
    return find_path(name, parent_directory)


def add_comfyui_directory_to_sys_path() -> None:
    """
    Add 'ComfyUI' to the sys.path
    """
    comfyui_path = find_path("ComfyUI")
    if comfyui_path is not None and os.path.isdir(comfyui_path):
        sys.path.append(comfyui_path)
        import __main__

        if getattr(__main__, "__file__", None) is None:
            __main__.__file__ = os.path.join(comfyui_path, "main.py")

        print(f"'{comfyui_path}' added to sys.path")


def add_extra_model_paths() -> None:
    """
    Parse the optional extra_model_paths.yaml file and add the parsed paths to the sys.path.
    """
    from main import load_extra_path_config

    extra_model_paths = find_path("extra_model_paths.yaml")

    if extra_model_paths is not None:
        load_extra_path_config(extra_model_paths)
    else:
        print("Could not find the extra_model_paths config file.")


def import_custom_nodes() -> None:
    """Find all custom nodes in the custom_nodes folder and add those node objects to NODE_CLASS_MAPPINGS

    This function sets up a new asyncio event loop, initializes the PromptServer,
    creates a PromptQueue, and initializes the custom nodes.
    """
    import asyncio
    import execution
    from nodes import init_custom_nodes
    import server

    # Creating a new event loop and setting it as the default loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Creating an instance of PromptServer with the loop
    server_instance = server.PromptServer(loop)
    execution.PromptQueue(server_instance)

    # Initializing custom nodes
    init_custom_nodes()


def save_image_wrapper(context, cls):
    if args.output_dir is None:
        return cls

    from PIL import Image, ImageOps, ImageSequence
    from PIL.PngImagePlugin import PngInfo

    import numpy as np

    class WrappedSaveImage(cls):
        counter = 0

        def save_images(
            self, images, filename_prefix="ComfyUI", prompt=None, extra_pnginfo=None
        ):
            if args.output_dir is None:
                return super().save_images(
                    images, filename_prefix, prompt, extra_pnginfo
                )
            else:
                if len(images) > 1 and args.output_dir == "-":
                    raise ValueError("Cannot save multiple images to stdout")
                filename_prefix += self.prefix_append

                results = list()
                for batch_number, image in enumerate(images):
                    i = 255.0 * image.cpu().numpy()
                    img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
                    metadata = None
                    if not args.disable_metadata:
                        metadata = PngInfo()
                        if prompt is not None:
                            metadata.add_text("prompt", json.dumps(prompt))
                        if extra_pnginfo is not None:
                            for x in extra_pnginfo:
                                metadata.add_text(x, json.dumps(extra_pnginfo[x]))

                    if args.output_dir == "-":
                        # Hack to briefly restore stdout
                        if context is not None:
                            context.__exit__(None, None, None)
                        try:
                            img.save(
                                sys.stdout.buffer,
                                format="png",
                                pnginfo=metadata,
                                compress_level=self.compress_level,
                            )
                        finally:
                            if context is not None:
                                context.__enter__()
                    else:
                        subfolder = ""
                        if len(images) == 1:
                            if os.path.isdir(args.output_dir):
                                subfolder = args.output_dir
                                file = "output.png"
                            else:
                                subfolder, file = os.path.split(args.output_dir)
                                if subfolder == "":
                                    subfolder = os.getcwd()
                        else:
                            if os.path.isdir(args.output_dir):
                                subfolder = args.output_dir
                                file = filename_prefix
                            else:
                                subfolder, file = os.path.split(args.output_dir)

                            if subfolder == "":
                                subfolder = os.getcwd()

                            files = os.listdir(subfolder)
                            file_pattern = file
                            while True:
                                filename_with_batch_num = file_pattern.replace(
                                    "%batch_num%", str(batch_number)
                                )
                                file = (
                                    f"{filename_with_batch_num}_{self.counter:05}.png"
                                )
                                self.counter += 1

                                if file not in files:
                                    break

                        img.save(
                            os.path.join(subfolder, file),
                            pnginfo=metadata,
                            compress_level=self.compress_level,
                        )
                        print("Saved image to", os.path.join(subfolder, file))
                        results.append(
                            {
                                "filename": file,
                                "subfolder": subfolder,
                                "type": self.type,
                            }
                        )

                return {"ui": {"images": results}}

    return WrappedSaveImage


def parse_arg(s: Any):
    """Parses a JSON string, returning it unchanged if the parsing fails."""
    if __name__ == "__main__" or not isinstance(s, str):
        return s

    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return s


parser = argparse.ArgumentParser(
    description="A converted ComfyUI workflow. Required inputs listed below. Values passed should be valid JSON (assumes string if not valid JSON)."
)
parser.add_argument(
    "--queue_size",
    "-q",
    type=int,
    default=1,
    help="How many times the workflow will be executed (default: 1)",
)

parser.add_argument(
    "--comfyui_directory",
    "-c",
    default=None,
    help="Where to look for ComfyUI (default: current directory)",
)

parser.add_argument(
    "--output_dir",
    "-o",
    default=None,
    help="The directory to save the output image. Either a file path, a directory, or - for stdout (default: the ComfyUI output directory)",
)

parser.add_argument(
    "--disable_metadata",
    action="store_true",
    help="Disables writing workflow metadata to the outputs",
)

parser.add_argument(
    "--source_video",
    type=str,
    required=True,
    help="The path to the source video file.",
)

parser.add_argument(
    "--frame_load_cap",
    type=int,
    default=16,
    help="The maximum number of frames to load.",
)

parser.add_argument(
    "--positive_prompt",
    type=str,
    required=True,
    help="The positive prompt for the generation.",
)

parser.add_argument(
    "--negative_prompt",
    type=str,
    required=True,
    help="The negative prompt for the generation.",
)

parser.add_argument(
    "--ref_image",
    type=str,
    required=True,
    help="The reference image to use for the generation.",
)

parser.add_argument(
    "--output_frame_rate",
    type=str,
    default=8,
    required=True,
    help="The reference image to use for the generation.",
)

comfy_args = [sys.argv[0]]
if "--" in sys.argv:
    idx = sys.argv.index("--")
    comfy_args += sys.argv[idx + 1 :]
    sys.argv = sys.argv[:idx]

args = None
if __name__ == "__main__":
    args = parser.parse_args()
    sys.argv = comfy_args
if args is not None and args.output_dir is not None and args.output_dir == "-":
    ctx = contextlib.redirect_stdout(sys.stderr)
else:
    ctx = contextlib.nullcontext()

PROMPT_DATA = json.loads(
    '{"10": {"inputs": {"diffusion_dtype": "auto", "vae_dtype": "auto", "model": ["127", 0], "vae": ["97", 0]}, "class_type": "champ_model_loader", "_meta": {"title": "champ_model_loader"}}, "12": {"inputs": {"width": ["108", 0], "height": ["108", 1], "steps": 36, "guidance_scale": 2, "frames": ["15", 1], "seed": 0, "keep_model_loaded": true, "scheduler": "DDPMScheduler", "champ_model": ["10", 0], "champ_vae": ["10", 1], "champ_encoder": ["10", 2], "image": ["20", 0], "depth_tensors": ["63", 0], "normal_tensors": ["59", 0], "semantic_tensors": ["66", 0], "dwpose_tensors": ["18", 0]}, "class_type": "champ_sampler", "_meta": {"title": "champ_sampler"}}, "15": {"inputs": {"video": "motion.mp4", "force_rate": 8, "force_size": "Disabled", "custom_width": 1024, "custom_height": 1024, "frame_load_cap": 16, "skip_first_frames": 0, "select_every_nth": 1}, "class_type": "VHS_LoadVideo", "_meta": {"title": "Load Video (Upload) \\ud83c\\udfa5\\ud83c\\udd65\\ud83c\\udd57\\ud83c\\udd62"}}, "18": {"inputs": {"detect_hand": "enable", "detect_body": "enable", "detect_face": "enable", "resolution": ["22", 0], "bbox_detector": "yolox_l.onnx", "pose_estimator": "dw-ll_ucoco_384_bs5.torchscript.pt", "image": ["23", 0]}, "class_type": "DWPreprocessor", "_meta": {"title": "DWPose Estimator"}}, "20": {"inputs": {"width": ["108", 0], "height": ["108", 1], "interpolation": "lanczos", "keep_proportion": true, "condition": "always", "multiple_of": 64, "image": ["233", 0]}, "class_type": "ImageResize+", "_meta": {"title": "\\ud83d\\udd27 Image Resize"}}, "22": {"inputs": {"image_gen_width": ["108", 0], "image_gen_height": ["108", 1], "resize_mode": "Resize and Fill", "original_image": ["23", 0]}, "class_type": "PixelPerfectResolution", "_meta": {"title": "Pixel Perfect Resolution"}}, "23": {"inputs": {"width": ["108", 0], "height": ["108", 1], "interpolation": "lanczos", "keep_proportion": false, "condition": "always", "multiple_of": 64, "image": ["15", 0]}, "class_type": "ImageResize+", "_meta": {"title": "\\ud83d\\udd27 Image Resize"}}, "28": {"inputs": {"width": ["108", 1], "height": ["108", 1], "batch_size": 1, "color": 0}, "class_type": "EmptyImage", "_meta": {"title": "EmptyImage"}}, "56": {"inputs": {"fov": 60, "iterations": 5, "resolution": ["103", 0], "image": ["23", 0]}, "class_type": "DSINE-NormalMapPreprocessor", "_meta": {"title": "DSINE Normal Map"}}, "59": {"inputs": {"x": 0, "y": 0, "resize_source": true, "destination": ["56", 0], "source": ["28", 0]}, "class_type": "ImageCompositeMasked", "_meta": {"title": "ImageCompositeMasked"}}, "62": {"inputs": {"ckpt_name": "depth_anything_vitl14.pth", "resolution": ["103", 0], "image": ["23", 0]}, "class_type": "DepthAnythingPreprocessor", "_meta": {"title": "Depth Anything"}}, "63": {"inputs": {"x": 0, "y": 0, "resize_source": true, "destination": ["62", 0], "source": ["28", 0]}, "class_type": "ImageCompositeMasked", "_meta": {"title": "ImageCompositeMasked"}}, "65": {"inputs": {"model": "densepose_r101_fpn_dl.torchscript", "cmap": "Viridis (MagicAnimate)", "resolution": ["103", 0], "image": ["23", 0]}, "class_type": "DensePosePreprocessor", "_meta": {"title": "DensePose Estimator"}}, "66": {"inputs": {"x": 0, "y": 0, "resize_source": true, "destination": ["65", 0], "source": ["28", 0]}, "class_type": "ImageCompositeMasked", "_meta": {"title": "ImageCompositeMasked"}}, "97": {"inputs": {"vae_name": "1.5\\\\vae-ft-mse-840000-ema-pruned.safetensors"}, "class_type": "VAELoader", "_meta": {"title": "Load VAE"}}, "101": {"inputs": {"enabled": true, "swap_model": "inswapper_128.onnx", "facedetection": "retinaface_resnet50", "face_restore_model": "codeformer-v0.1.0.pth", "face_restore_visibility": 1, "codeformer_weight": 0.5, "detect_gender_input": "female", "detect_gender_source": "female", "input_faces_index": "0", "source_faces_index": "0", "console_log_level": 1, "input_image": ["12", 0], "source_image": ["233", 0]}, "class_type": "ReActorFaceSwap", "_meta": {"title": "ReActor - Fast Face Swap"}}, "102": {"inputs": {"frame_rate": 8, "loop_count": 0, "filename_prefix": "ComfyUI_swap", "format": "video/h264-mp4", "pix_fmt": "yuv420p", "crf": 19, "save_metadata": false, "pingpong": false, "save_output": false, "images": ["101", 0], "audio": ["15", 2]}, "class_type": "VHS_VideoCombine", "_meta": {"title": "Video Combine \\ud83c\\udfa5\\ud83c\\udd65\\ud83c\\udd57\\ud83c\\udd62"}}, "103": {"inputs": {"image_gen_width": ["108", 0], "image_gen_height": ["108", 1], "resize_mode": "Just Resize", "original_image": ["107", 0]}, "class_type": "PixelPerfectResolution", "_meta": {"title": "Pixel Perfect Resolution"}}, "107": {"inputs": {"upscale_method": "lanczos", "megapixels": 0.26, "image": ["15", 0]}, "class_type": "ImageScaleToTotalPixels", "_meta": {"title": "ImageScaleToTotalPixels"}}, "108": {"inputs": {"image": ["107", 0]}, "class_type": "GetImageSize+", "_meta": {"title": "\\ud83d\\udd27 Get Image Size"}}, "127": {"inputs": {"ckpt_name": "photonLCM_v10.safetensors"}, "class_type": "CheckpointLoaderSimple", "_meta": {"title": "Load Checkpoint"}}, "142": {"inputs": {"seed": 0, "steps": 30, "cfg": 3, "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": 1, "model": ["203", 0], "positive": ["311", 0], "negative": ["311", 1], "latent_image": ["144", 0]}, "class_type": "KSampler", "_meta": {"title": "KSampler"}}, "143": {"inputs": {"ckpt_name": "albedobaseXL_v21.safetensors"}, "class_type": "CheckpointLoaderSimple", "_meta": {"title": "Load Checkpoint"}}, "144": {"inputs": {"width": ["227", 0], "height": ["227", 1], "batch_size": 1}, "class_type": "EmptyLatentImage", "_meta": {"title": "Empty Latent Image"}}, "145": {"inputs": {"text": "yellow hair, black jacket, sports clothes, walking on a sidewalk, full body, (looking_at_viewer:1.2), smiling, cityscape background, night, (8k, RAW photo, best quality, masterpiece:1.2), (realistic, photo-realistic:1.37), professional lighting, photon mapping, physically-based rendering, octane render, perfect anatomy, realistic hands", "clip": ["143", 1]}, "class_type": "CLIPTextEncode", "_meta": {"title": "CLIP Text Encode (Prompt)"}}, "146": {"inputs": {"text": "contortionist, amputee, polydactyly, deformed, distorted, misshapen, malformed, abnormal, mutant, defaced, shapeless\\n(worst quality:2), (low quality:2), (normal quality:2), normal quality, ((monochrome)), ((grayscale)), skin spots, acnes, skin blemishes, age spot, glans, sketches, drawing, painting, (watermark, username, signature, text:1.3),((bad anatomy, bad proportions)), blurry, cloned face, cropped, ((deformed)), dehydrated, disfigured, (duplicate, morbid:2), error, extra arms, extra fingers, extra legs, extra limbs, fused fingers, gross proportions, jpeg artifacts, long neck, low quality, malformed limbs, missing arms, missing legs, mutated hands, mutation, mutilated, out of frame, poorly drawn face, poorly drawn hands, signature, text, ugly, username, watermark, worst quality, fused fingers, too many fingers, extra fingers", "clip": ["143", 1]}, "class_type": "CLIPTextEncode", "_meta": {"title": "CLIP Text Encode (Prompt)"}}, "147": {"inputs": {"samples": ["142", 0], "vae": ["143", 2]}, "class_type": "VAEDecode", "_meta": {"title": "VAE Decode"}}, "149": {"inputs": {"crop_padding_factor": 0.25, "cascade_xml": "lbpcascade_animeface.xml", "image": ["150", 0]}, "class_type": "Image Crop Face", "_meta": {"title": "Image Crop Face"}}, "150": {"inputs": {"image": "PEB6 (2).jpeg", "upload": "image"}, "class_type": "LoadImage", "_meta": {"title": "Load Image"}}, "153": {"inputs": {"instantid_file": "ip-adapter.bin"}, "class_type": "InstantIDModelLoader", "_meta": {"title": "Load InstantID Model"}}, "154": {"inputs": {"provider": "CUDA"}, "class_type": "InstantIDFaceAnalysis", "_meta": {"title": "InstantID Face Analysis"}}, "155": {"inputs": {"control_net_name": "InstantID IdentityNet.safetensors"}, "class_type": "ControlNetLoader", "_meta": {"title": "Load ControlNet Model"}}, "156": {"inputs": {"width": 1024, "height": 1024, "interpolation": "lanczos", "keep_proportion": false, "condition": "always", "multiple_of": 0, "image": ["149", 0]}, "class_type": "ImageResize+", "_meta": {"title": "\\ud83d\\udd27 Image Resize"}}, "194": {"inputs": {"weight": 0.3, "weight_type": "linear", "combine_embeds": "concat", "start_at": 0, "end_at": 1, "embeds_scaling": "V only", "model": ["224", 0], "ipadapter": ["197", 0], "image": ["156", 0], "clip_vision": ["195", 0]}, "class_type": "IPAdapterAdvanced", "_meta": {"title": "IPAdapter Advanced"}}, "195": {"inputs": {"clip_name": "IPAdapter image_encoder_sd15.safetensors"}, "class_type": "CLIPVisionLoader", "_meta": {"title": "Load CLIP Vision"}}, "197": {"inputs": {"ipadapter_file": "ip-adapter-plus-face_sdxl_vit-h.safetensors"}, "class_type": "IPAdapterModelLoader", "_meta": {"title": "IPAdapter Model Loader"}}, "203": {"inputs": {"multiplier": 0.7, "model": ["194", 0]}, "class_type": "RescaleCFG", "_meta": {"title": "RescaleCFG"}}, "215": {"inputs": {"faceanalysis": ["154", 0], "image": ["293", 0]}, "class_type": "FaceKeypointsPreprocessor", "_meta": {"title": "Face Keypoints Preprocessor"}}, "216": {"inputs": {"images": ["215", 0]}, "class_type": "PreviewImage", "_meta": {"title": "Preview Image"}}, "224": {"inputs": {"weight": 0.6, "start_at": 0, "end_at": 1, "instantid": ["153", 0], "insightface": ["154", 0], "control_net": ["155", 0], "image": ["156", 0], "model": ["143", 0], "positive": ["145", 0], "negative": ["146", 0], "image_kps": ["215", 0]}, "class_type": "ApplyInstantID", "_meta": {"title": "Apply InstantID"}}, "227": {"inputs": {"image": ["150", 0]}, "class_type": "Get image size", "_meta": {"title": "Get image size"}}, "233": {"inputs": {"enabled": true, "swap_model": "inswapper_128.onnx", "facedetection": "retinaface_resnet50", "face_restore_model": "codeformer-v0.1.0.pth", "face_restore_visibility": 1, "codeformer_weight": 0.5, "detect_gender_input": "no", "detect_gender_source": "no", "input_faces_index": "0", "source_faces_index": "0", "console_log_level": 1, "input_image": ["147", 0], "source_image": ["150", 0]}, "class_type": "ReActorFaceSwap", "_meta": {"title": "ReActor - Fast Face Swap"}}, "293": {"inputs": {"image": "man_full_shot_origin.jpeg", "upload": "image"}, "class_type": "LoadImage", "_meta": {"title": "Load Image"}}, "307": {"inputs": {"detect_hand": "enable", "detect_body": "enable", "detect_face": "enable", "resolution": ["227", 0], "bbox_detector": "yolox_l.onnx", "pose_estimator": "dw-ll_ucoco_384_bs5.torchscript.pt", "image": ["293", 0]}, "class_type": "DWPreprocessor", "_meta": {"title": "DWPose Estimator"}}, "311": {"inputs": {"strength": 0.3, "start_percent": 0, "end_percent": 1, "positive": ["224", 1], "negative": ["224", 2], "control_net": ["312", 0], "image": ["307", 0]}, "class_type": "ControlNetApplyAdvanced", "_meta": {"title": "Apply ControlNet (Advanced)"}}, "312": {"inputs": {"control_net_name": "controlnet_openposeXL_sdxl.safetensors"}, "class_type": "ControlNetLoader", "_meta": {"title": "Load ControlNet Model"}}}'
)


def import_custom_nodes() -> None:
    """Find all custom nodes in the custom_nodes folder and add those node objects to NODE_CLASS_MAPPINGS

    This function sets up a new asyncio event loop, initializes the PromptServer,
    creates a PromptQueue, and initializes the custom nodes.
    """
    import asyncio
    import execution
    from nodes import init_custom_nodes
    import server

    # Creating a new event loop and setting it as the default loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Creating an instance of PromptServer with the loop
    server_instance = server.PromptServer(loop)
    execution.PromptQueue(server_instance)

    # Initializing custom nodes
    init_custom_nodes()


_custom_nodes_imported = False
_custom_path_added = False


def main(*func_args, **func_kwargs):
    global args, _custom_nodes_imported, _custom_path_added
    if __name__ == "__main__":
        if args is None:
            args = parser.parse_args()
    else:
        defaults = dict(
            (arg, parser.get_default(arg))
            for arg in ["queue_size", "comfyui_directory", "output", "disable_metadata"]
        )
        ordered_args = dict(zip([], func_args))

        all_args = dict()
        all_args.update(defaults)
        all_args.update(ordered_args)
        all_args.update(func_kwargs)

        args = argparse.Namespace(**all_args)

    with ctx:
        if not _custom_path_added:
            add_comfyui_directory_to_sys_path()
            add_extra_model_paths()

            _custom_path_added = True

        if not _custom_nodes_imported:
            import_custom_nodes()

            _custom_nodes_imported = True

        from nodes import NODE_CLASS_MAPPINGS

    with torch.inference_mode(), ctx:
        vhs_loadvideo = NODE_CLASS_MAPPINGS["VHS_LoadVideo"]()
        vhs_loadvideo_15 = vhs_loadvideo.load_video(
            video=args.source_video,
            force_rate=8,
            force_size="Disabled",
            custom_width=1024,
            custom_height=1024,
            frame_load_cap=args.frame_load_cap,
            skip_first_frames=0,
            select_every_nth=1,
        )

        vaeloader = NODE_CLASS_MAPPINGS["VAELoader"]()
        vaeloader_97 = vaeloader.load_vae(
            vae_name="1.5\\vae-ft-mse-840000-ema-pruned.safetensors"
        )

        checkpointloadersimple = NODE_CLASS_MAPPINGS["CheckpointLoaderSimple"]()
        checkpointloadersimple_127 = checkpointloadersimple.load_checkpoint(
            ckpt_name="photonLCM_v10.safetensors"
        )

        checkpointloadersimple_143 = checkpointloadersimple.load_checkpoint(
            ckpt_name="albedobaseXL_v21.safetensors"
        )

        cliptextencode = NODE_CLASS_MAPPINGS["CLIPTextEncode"]()
        cliptextencode_145 = cliptextencode.encode(
            text=args.positive_prompt,
            clip=get_value_at_index(checkpointloadersimple_143, 1),
        )

        cliptextencode_146 = cliptextencode.encode(
            text=args.negative_prompt,
            clip=get_value_at_index(checkpointloadersimple_143, 1),
        )

        loadimage = NODE_CLASS_MAPPINGS["LoadImage"]()
        loadimage_150 = loadimage.load_image(image=args.ref_image)

        instantidmodelloader = NODE_CLASS_MAPPINGS["InstantIDModelLoader"]()
        instantidmodelloader_153 = instantidmodelloader.load_model(
            instantid_file="ip-adapter.bin"
        )

        instantidfaceanalysis = NODE_CLASS_MAPPINGS["InstantIDFaceAnalysis"]()
        instantidfaceanalysis_154 = instantidfaceanalysis.load_insight_face(
            provider="CUDA"
        )

        controlnetloader = NODE_CLASS_MAPPINGS["ControlNetLoader"]()
        controlnetloader_155 = controlnetloader.load_controlnet(
            control_net_name="InstantID IdentityNet.safetensors"
        )

        clipvisionloader = NODE_CLASS_MAPPINGS["CLIPVisionLoader"]()
        clipvisionloader_195 = clipvisionloader.load_clip(
            clip_name="IPAdapter image_encoder_sd15.safetensors"
        )

        ipadaptermodelloader = NODE_CLASS_MAPPINGS["IPAdapterModelLoader"]()
        ipadaptermodelloader_197 = ipadaptermodelloader.load_ipadapter_model(
            ipadapter_file="ip-adapter-plus-face_sdxl_vit-h.safetensors"
        )

        loadimage_293 = loadimage.load_image(image="man_full_shot_origin.jpeg")

        controlnetloader_312 = controlnetloader.load_controlnet(
            control_net_name="controlnet_openposeXL_sdxl.safetensors"
        )

        champ_model_loader = NODE_CLASS_MAPPINGS["champ_model_loader"]()
        imagescaletototalpixels = NODE_CLASS_MAPPINGS["ImageScaleToTotalPixels"]()
        getimagesize = NODE_CLASS_MAPPINGS["GetImageSize+"]()
        image_crop_face = NODE_CLASS_MAPPINGS["Image Crop Face"]()
        imageresize = NODE_CLASS_MAPPINGS["ImageResize+"]()
        facekeypointspreprocessor = NODE_CLASS_MAPPINGS["FaceKeypointsPreprocessor"]()
        applyinstantid = NODE_CLASS_MAPPINGS["ApplyInstantID"]()
        ipadapteradvanced = NODE_CLASS_MAPPINGS["IPAdapterAdvanced"]()
        rescalecfg = NODE_CLASS_MAPPINGS["RescaleCFG"]()
        get_image_size = NODE_CLASS_MAPPINGS["Get image size"]()
        dwpreprocessor = NODE_CLASS_MAPPINGS["DWPreprocessor"]()
        controlnetapplyadvanced = NODE_CLASS_MAPPINGS["ControlNetApplyAdvanced"]()
        emptylatentimage = NODE_CLASS_MAPPINGS["EmptyLatentImage"]()
        ksampler = NODE_CLASS_MAPPINGS["KSampler"]()
        vaedecode = NODE_CLASS_MAPPINGS["VAEDecode"]()
        reactorfaceswap = NODE_CLASS_MAPPINGS["ReActorFaceSwap"]()
        pixelperfectresolution = NODE_CLASS_MAPPINGS["PixelPerfectResolution"]()
        depthanythingpreprocessor = NODE_CLASS_MAPPINGS["DepthAnythingPreprocessor"]()
        emptyimage = NODE_CLASS_MAPPINGS["EmptyImage"]()
        imagecompositemasked = NODE_CLASS_MAPPINGS["ImageCompositeMasked"]()
        dsine_normalmappreprocessor = NODE_CLASS_MAPPINGS[
            "DSINE-NormalMapPreprocessor"
        ]()
        denseposepreprocessor = NODE_CLASS_MAPPINGS["DensePosePreprocessor"]()
        champ_sampler = NODE_CLASS_MAPPINGS["champ_sampler"]()
        vhs_videocombine = NODE_CLASS_MAPPINGS["VHS_VideoCombine"]()
        for q in range(args.queue_size):
            champ_model_loader_10 = champ_model_loader.loadmodel(
                diffusion_dtype="auto",
                vae_dtype="auto",
                model=get_value_at_index(checkpointloadersimple_127, 0),
                vae=get_value_at_index(vaeloader_97, 0),
            )

            imagescaletototalpixels_107 = imagescaletototalpixels.upscale(
                upscale_method="lanczos",
                megapixels=0.26,
                image=get_value_at_index(vhs_loadvideo_15, 0),
            )

            getimagesize_108 = getimagesize.execute(
                image=get_value_at_index(imagescaletototalpixels_107, 0)
            )

            image_crop_face_149 = image_crop_face.image_crop_face(
                crop_padding_factor=0.25,
                cascade_xml="lbpcascade_animeface.xml",
                image=get_value_at_index(loadimage_150, 0),
            )

            imageresize_156 = imageresize.execute(
                width=1024,
                height=1024,
                interpolation="lanczos",
                keep_proportion=False,
                condition="always",
                multiple_of=0,
                image=get_value_at_index(image_crop_face_149, 0),
            )

            facekeypointspreprocessor_215 = facekeypointspreprocessor.preprocess_image(
                faceanalysis=get_value_at_index(instantidfaceanalysis_154, 0),
                image=get_value_at_index(loadimage_293, 0),
            )

            applyinstantid_224 = applyinstantid.apply_instantid(
                weight=0.6,
                start_at=0,
                end_at=1,
                instantid=get_value_at_index(instantidmodelloader_153, 0),
                insightface=get_value_at_index(instantidfaceanalysis_154, 0),
                control_net=get_value_at_index(controlnetloader_155, 0),
                image=get_value_at_index(imageresize_156, 0),
                model=get_value_at_index(checkpointloadersimple_143, 0),
                positive=get_value_at_index(cliptextencode_145, 0),
                negative=get_value_at_index(cliptextencode_146, 0),
                image_kps=get_value_at_index(facekeypointspreprocessor_215, 0),
            )

            ipadapteradvanced_194 = ipadapteradvanced.apply_ipadapter(
                weight=0.3,
                weight_type="linear",
                combine_embeds="concat",
                start_at=0,
                end_at=1,
                embeds_scaling="V only",
                model=get_value_at_index(applyinstantid_224, 0),
                ipadapter=get_value_at_index(ipadaptermodelloader_197, 0),
                image=get_value_at_index(imageresize_156, 0),
                clip_vision=get_value_at_index(clipvisionloader_195, 0),
            )

            rescalecfg_203 = rescalecfg.patch(
                multiplier=0.7, model=get_value_at_index(ipadapteradvanced_194, 0)
            )

            get_image_size_227 = get_image_size.get_size(
                image=get_value_at_index(loadimage_150, 0)
            )

            dwpreprocessor_307 = dwpreprocessor.estimate_pose(
                detect_hand="enable",
                detect_body="enable",
                detect_face="enable",
                resolution=get_value_at_index(get_image_size_227, 0),
                bbox_detector="yolox_l.onnx",
                pose_estimator="dw-ll_ucoco_384_bs5.torchscript.pt",
                image=get_value_at_index(loadimage_293, 0),
            )

            controlnetapplyadvanced_311 = controlnetapplyadvanced.apply_controlnet(
                strength=0.3,
                start_percent=0,
                end_percent=1,
                positive=get_value_at_index(applyinstantid_224, 1),
                negative=get_value_at_index(applyinstantid_224, 2),
                control_net=get_value_at_index(controlnetloader_312, 0),
                image=get_value_at_index(dwpreprocessor_307, 0),
            )

            emptylatentimage_144 = emptylatentimage.generate(
                width=get_value_at_index(get_image_size_227, 0),
                height=get_value_at_index(get_image_size_227, 1),
                batch_size=1,
            )

            ksampler_142 = ksampler.sample(
                seed=random.randint(1, 2**64),
                steps=30,
                cfg=3,
                sampler_name="dpmpp_2m",
                scheduler="karras",
                denoise=1,
                model=get_value_at_index(rescalecfg_203, 0),
                positive=get_value_at_index(controlnetapplyadvanced_311, 0),
                negative=get_value_at_index(controlnetapplyadvanced_311, 1),
                latent_image=get_value_at_index(emptylatentimage_144, 0),
            )

            vaedecode_147 = vaedecode.decode(
                samples=get_value_at_index(ksampler_142, 0),
                vae=get_value_at_index(checkpointloadersimple_143, 2),
            )

            reactorfaceswap_233 = reactorfaceswap.execute(
                enabled=True,
                swap_model="inswapper_128.onnx",
                facedetection="retinaface_resnet50",
                face_restore_model="codeformer-v0.1.0.pth",
                face_restore_visibility=1,
                codeformer_weight=0.5,
                detect_gender_input="no",
                detect_gender_source="no",
                input_faces_index="0",
                source_faces_index="0",
                console_log_level=1,
                input_image=get_value_at_index(vaedecode_147, 0),
                source_image=get_value_at_index(loadimage_150, 0),
            )

            imageresize_20 = imageresize.execute(
                width=get_value_at_index(getimagesize_108, 0),
                height=get_value_at_index(getimagesize_108, 1),
                interpolation="lanczos",
                keep_proportion=True,
                condition="always",
                multiple_of=64,
                image=get_value_at_index(reactorfaceswap_233, 0),
            )

            pixelperfectresolution_103 = pixelperfectresolution.execute(
                image_gen_width=get_value_at_index(getimagesize_108, 0),
                image_gen_height=get_value_at_index(getimagesize_108, 1),
                resize_mode="Just Resize",
                original_image=get_value_at_index(imagescaletototalpixels_107, 0),
            )

            imageresize_23 = imageresize.execute(
                width=get_value_at_index(getimagesize_108, 0),
                height=get_value_at_index(getimagesize_108, 1),
                interpolation="lanczos",
                keep_proportion=False,
                condition="always",
                multiple_of=64,
                image=get_value_at_index(vhs_loadvideo_15, 0),
            )

            depthanythingpreprocessor_62 = depthanythingpreprocessor.execute(
                ckpt_name="depth_anything_vitl14.pth",
                resolution=get_value_at_index(pixelperfectresolution_103, 0),
                image=get_value_at_index(imageresize_23, 0),
            )

            emptyimage_28 = emptyimage.generate(
                width=get_value_at_index(getimagesize_108, 1),
                height=get_value_at_index(getimagesize_108, 1),
                batch_size=1,
                color=0,
            )

            imagecompositemasked_63 = imagecompositemasked.composite(
                x=0,
                y=0,
                resize_source=True,
                destination=get_value_at_index(depthanythingpreprocessor_62, 0),
                source=get_value_at_index(emptyimage_28, 0),
            )

            dsine_normalmappreprocessor_56 = dsine_normalmappreprocessor.execute(
                fov=60,
                iterations=5,
                resolution=get_value_at_index(pixelperfectresolution_103, 0),
                image=get_value_at_index(imageresize_23, 0),
            )

            imagecompositemasked_59 = imagecompositemasked.composite(
                x=0,
                y=0,
                resize_source=True,
                destination=get_value_at_index(dsine_normalmappreprocessor_56, 0),
                source=get_value_at_index(emptyimage_28, 0),
            )

            denseposepreprocessor_65 = denseposepreprocessor.execute(
                model="densepose_r101_fpn_dl.torchscript",
                cmap="Viridis (MagicAnimate)",
                resolution=get_value_at_index(pixelperfectresolution_103, 0),
                image=get_value_at_index(imageresize_23, 0),
            )

            imagecompositemasked_66 = imagecompositemasked.composite(
                x=0,
                y=0,
                resize_source=True,
                destination=get_value_at_index(denseposepreprocessor_65, 0),
                source=get_value_at_index(emptyimage_28, 0),
            )

            pixelperfectresolution_22 = pixelperfectresolution.execute(
                image_gen_width=get_value_at_index(getimagesize_108, 0),
                image_gen_height=get_value_at_index(getimagesize_108, 1),
                resize_mode="Resize and Fill",
                original_image=get_value_at_index(imageresize_23, 0),
            )

            dwpreprocessor_18 = dwpreprocessor.estimate_pose(
                detect_hand="enable",
                detect_body="enable",
                detect_face="enable",
                resolution=get_value_at_index(pixelperfectresolution_22, 0),
                bbox_detector="yolox_l.onnx",
                pose_estimator="dw-ll_ucoco_384_bs5.torchscript.pt",
                image=get_value_at_index(imageresize_23, 0),
            )

            champ_sampler_12 = champ_sampler.process(
                width=get_value_at_index(getimagesize_108, 0),
                height=get_value_at_index(getimagesize_108, 1),
                steps=36,
                guidance_scale=2,
                frames=get_value_at_index(vhs_loadvideo_15, 1),
                seed=random.randint(1, 2**64),
                keep_model_loaded=True,
                scheduler="DDPMScheduler",
                champ_model=get_value_at_index(champ_model_loader_10, 0),
                champ_vae=get_value_at_index(champ_model_loader_10, 1),
                champ_encoder=get_value_at_index(champ_model_loader_10, 2),
                image=get_value_at_index(imageresize_20, 0),
                depth_tensors=get_value_at_index(imagecompositemasked_63, 0),
                normal_tensors=get_value_at_index(imagecompositemasked_59, 0),
                semantic_tensors=get_value_at_index(imagecompositemasked_66, 0),
                dwpose_tensors=get_value_at_index(dwpreprocessor_18, 0),
            )

            reactorfaceswap_101 = reactorfaceswap.execute(
                enabled=True,
                swap_model="inswapper_128.onnx",
                facedetection="retinaface_resnet50",
                face_restore_model="codeformer-v0.1.0.pth",
                face_restore_visibility=1,
                codeformer_weight=0.5,
                detect_gender_input="female",
                detect_gender_source="female",
                input_faces_index="0",
                source_faces_index="0",
                console_log_level=1,
                input_image=get_value_at_index(champ_sampler_12, 0),
                source_image=get_value_at_index(reactorfaceswap_233, 0),
            )

            # vhs_videocombine_102 = vhs_videocombine.combine_video(
            #     frame_rate=8,
            #     loop_count=0,
            #     filename_prefix="ComfyUI_swap",
            #     format="video/h264-mp4",
            #     pingpong=False,
            #     save_output=False,
            #     images=get_value_at_index(reactorfaceswap_101, 0),
            #     audio=get_value_at_index(vhs_loadvideo_15, 2),
            #     unique_id=8408973814857971085,
            #     prompt=PROMPT_DATA,
            # )
            
            def tensor_to_bytes(tensor):
                tensor = tensor.cpu().numpy() * 255
                return np.clip(tensor, 0, 255).astype(np.uint8)
            
            def ffmpeg_process(args, video_metadata, file_path):
                frame_data = yield
                with subprocess.Popen(args, stderr=subprocess.PIPE, stdin=subprocess.PIPE) as proc:
                    try:
                        while frame_data is not None:
                            proc.stdin.write(frame_data)
                            frame_data = yield
                        proc.stdin.flush()
                        proc.stdin.close()
                        proc.stderr.read()
                    except BrokenPipeError:
                        pass
            
            def save_video(frames, frame_rate, output_dir, filename_prefix="output_video"):
                os.makedirs(output_dir, exist_ok=True)
            
                metadata = PngImagePlugin.PngInfo()
                metadata.add_text("CreationTime", datetime.now().isoformat(" ")[:19])
            
                video_metadata = {"CreationTime": datetime.now().isoformat(" ")[:19]}
            
                counter = 1
                file = f"{filename_prefix}_{counter:05}.mp4"
                file_path = os.path.join(output_dir, file)
            
                dimensions = f"{frames[0].shape[1]}x{frames[0].shape[0]}"
                args = [
                    "ffmpeg", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                    "-s", dimensions, "-r", str(frame_rate), "-i", "-",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", file_path
                ]
            
                process = ffmpeg_process(args, video_metadata, file_path)
                process.send(None)
            
                for frame in frames:
                    process.send(tensor_to_bytes(frame).tobytes())
                try:
                    process.send(None)
                except StopIteration:
                    pass
            
                return file_path
            
            # 예시로 frame들을 생성합니다. 실제로는 입력 프레임들을 여기에 넣어야 합니다.
            frames = get_value_at_index(reactorfaceswap_101, 0)
            frame_rate = args.output_dir_frame_rate
            output_dir = args.output_dir
            
            video_path = save_video(frames, frame_rate, output_dir)
            print(f"Video saved to {video_path}")


if __name__ == "__main__":
    main()
