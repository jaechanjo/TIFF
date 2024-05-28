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
from PIL import Image
import datetime


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
    if args.output is None:
        return cls

    from PIL import Image, ImageOps, ImageSequence
    from PIL.PngImagePlugin import PngInfo

    import numpy as np

    class WrappedSaveImage(cls):
        counter = 0

        def save_images(
            self, images, filename_prefix="ComfyUI", prompt=None, extra_pnginfo=None
        ):
            if args.output is None:
                return super().save_images(
                    images, filename_prefix, prompt, extra_pnginfo
                )
            else:
                if len(images) > 1 and args.output == "-":
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

                    if args.output == "-":
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
                            if os.path.isdir(args.output):
                                subfolder = args.output
                                file = "output.png"
                            else:
                                subfolder, file = os.path.split(args.output)
                                if subfolder == "":
                                    subfolder = os.getcwd()
                        else:
                            if os.path.isdir(args.output):
                                subfolder = args.output
                                file = filename_prefix
                            else:
                                subfolder, file = os.path.split(args.output)

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
    "--queue-size",
    "-q",
    type=int,
    default=1,
    help="How many times the workflow will be executed (default: 1)",
)

parser.add_argument(
    "--comfyui-directory",
    "-c",
    default=None,
    help="Where to look for ComfyUI (default: current directory)",
)

parser.add_argument(
    "--output",
    "-o",
    default=None,
    help="The location to save the output image. Either a file path, a directory, or - for stdout (default: the ComfyUI output directory)",
)

parser.add_argument(
    "--disable-metadata",
    action="store_true",
    help="Disables writing workflow metadata to the outputs",
)

# 추가된 인자들
parser.add_argument(
    "--video",
    required=True,
    help="Path to the source motion video",
)

parser.add_argument(
    "--frame-load-cap",
    type=int,
    default=0,
    help="Number of frames to load from the video (default: 0, meaning all frames)",
)

parser.add_argument(
    "--positive-prompt",
    type=str,
    required=True,
    help="Positive text prompt for the model",
)
parser.add_argument(
    "--negative-prompt",
    type=str,
    required=True,
    help="Negative text prompt for the model",
)
parser.add_argument(
    "--ref-image",
    required=True,
    help="Path to the reference image for the person",
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
if args is not None and args.output is not None and args.output == "-":
    ctx = contextlib.redirect_stdout(sys.stderr)
else:
    ctx = contextlib.nullcontext()

PROMPT_DATA = {
    "10": {
        "inputs": {
            "diffusion_dtype": "auto",
            "vae_dtype": "auto",
            "model": ["51", 0],
            "vae": ["43", 0]
        },
        "class_type": "champ_model_loader",
        "_meta": {"title": "champ_model_loader"}
    },
    "11": {
        "inputs": {
            "width": ["48", 0],
            "height": ["48", 1],
            "steps": 36,
            "guidance_scale": 2,
            "frames": ["12", 1],
            "seed": 0,
            "keep_model_loaded": True,
            "scheduler": "DDPMScheduler",
            "champ_model": ["10", 0],
            "champ_vae": ["10", 1],
            "champ_encoder": ["10", 2],
            "image": ["14", 0],
            "depth_tensors": ["31", 0],
            "normal_tensors": ["28", 0],
            "semantic_tensors": ["33", 0],
            "dwpose_tensors": ["13", 0]
        },
        "class_type": "champ_sampler",
        "_meta": {"title": "champ_sampler"}
    },
    "12": {
        "inputs": {
            "video": args.video,  # 소스 모션 비디오 경로
            "force_rate": 8,
            "force_size": "Disabled",
            "custom_width": 1024,
            "custom_height": 1024,
            "frame_load_cap": args.frame_load_cap,  # 비디오의 처음부터 사용할 프레임 수
            "skip_first_frames": 0,
            "select_every_nth": 1
        },
        "class_type": "VHS_LoadVideo",
        "_meta": {"title": "Load Video (Upload) 🎥🆕🆗🆂"}
    },
    "13": {
        "inputs": {
            "detect_hand": "enable",
            "detect_body": "enable",
            "detect_face": "enable",
            "resolution": ["15", 0],
            "bbox_detector": "yolox_l.onnx",
            "pose_estimator": "dw-ll_ucoco_384_bs5.torchscript.pt",
            "image": ["18", 0]
        },
        "class_type": "DWPreprocessor",
        "_meta": {"title": "DWPose Estimator"}
    },
    "14": {
        "inputs": {
            "width": ["48", 0],
            "height": ["48", 1],
            "interpolation": "lanczos",
            "keep_proportion": True,
            "condition": "always",
            "multiple_of": 64,
            "image": ["73", 0]
        },
        "class_type": "ImageResize+",
        "_meta": {"title": "🔧 Image Resize"}
    },
    "15": {
        "inputs": {
            "image_gen_width": ["48", 0],
            "image_gen_height": ["48", 1],
            "resize_mode": "Resize and Fill",
            "original_image": ["18", 0]
        },
        "class_type": "PixelPerfectResolution",
        "_meta": {"title": "Pixel Perfect Resolution"}
    },
    "16": {
        "inputs": {
            "width": ["48", 0],
            "height": ["48", 1],
            "interpolation": "lanczos",
            "keep_proportion": False,
            "condition": "always",
            "multiple_of": 64,
            "image": ["12", 0]
        },
        "class_type": "ImageResize+",
        "_meta": {"title": "🔧 Image Resize"}
    },
    "17": {
        "inputs": {
            "device": "auto",
            "image": ["19", 0]
        },
        "class_type": "BiRefNet",
        "_meta": {"title": "BiRefNet Segmentation"}
    },
    "18": {
        "inputs": {
            "x": 0,
            "y": 0,
            "resize_source": True,
            "destination": ["16", 0],
            "source": ["20", 0],
            "mask": ["29", 0]
        },
        "class_type": "ImageCompositeMasked",
        "_meta": {"title": "ImageCompositeMasked"}
    },
    "19": {
        "inputs": {
            "image": ["16", 0]
        },
        "class_type": "ImpactImageBatchToImageList",
        "_meta": {"title": "Image batch to Image List"}
    },
    "20": {
        "inputs": {
            "width": ["48", 1],
            "height": ["48", 1],
            "batch_size": 1,
            "color": 0
        },
        "class_type": "EmptyImage",
        "_meta": {"title": "EmptyImage"}
    },
    "21": {
        "inputs": {
            "mask": ["17", 0]
        },
        "class_type": "MaskListToMaskBatch",
        "_meta": {"title": "Mask List to Masks"}
    },
    "22": {
        "inputs": {
            "x": 0,
            "y": 0,
            "resize_source": True,
            "destination": ["14", 0],
            "source": ["24", 0],
            "mask": ["25", 0]
        },
        "class_type": "ImageCompositeMasked",
        "_meta": {"title": "ImageCompositeMasked"}
    },
    "23": {
        "inputs": {
            "device": "cuda:0",
            "image": ["73", 0]
        },
        "class_type": "BiRefNet",
        "_meta": {"title": "BiRefNet Segmentation"}
    },
    "24": {
        "inputs": {
            "width": ["48", 0],
            "height": ["48", 1],
            "batch_size": 1,
            "color": 0
        },
        "class_type": "EmptyImage",
        "_meta": {"title": "EmptyImage"}
    },
    "25": {
        "inputs": {
            "expand": 2,
            "incremental_expandrate": 0,
            "tapered_corners": True,
            "flip_input": True,
            "blur_radius": 0.2,
            "lerp_alpha": 1,
            "decay_factor": 1,
            "fill_holes": False,
            "mask": ["23", 0]
        },
        "class_type": "GrowMaskWithBlur",
        "_meta": {"title": "GrowMaskWithBlur"}
    },
    "26": {
        "inputs": {
            "fov": 60,
            "iterations": 5,
            "resolution": ["46", 0],
            "image": ["18", 0]
        },
        "class_type": "DSINE-NormalMapPreprocessor",
        "_meta": {"title": "DSINE Normal Map"}
    },
    "27": {
        "inputs": {
            "images": ["18", 0]
        },
        "class_type": "PreviewImage",
        "_meta": {"title": "Preview Image"}
    },
    "28": {
        "inputs": {
            "x": 0,
            "y": 0,
            "resize_source": True,
            "destination": ["26", 0],
            "source": ["20", 0],
            "mask": ["29", 0]
        },
        "class_type": "ImageCompositeMasked",
        "_meta": {"title": "ImageCompositeMasked"}
    },
    "29": {
        "inputs": {
            "mask": ["21", 0]
        },
        "class_type": "InvertMask",
        "_meta": {"title": "InvertMask"}
    },
    "30": {
        "inputs": {
            "ckpt_name": "depth_anything_vitl14.pth",
            "resolution": ["46", 0],
            "image": ["18", 0]
        },
        "class_type": "DepthAnythingPreprocessor",
        "_meta": {"title": "Depth Anything"}
    },
    "31": {
        "inputs": {
            "x": 0,
            "y": 0,
            "resize_source": True,
            "destination": ["30", 0],
            "source": ["20", 0],
            "mask": ["29", 0]
        },
        "class_type": "ImageCompositeMasked",
        "_meta": {"title": "ImageCompositeMasked"}
    },
    "32": {
        "inputs": {
            "model": "densepose_r101_fpn_dl.torchscript",
            "cmap": "Viridis (MagicAnimate)",
            "resolution": ["46", 0],
            "image": ["18", 0]
        },
        "class_type": "DensePosePreprocessor",
        "_meta": {"title": "DensePose Estimator"}
    },
    "33": {
        "inputs": {
            "x": 0,
            "y": 0,
            "resize_source": True,
            "destination": ["32", 0],
            "source": ["20", 0],
            "mask": ["29", 0]
        },
        "class_type": "ImageCompositeMasked",
        "_meta": {"title": "ImageCompositeMasked"}
    },
    "34": {
        "inputs": {
            "mask_threshold": 250,
            "gaussblur_radius": 8,
            "invert_mask": False,
            "images": ["59", 0],
            "masks": ["25", 1]
        },
        "class_type": "LamaRemover",
        "_meta": {"title": "Big lama Remover"}
    },
    "35": {
        "inputs": {
            "x": 0,
            "y": 0,
            "resize_source": True,
            "destination": ["11", 0],
            "source": ["41", 0],
            "mask": ["40", 0]
        },
        "class_type": "ImageCompositeMasked",
        "_meta": {"title": "ImageCompositeMasked"}
    },
    "36": {
        "inputs": {
            "device": "cuda:0",
            "image": ["38", 0]
        },
        "class_type": "BiRefNet",
        "_meta": {"title": "BiRefNet Segmentation"}
    },
    "37": {
        "inputs": {
            "mask": ["39", 0]
        },
        "class_type": "InvertMask",
        "_meta": {"title": "InvertMask"}
    },
    "38": {
        "inputs": {
            "image": ["11", 0]
        },
        "class_type": "ImpactImageBatchToImageList",
        "_meta": {"title": "Image batch to Image List"}
    },
    "39": {
        "inputs": {
            "mask": ["36", 0]
        },
        "class_type": "MaskListToMaskBatch",
        "_meta": {"title": "Mask List to Masks"}
    },
    "40": {
        "inputs": {
            "expand": 1,
            "incremental_expandrate": 0,
            "tapered_corners": True,
            "flip_input": False,
            "blur_radius": 0.1,
            "lerp_alpha": 1,
            "decay_factor": 1,
            "fill_holes": False,
            "mask": ["37", 0]
        },
        "class_type": "GrowMaskWithBlur",
        "_meta": {"title": "GrowMaskWithBlur"}
    },
    "41": {
        "inputs": {
            "mask_threshold": 250,
            "gaussblur_radius": 8,
            "invert_mask": False,
            "images": ["16", 0],
            "masks": ["21", 0]
        },
        "class_type": "LamaRemover",
        "_meta": {"title": "Big lama Remover"}
    },
    "42": {
        "inputs": {
            "images": ["41", 0]
        },
        "class_type": "PreviewImage",
        "_meta": {"title": "Preview Image"}
    },
    "43": {
        "inputs": {
            "vae_name": "1.5\\vae-ft-mse-840000-ema-pruned.safetensors"
        },
        "class_type": "VAELoader",
        "_meta": {"title": "Load VAE"}
    },
    "44": {
        "inputs": {
            "enabled": True,
            "swap_model": "inswapper_128.onnx",
            "facedetection": "retinaface_resnet50",
            "face_restore_model": "codeformer-v0.1.0.pth",
            "face_restore_visibility": 1,
            "codeformer_weight": 0.5,
            "detect_gender_input": "female",
            "detect_gender_source": "female",
            "input_faces_index": "0",
            "source_faces_index": "0",
            "console_log_level": 1,
            "input_image": ["35", 0],
            "source_image": ["73", 0]
        },
        "class_type": "ReActorFaceSwap",
        "_meta": {"title": "ReActor - Fast Face Swap"}
    },
    "45": {
        "inputs": {
            "frame_rate": 8,
            "loop_count": 0,
            "filename_prefix": "ComfyUI_swap",
            "format": "video/h264-mp4",
            "pix_fmt": "yuv420p",
            "crf": 19,
            "save_metadata": False,
            "pingpong": False,
            "save_output": False,
            "images": ["44", 0],
            "audio": ["12", 2]
        },
        "class_type": "VHS_VideoCombine",
        "_meta": {"title": "Video Combine 🎥🆕🆗🆂"}
    },
    "46": {
        "inputs": {
            "image_gen_width": ["48", 0],
            "image_gen_height": ["48", 1],
            "resize_mode": "Just Resize",
            "original_image": ["47", 0]
        },
        "class_type": "PixelPerfectResolution",
        "_meta": {"title": "Pixel Perfect Resolution"}
    },
    "47": {
        "inputs": {
            "upscale_method": "lanczos",
            "megapixels": 0.26,
            "image": ["12", 0]
        },
        "class_type": "ImageScaleToTotalPixels",
        "_meta": {"title": "ImageScaleToTotalPixels"}
    },
    "48": {
        "inputs": {
            "image": ["47", 0]
        },
        "class_type": "GetImageSize+",
        "_meta": {"title": "🔧 Get Image Size"}
    },
    "51": {
        "inputs": {
            "ckpt_name": "photonLCM_v10.safetensors"
        },
        "class_type": "CheckpointLoaderSimple",
        "_meta": {"title": "Load Checkpoint"}
    },
    "52": {
        "inputs": {
            "seed": 0,
            "steps": 30,
            "cfg": 3,
            "sampler_name": "dpmpp_2m",
            "scheduler": "karras",
            "denoise": 1,
            "model": ["67", 0],
            "positive": ["77", 0],
            "negative": ["77", 1],
            "latent_image": ["54", 0]
        },
        "class_type": "KSampler",
        "_meta": {"title": "KSampler"}
    },
    "53": {
        "inputs": {
            "ckpt_name": "albedobaseXL_v21.safetensors"
        },
        "class_type": "CheckpointLoaderSimple",
        "_meta": {"title": "Load Checkpoint"}
    },
    "54": {
        "inputs": {
            "width": ["72", 0],
            "height": ["72", 1],
            "batch_size": 1
        },
        "class_type": "EmptyLatentImage",
        "_meta": {"title": "Empty Latent Image"}
    },
    "55": {
        "inputs": {
            "text": args.positive_prompt,  # positive_prompt 인자
            "clip": ["53", 1]
        },
        "class_type": "CLIPTextEncode",
        "_meta": {"title": "CLIP Text Encode (Prompt)"}
    },
    "56": {
        "inputs": {
            "text": args.negative_prompt,  # negative_prompt 인자
            "clip": ["53", 1]
        },
        "class_type": "CLIPTextEncode",
        "_meta": {"title": "CLIP Text Encode (Prompt)"}
    },
    "57": {
        "inputs": {
            "samples": ["52", 0],
            "vae": ["53", 2]
        },
        "class_type": "VAEDecode",
        "_meta": {"title": "VAE Decode"}
    },
    "58": {
        "inputs": {
            "crop_padding_factor": 0.25,
            "cascade_xml": "lbpcascade_animeface.xml",
            "image": ["59", 0]
        },
        "class_type": "Image Crop Face",
        "_meta": {"title": "Image Crop Face"}
    },
    "59": {
        "inputs": {
            "image": args.ref_image,  # 참조 인물 이미지 경로
            "upload": "image"
        },
        "class_type": "LoadImage",
        "_meta": {"title": "Load Image"}
    },
    "60": {
        "inputs": {
            "instantid_file": "ip-adapter.bin"
        },
        "class_type": "InstantIDModelLoader",
        "_meta": {"title": "Load InstantID Model"}
    },
    "61": {
        "inputs": {
            "provider": "CUDA"
        },
        "class_type": "InstantIDFaceAnalysis",
        "_meta": {"title": "InstantID Face Analysis"}
    },
    "62": {
        "inputs": {
            "control_net_name": "InstantID IdentityNet.safetensors"
        },
        "class_type": "ControlNetLoader",
        "_meta": {"title": "Load ControlNet Model"}
    },
    "63": {
        "inputs": {
            "width": 1024,
            "height": 1024,
            "interpolation": "lanczos",
            "keep_proportion": False,
            "condition": "always",
            "multiple_of": 0,
            "image": ["58", 0]
        },
        "class_type": "ImageResize+",
        "_meta": {"title": "🔧 Image Resize"}
    },
    "64": {
        "inputs": {
            "weight": 0.3,
            "weight_type": "linear",
            "combine_embeds": "concat",
            "start_at": 0,
            "end_at": 1,
            "embeds_scaling": "V only",
            "model": ["71", 0],
            "ipadapter": ["66", 0],
            "image": ["63", 0],
            "clip_vision": ["65", 0]
        },
        "class_type": "IPAdapterAdvanced",
        "_meta": {"title": "IPAdapter Advanced"}
    },
    "65": {
        "inputs": {
            "clip_name": "IPAdapter image_encoder_sd15.safetensors"
        },
        "class_type": "CLIPVisionLoader",
        "_meta": {"title": "Load CLIP Vision"}
    },
    "66": {
        "inputs": {
            "ipadapter_file": "ip-adapter-plus-face_sdxl_vit-h.safetensors"
        },
        "class_type": "IPAdapterModelLoader",
        "_meta": {"title": "IPAdapter Model Loader"}
    },
    "67": {
        "inputs": {
            "multiplier": 0.7,
            "model": ["64", 0]
        },
        "class_type": "RescaleCFG",
        "_meta": {"title": "RescaleCFG"}
    },
    "68": {
        "inputs": {
            "faceanalysis": ["61", 0],
            "image": ["75", 0]
        },
        "class_type": "FaceKeypointsPreprocessor",
        "_meta": {"title": "Face Keypoints Preprocessor"}
    },
    "69": {
        "inputs": {
            "images": ["68", 0]
        },
        "class_type": "PreviewImage",
        "_meta": {"title": "Preview Image"}
    },
    "71": {
        "inputs": {
            "weight": 0.6,
            "start_at": 0,
            "end_at": 1,
            "instantid": ["60", 0],
            "insightface": ["61", 0],
            "control_net": ["62", 0],
            "image": ["63", 0],
            "model": ["53", 0],
            "positive": ["55", 0],
            "negative": ["56", 0],
            "image_kps": ["68", 0]
        },
        "class_type": "ApplyInstantID",
        "_meta": {"title": "Apply InstantID"}
    },
    "72": {
        "inputs": {
            "image": ["59", 0]
        },
        "class_type": "Get image size",
        "_meta": {"title": "Get image size"}
    },
    "73": {
        "inputs": {
            "enabled": True,
            "swap_model": "inswapper_128.onnx",
            "facedetection": "retinaface_resnet50",
            "face_restore_model": "codeformer-v0.1.0.pth",
            "face_restore_visibility": 1,
            "codeformer_weight": 0.5,
            "detect_gender_input": "no",
            "detect_gender_source": "no",
            "input_faces_index": "0",
            "source_faces_index": "0",
            "console_log_level": 1,
            "input_image": ["57", 0],
            "source_image": ["59", 0]
        },
        "class_type": "ReActorFaceSwap",
        "_meta": {"title": "ReActor - Fast Face Swap"}
    },
    "75": {
        "inputs": {
            "image": "man_full_shot_origin.jpeg",
            "upload": "image"
        },
        "class_type": "LoadImage",
        "_meta": {"title": "Load Image"}
    },
    "76": {
        "inputs": {
            "detect_hand": "enable",
            "detect_body": "enable",
            "detect_face": "enable",
            "resolution": ["72", 0],
            "bbox_detector": "yolox_l.onnx",
            "pose_estimator": "dw-ll_ucoco_384_bs5.torchscript.pt",
            "image": ["75", 0]
        },
        "class_type": "DWPreprocessor",
        "_meta": {"title": "DWPose Estimator"}
    },
    "77": {
        "inputs": {
            "strength": 0.3,
            "start_percent": 0,
            "end_percent": 1,
            "positive": ["71", 1],
            "negative": ["71", 2],
            "control_net": ["78", 0],
            "image": ["76", 0]
        },
        "class_type": "ControlNetApplyAdvanced",
        "_meta": {"title": "Apply ControlNet (Advanced)"}
    },
    "78": {
        "inputs": {
            "control_net_name": "controlnet_openposeXL_sdxl.safetensors"
        },
        "class_type": "ControlNetLoader",
        "_meta": {"title": "Load ControlNet Model"}
    },
    "79": {
        "inputs": {
            "frame_rate": 8,
            "loop_count": 0,
            "filename_prefix": "ComfyUI_swap",
            "format": "video/h264-mp4",
            "pix_fmt": "yuv420p",
            "crf": 19,
            "save_metadata": False,
            "pingpong": False,
            "save_output": True,
            "images": ["44", 0],
            "audio": ["12", 2]
        },
        "class_type": "VHS_VideoCombine",
        "_meta": {"title": "Video Combine 🎥🆕🆗🆂"}
    }
}


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
            for arg in ["queue_size", "comfyui_directory", "output", "disable_metadata", "video", "frame_load_cap", "positive_prompt", "negative_prompt", "ref_image"]
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
        vhs_loadvideo_12 = vhs_loadvideo.load_video(
            video=args.video,  # 소스 모션 비디오 경로
            force_rate=8,
            force_size="Disabled",
            custom_width=1024,
            custom_height=1024,
            frame_load_cap=args.frame_load_cap,  # 비디오의 처음부터 사용할 프레임 수
            skip_first_frames=0,
            select_every_nth=1,
        )

        vaeloader = NODE_CLASS_MAPPINGS["VAELoader"]()
        vaeloader_43 = vaeloader.load_vae(
            vae_name="1.5\\vae-ft-mse-840000-ema-pruned.safetensors"
        )

        checkpointloadersimple = NODE_CLASS_MAPPINGS["CheckpointLoaderSimple"]()
        checkpointloadersimple_51 = checkpointloadersimple.load_checkpoint(
            ckpt_name="photonLCM_v10.safetensors"
        )

        checkpointloadersimple_53 = checkpointloadersimple.load_checkpoint(
            ckpt_name="albedobaseXL_v21.safetensors"
        )

        cliptextencode = NODE_CLASS_MAPPINGS["CLIPTextEncode"]()
        cliptextencode_55 = cliptextencode.encode(
            text=args.positive_prompt,  # positive_prompt 인자
            clip=get_value_at_index(checkpointloadersimple_53, 1)
        )

        cliptextencode_56 = cliptextencode.encode(
            text=args.negative_prompt,  # negative_prompt 인자
            clip=get_value_at_index(checkpointloadersimple_53, 1)
        )

        loadimage = NODE_CLASS_MAPPINGS["LoadImage"]()
        loadimage_59 = loadimage.load_image(image=args.ref_image)  # 참조 인물 이미지 경로

        instantidmodelloader = NODE_CLASS_MAPPINGS["InstantIDModelLoader"]()
        instantidmodelloader_60 = instantidmodelloader.load_model(
            instantid_file="ip-adapter.bin"
        )

        instantidfaceanalysis = NODE_CLASS_MAPPINGS["InstantIDFaceAnalysis"]()
        instantidfaceanalysis_61 = instantidfaceanalysis.load_insight_face(
            provider="CUDA"
        )

        controlnetloader = NODE_CLASS_MAPPINGS["ControlNetLoader"]()
        controlnetloader_62 = controlnetloader.load_controlnet(
            control_net_name="InstantID IdentityNet.safetensors"
        )

        clipvisionloader = NODE_CLASS_MAPPINGS["CLIPVisionLoader"]()
        clipvisionloader_65 = clipvisionloader.load_clip(
            clip_name="IPAdapter image_encoder_sd15.safetensors"
        )

        ipadaptermodelloader = NODE_CLASS_MAPPINGS["IPAdapterModelLoader"]()
        ipadaptermodelloader_66 = ipadaptermodelloader.load_ipadapter_model(
            ipadapter_file="ip-adapter-plus-face_sdxl_vit-h.safetensors"
        )

        loadimage_75 = loadimage.load_image(image="man_full_shot_origin.jpeg")

        controlnetloader_78 = controlnetloader.load_controlnet(
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
        emptyimage = NODE_CLASS_MAPPINGS["EmptyImage"]()
        impactimagebatchtoimagelist = NODE_CLASS_MAPPINGS[
            "ImpactImageBatchToImageList"
        ]()
        birefnet = NODE_CLASS_MAPPINGS["BiRefNet"]()
        masklisttomaskbatch = NODE_CLASS_MAPPINGS["MaskListToMaskBatch"]()
        invertmask = NODE_CLASS_MAPPINGS["InvertMask"]()
        imagecompositemasked = NODE_CLASS_MAPPINGS["ImageCompositeMasked"]()
        depthanythingpreprocessor = NODE_CLASS_MAPPINGS["DepthAnythingPreprocessor"]()
        dsine_normalmappreprocessor = NODE_CLASS_MAPPINGS[
            "DSINE-NormalMapPreprocessor"
        ]()
        denseposepreprocessor = NODE_CLASS_MAPPINGS["DensePosePreprocessor"]()
        champ_sampler = NODE_CLASS_MAPPINGS["champ_sampler"]()
        growmaskwithblur = NODE_CLASS_MAPPINGS["GrowMaskWithBlur"]()
        lamaremover = NODE_CLASS_MAPPINGS["LamaRemover"]()
        vhs_videocombine = NODE_CLASS_MAPPINGS["VHS_VideoCombine"]()
        for q in range(args.queue_size):
            champ_model_loader_10 = champ_model_loader.loadmodel(
                diffusion_dtype="auto",
                vae_dtype="auto",
                model=get_value_at_index(checkpointloadersimple_51, 0),
                vae=get_value_at_index(vaeloader_43, 0),
            )

            imagescaletototalpixels_47 = imagescaletototalpixels.upscale(
                upscale_method="lanczos",
                megapixels=0.26,
                image=get_value_at_index(vhs_loadvideo_12, 0),
            )

            getimagesize_48 = getimagesize.execute(
                image=get_value_at_index(imagescaletototalpixels_47, 0)
            )

            image_crop_face_58 = image_crop_face.image_crop_face(
                crop_padding_factor=0.25,
                cascade_xml="lbpcascade_animeface.xml",
                image=get_value_at_index(loadimage_59, 0),
            )

            imageresize_63 = imageresize.execute(
                width=1024,
                height=1024,
                interpolation="lanczos",
                keep_proportion=False,
                condition="always",
                multiple_of=0,
                image=get_value_at_index(image_crop_face_58, 0),
            )

            facekeypointspreprocessor_68 = facekeypointspreprocessor.preprocess_image(
                faceanalysis=get_value_at_index(instantidfaceanalysis_61, 0),
                image=get_value_at_index(loadimage_75, 0),
            )

            applyinstantid_71 = applyinstantid.apply_instantid(
                weight=0.6,
                start_at=0,
                end_at=1,
                instantid=get_value_at_index(instantidmodelloader_60, 0),
                insightface=get_value_at_index(instantidfaceanalysis_61, 0),
                control_net=get_value_at_index(controlnetloader_62, 0),
                image=get_value_at_index(imageresize_63, 0),
                model=get_value_at_index(checkpointloadersimple_53, 0),
                positive=get_value_at_index(cliptextencode_55, 0),
                negative=get_value_at_index(cliptextencode_56, 0),
                image_kps=get_value_at_index(facekeypointspreprocessor_68, 0),
            )

            ipadapteradvanced_64 = ipadapteradvanced.apply_ipadapter(
                weight=0.3,
                weight_type="linear",
                combine_embeds="concat",
                start_at=0,
                end_at=1,
                embeds_scaling="V only",
                model=get_value_at_index(applyinstantid_71, 0),
                ipadapter=get_value_at_index(ipadaptermodelloader_66, 0),
                image=get_value_at_index(imageresize_63, 0),
                clip_vision=get_value_at_index(clipvisionloader_65, 0),
            )

            rescalecfg_67 = rescalecfg.patch(
                multiplier=0.7, model=get_value_at_index(ipadapteradvanced_64, 0)
            )

            get_image_size_72 = get_image_size.get_size(
                image=get_value_at_index(loadimage_59, 0)
            )

            dwpreprocessor_76 = dwpreprocessor.estimate_pose(
                detect_hand="enable",
                detect_body="enable",
                detect_face="enable",
                resolution=get_value_at_index(get_image_size_72, 0),
                bbox_detector="yolox_l.onnx",
                pose_estimator="dw-ll_ucoco_384_bs5.torchscript.pt",
                image=get_value_at_index(loadimage_75, 0),
            )

            controlnetapplyadvanced_77 = controlnetapplyadvanced.apply_controlnet(
                strength=0.3,
                start_percent=0,
                end_percent=1,
                positive=get_value_at_index(applyinstantid_71, 1),
                negative=get_value_at_index(applyinstantid_71, 2),
                control_net=get_value_at_index(controlnetloader_78, 0),
                image=get_value_at_index(dwpreprocessor_76, 0),
            )

            emptylatentimage_54 = emptylatentimage.generate(
                width=get_value_at_index(get_image_size_72, 0),
                height=get_value_at_index(get_image_size_72, 1),
                batch_size=1,
            )

            ksampler_52 = ksampler.sample(
                seed=random.randint(1, 2**64),
                steps=30,
                cfg=3,
                sampler_name="dpmpp_2m",
                scheduler="karras",
                denoise=1,
                model=get_value_at_index(rescalecfg_67, 0),
                positive=get_value_at_index(controlnetapplyadvanced_77, 0),
                negative=get_value_at_index(controlnetapplyadvanced_77, 1),
                latent_image=get_value_at_index(emptylatentimage_54, 0),
            )

            vaedecode_57 = vaedecode.decode(
                samples=get_value_at_index(ksampler_52, 0),
                vae=get_value_at_index(checkpointloadersimple_53, 2),
            )

            reactorfaceswap_73 = reactorfaceswap.execute(
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
                input_image=get_value_at_index(vaedecode_57, 0),
                source_image=get_value_at_index(loadimage_59, 0),
            )

            imageresize_14 = imageresize.execute(
                width=get_value_at_index(getimagesize_48, 0),
                height=get_value_at_index(getimagesize_48, 1),
                interpolation="lanczos",
                keep_proportion=True,
                condition="always",
                multiple_of=64,
                image=get_value_at_index(reactorfaceswap_73, 0),
            )

            pixelperfectresolution_46 = pixelperfectresolution.execute(
                image_gen_width=get_value_at_index(getimagesize_48, 0),
                image_gen_height=get_value_at_index(getimagesize_48, 1),
                resize_mode="Just Resize",
                original_image=get_value_at_index(imagescaletototalpixels_47, 0),
            )

            imageresize_16 = imageresize.execute(
                width=get_value_at_index(getimagesize_48, 0),
                height=get_value_at_index(getimagesize_48, 1),
                interpolation="lanczos",
                keep_proportion=False,
                condition="always",
                multiple_of=64,
                image=get_value_at_index(vhs_loadvideo_12, 0),
            )

            emptyimage_20 = emptyimage.generate(
                width=get_value_at_index(getimagesize_48, 1),
                height=get_value_at_index(getimagesize_48, 1),
                batch_size=1,
                color=0,
            )

            impactimagebatchtoimagelist_19 = impactimagebatchtoimagelist.doit(
                image=get_value_at_index(imageresize_16, 0)
            )

            birefnet_17 = birefnet.matting(
                device="auto",
                image=get_value_at_index(impactimagebatchtoimagelist_19, 0),
            )

            masklisttomaskbatch_21 = masklisttomaskbatch.doit(
                mask=get_value_at_index(birefnet_17, 0)
            )

            invertmask_29 = invertmask.invert(
                mask=get_value_at_index(masklisttomaskbatch_21, 0)
            )

            imagecompositemasked_18 = imagecompositemasked.composite(
                x=0,
                y=0,
                resize_source=True,
                destination=get_value_at_index(imageresize_16, 0),
                source=get_value_at_index(emptyimage_20, 0),
                mask=get_value_at_index(invertmask_29, 0),
            )

            depthanythingpreprocessor_30 = depthanythingpreprocessor.execute(
                ckpt_name="depth_anything_vitl14.pth",
                resolution=get_value_at_index(pixelperfectresolution_46, 0),
                image=get_value_at_index(imagecompositemasked_18, 0),
            )

            imagecompositemasked_31 = imagecompositemasked.composite(
                x=0,
                y=0,
                resize_source=True,
                destination=get_value_at_index(depthanythingpreprocessor_30, 0),
                source=get_value_at_index(emptyimage_20, 0),
                mask=get_value_at_index(invertmask_29, 0),
            )

            dsine_normalmappreprocessor_26 = dsine_normalmappreprocessor.execute(
                fov=60,
                iterations=5,
                resolution=get_value_at_index(pixelperfectresolution_46, 0),
                image=get_value_at_index(imagecompositemasked_18, 0),
            )

            imagecompositemasked_28 = imagecompositemasked.composite(
                x=0,
                y=0,
                resize_source=True,
                destination=get_value_at_index(dsine_normalmappreprocessor_26, 0),
                source=get_value_at_index(emptyimage_20, 0),
                mask=get_value_at_index(invertmask_29, 0),
            )

            denseposepreprocessor_32 = denseposepreprocessor.execute(
                model="densepose_r101_fpn_dl.torchscript",
                cmap="Viridis (MagicAnimate)",
                resolution=get_value_at_index(pixelperfectresolution_46, 0),
                image=get_value_at_index(imagecompositemasked_18, 0),
            )

            imagecompositemasked_33 = imagecompositemasked.composite(
                x=0,
                y=0,
                resize_source=True,
                destination=get_value_at_index(denseposepreprocessor_32, 0),
                source=get_value_at_index(emptyimage_20, 0),
                mask=get_value_at_index(invertmask_29, 0),
            )

            pixelperfectresolution_15 = pixelperfectresolution.execute(
                image_gen_width=get_value_at_index(getimagesize_48, 0),
                image_gen_height=get_value_at_index(getimagesize_48, 1),
                resize_mode="Resize and Fill",
                original_image=get_value_at_index(imagecompositemasked_18, 0),
            )

            dwpreprocessor_13 = dwpreprocessor.estimate_pose(
                detect_hand="enable",
                detect_body="enable",
                detect_face="enable",
                resolution=get_value_at_index(pixelperfectresolution_15, 0),
                bbox_detector="yolox_l.onnx",
                pose_estimator="dw-ll_ucoco_384_bs5.torchscript.pt",
                image=get_value_at_index(imagecompositemasked_18, 0),
            )

            champ_sampler_11 = champ_sampler.process(
                width=get_value_at_index(getimagesize_48, 0),
                height=get_value_at_index(getimagesize_48, 1),
                steps=36,
                guidance_scale=2,
                frames=get_value_at_index(vhs_loadvideo_12, 1),
                seed=random.randint(1, 2**64),
                keep_model_loaded=True,
                scheduler="DDPMScheduler",
                champ_model=get_value_at_index(champ_model_loader_10, 0),
                champ_vae=get_value_at_index(champ_model_loader_10, 1),
                champ_encoder=get_value_at_index(champ_model_loader_10, 2),
                image=get_value_at_index(imageresize_14, 0),
                depth_tensors=get_value_at_index(imagecompositemasked_31, 0),
                normal_tensors=get_value_at_index(imagecompositemasked_28, 0),
                semantic_tensors=get_value_at_index(imagecompositemasked_33, 0),
                dwpose_tensors=get_value_at_index(dwpreprocessor_13, 0),
            )

            emptyimage_24 = emptyimage.generate(
                width=get_value_at_index(getimagesize_48, 0),
                height=get_value_at_index(getimagesize_48, 1),
                batch_size=1,
                color=0,
            )

            birefnet_23 = birefnet.matting(
                device="cuda:0", image=get_value_at_index(reactorfaceswap_73, 0)
            )

            growmaskwithblur_25 = growmaskwithblur.expand_mask(
                expand=2,
                incremental_expandrate=0,
                tapered_corners=True,
                flip_input=True,
                blur_radius=0.2,
                lerp_alpha=1,
                decay_factor=1,
                fill_holes=False,
                mask=get_value_at_index(birefnet_23, 0),
            )

            imagecompositemasked_22 = imagecompositemasked.composite(
                x=0,
                y=0,
                resize_source=True,
                destination=get_value_at_index(imageresize_14, 0),
                source=get_value_at_index(emptyimage_24, 0),
                mask=get_value_at_index(growmaskwithblur_25, 0),
            )

            lamaremover_34 = lamaremover.lama_remover(
                mask_threshold=250,
                gaussblur_radius=8,
                invert_mask=False,
                images=get_value_at_index(loadimage_59, 0),
                masks=get_value_at_index(growmaskwithblur_25, 1),
            )

            lamaremover_41 = lamaremover.lama_remover(
                mask_threshold=250,
                gaussblur_radius=8,
                invert_mask=False,
                images=get_value_at_index(imageresize_16, 0),
                masks=get_value_at_index(masklisttomaskbatch_21, 0),
            )

            impactimagebatchtoimagelist_38 = impactimagebatchtoimagelist.doit(
                image=get_value_at_index(champ_sampler_11, 0)
            )

            birefnet_36 = birefnet.matting(
                device="cuda:0",
                image=get_value_at_index(impactimagebatchtoimagelist_38, 0),
            )

            masklisttomaskbatch_39 = masklisttomaskbatch.doit(
                mask=get_value_at_index(birefnet_36, 0)
            )

            invertmask_37 = invertmask.invert(
                mask=get_value_at_index(masklisttomaskbatch_39, 0)
            )

            growmaskwithblur_40 = growmaskwithblur.expand_mask(
                expand=1,
                incremental_expandrate=0,
                tapered_corners=True,
                flip_input=False,
                blur_radius=0.1,
                lerp_alpha=1,
                decay_factor=1,
                fill_holes=False,
                mask=get_value_at_index(invertmask_37, 0),
            )

            imagecompositemasked_35 = imagecompositemasked.composite(
                x=0,
                y=0,
                resize_source=True,
                destination=get_value_at_index(champ_sampler_11, 0),
                source=get_value_at_index(lamaremover_41, 0),
                mask=get_value_at_index(growmaskwithblur_40, 0),
            )

            reactorfaceswap_44 = reactorfaceswap.execute(
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
                input_image=get_value_at_index(imagecompositemasked_35, 0),
                source_image=get_value_at_index(reactorfaceswap_73, 0),
            )
            
            # 필요한 경로와 파일 설정
            output_dir = args.output if args.output else "output.mp4"
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            # 터미널에서 확인한 FFmpeg의 경로를 설정
            ffmpeg_path = "/usr/bin/ffmpeg"  # 실제 경로로 설정
            
            def tensor_to_bytes(tensor):
                tensor = tensor.cpu().numpy() * 255
                return np.clip(tensor, 0, 255).astype(np.uint8)
            
            def save_video_with_audio(images, audio_data, frame_rate, filename_prefix="output_video", format="h264-mp4"):
                # 이미지 시퀀스를 저장할 임시 폴더 생성
                temp_image_dir = os.path.join(output_dir, "temp_images")
                if not os.path.exists(temp_image_dir):
                    os.makedirs(temp_image_dir)
                
                # 각 프레임 이미지를 PNG 파일로 저장
                for i, image in enumerate(images):
                    image = Image.fromarray(tensor_to_bytes(image))
                    image.save(os.path.join(temp_image_dir, f"frame_{i:05}.png"))
            
                # 이미지 시퀀스를 비디오 파일로 변환
                video_output_path = os.path.join(output_dir, f"{filename_prefix}.mp4")
                ffmpeg_cmd = [
                    ffmpeg_path,
                    "-framerate", str(frame_rate),
                    "-i", os.path.join(temp_image_dir, "frame_%05d.png"),
                    "-c:v", "libx264",
                    "-pix_fmt", "yuv420p",
                    video_output_path
                ]
                subprocess.run(ffmpeg_cmd, check=True)
            
                # 오디오를 비디오에 추가
                if audio_data:
                    audio_path = os.path.join(output_dir, "temp_audio.wav")
                    with open(audio_path, "wb") as f:
                        f.write(audio_data)
                    
                    final_output_path = os.path.join(output_dir, f"{filename_prefix}_with_audio.mp4")
                    ffmpeg_cmd_with_audio = [
                        ffmpeg_path,
                        "-i", video_output_path,
                        "-i", audio_path,
                        "-c:v", "copy",
                        "-c:a", "aac",
                        "-shortest",
                        final_output_path
                    ]
                    subprocess.run(ffmpeg_cmd_with_audio, check=True)
                    os.remove(video_output_path)  # 중간 파일 삭제
                    os.remove(audio_path)
                    video_output_path = final_output_path
                
                # 임시 이미지 폴더 삭제
                for file in os.listdir(temp_image_dir):
                    os.remove(os.path.join(temp_image_dir, file))
                os.rmdir(temp_image_dir)
            
                print(f"Video saved at: {video_output_path}")
            
            # 예제 사용
            images = get_value_at_index(reactorfaceswap_44, 0)  # 텐서 이미지 시퀀스
            audio = get_value_at_index(vhs_loadvideo_12, 2)  # 바이너리 오디오 데이터
            
            save_video_with_audio(images, audio, frame_rate=8, filename_prefix="ComfyUI_swap", format="h264-mp4")

if __name__ == "__main__":
    main()
