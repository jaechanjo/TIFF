import os
import random
import sys
import json
import argparse
import contextlib
import numpy as np
from PIL import Image
from typing import Sequence, Mapping, Any, Union
import torch

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
    if args.output_path is None:
        return cls

    from PIL import Image, ImageOps, ImageSequence
    from PIL.PngImagePlugin import PngInfo

    import numpy as np

    class WrappedSaveImage(cls):
        counter = 0

        def save_images(
            self, images, filename_prefix="ComfyUI", prompt=None, extra_pnginfo=None
        ):
            if args.output_path is None:
                return super().save_images(
                    images, filename_prefix, prompt, extra_pnginfo
                )
            else:
                if len(images) > 1 and args.output_path == "-":
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

                    if args.output_path == "-":
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
                            if os.path.isdir(args.output_path):
                                subfolder = args.output_path
                                file = "output.png"
                            else:
                                subfolder, file = os.path.split(args.output_path)
                                if subfolder == "":
                                    subfolder = os.getcwd()
                        else:
                            if os.path.isdir(args.output_path):
                                subfolder = args.output_path
                                file = filename_prefix
                            else:
                                subfolder, file = os.path.split(args.output_path)

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
    "--output_path",
    "-o",
    default=None,
    help="The file path to save the output image. Either a file path, a directory, or - for stdout (default: the ComfyUI output directory)",
)

parser.add_argument(
    "--disable_metadata",
    action="store_true",
    help="Disables writing workflow metadata to the outputs",
)

parser.add_argument(
    "--ref_image",
    "-r",
    type=str,
    required=True,
    help="The reference image to use for the generation.",
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

comfy_args = [sys.argv[0]]
if "--" in sys.argv:
    idx = sys.argv.index("--")
    comfy_args += sys.argv[idx + 1 :]
    sys.argv = sys.argv[:idx]

args = None
if __name__ == "__main__":
    args = parser.parse_args()
    sys.argv = comfy_args
if args is not None and args.output_path is not None and args.output_path == "-":
    ctx = contextlib.redirect_stdout(sys.stderr)
else:
    ctx = contextlib.nullcontext()

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
        checkpointloadersimple = NODE_CLASS_MAPPINGS["CheckpointLoaderSimple"]()
        checkpointloadersimple_143 = checkpointloadersimple.load_checkpoint(
            ckpt_name="albedobaseXL_v21.safetensors"
        )

        cliptextencode = NODE_CLASS_MAPPINGS["CLIPTextEncode"]()
        cliptextencode_145 = cliptextencode.encode(
            text=args.positive_prompt,  # 수정된 부분: 긍정적 프롬프트 사용
            clip=get_value_at_index(checkpointloadersimple_143, 1),
        )

        cliptextencode_146 = cliptextencode.encode(
            text=args.negative_prompt,  # 수정된 부분: 부정적 프롬프트 사용
            clip=get_value_at_index(checkpointloadersimple_143, 1),
        )

        loadimage = NODE_CLASS_MAPPINGS["LoadImage"]()
        loadimage_150 = loadimage.load_image(image=args.ref_image)  # 수정된 부분: 참조 이미지 사용

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
        for q in range(args.queue_size):
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
                combine_embeds="add",
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
                face_restore_model="GFPGANv1.4.pth",
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
            
            # 생성된 이미지 받아오기
            generated_image_tensor = get_value_at_index(reactorfaceswap_233, 0)
            
            # Tensor의 형태를 (height, width, channels)로 변환
            if len(generated_image_tensor.shape) == 4:  # 예: (1, 1, 696, 3)
                generated_image_tensor = generated_image_tensor.squeeze(0).squeeze(0)
            
            # Tensor를 이미지로 변환
            generated_image_array = (generated_image_tensor.cpu().numpy() * 255).astype(np.uint8)
            generated_image = Image.fromarray(generated_image_array)
        
            # 이미지 저장 경로 설정
            output_path = args.output_path if args.output_path else os.path.join(os.getcwd(), "output.png")
            
            # 이미지 저장
            generated_image.save(output_path)
        
            # 저장 로그 출력
            print(f"이미지가 {output_path}에 저장되었습니다.")

if __name__ == "__main__":
    main()
