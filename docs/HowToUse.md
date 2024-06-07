All commands should be executed within the `TIFF/` subfolder

## infer.py
This script orchestrates a comprehensive system for generating customized animations using text prompts, reference images, and source motion videos. It is the inference code with BiRefNet removed, as proposed in the paper.

### 1. Run Python

```shell
python3 infer.py \
--source_video # video for motion guidance
--frame-load-cap # frame number to load video
--positive-prompt # A detailed description of the desired output
--negative-prompt # A description of undesired elements
--ref-image # refered person image including face
--output_dir # directory of saving generated video
--output_frame_rate # the frame rate number of generated video
--queue_size # Number of times the workflow will be executed (default: 1)
```

Example command:

```shell
python3 infer.py \
--source_video ./source_videos/motion.mp4 \
--frame_load_cap 16 \
--positive-prompt "full body, yellow hair, black jacket, sports clothes, walking on a sidewalk" \
--negative-prompt "contortionist, amputee, polydactyly, deformed, distorted, misshapen, malformed, abnormal, mutant, defaced, shapeless" \
--ref-image ./images/AdultWoman.jpeg \
--output_dir ./output/
--output_frame_rate 8 \
--queue_size 1
```

### 2. Import Class

```python
>>> import infer
>>> args = {
...     "source_video": "./source_videos/motion.mp4",
...     "frame_load_cap": 16,
...     "positive_prompt": "full body, yellow hair, black jacket, sports clothes, walking on a sidewalk",
...     "negative_prompt": "contortionist, amputee, polydactyly, deformed, distorted, misshapen, malformed, abnormal, mutant, defaced, shapeless",
...     "ref_image": "./images/AdultWoman.jpeg",
...     "output": "./output/",
...     "output_frame_rate": 8,
...     "queue_size" : 1
... }
>>> infer.main(**args)
```

Example result:

<img src="https://github.com/jaechanjo/TIFF/assets/89237860/e3029d21-1845-42b9-9325-3a80187dbb98" alt="TIFF_result" width="200"/>



## TIFF.py
This script is a component of the system dedicated to generating text-guided, full-body human images.

### 1. Run Python

```shell
python3 TIFF.py \
--ref_image # Path to the reference image
--positive_prompt # Positive prompt for the desired image
--negative_prompt # Negative prompt for the undesired elements
--output_path # file path to save the generated image
--queue_size # Number of times the workflow will be executed (default: 1)
```

Example command:

```shell
python3 TIFF.py \
--ref_image ./images/AdultWoman.jpeg \
--positive_prompt "full body, yellow hair, black jacket, sports clothes, walking on a sidewalk" \
--negative_prompt "contortionist, amputee, polydactyly, deformed, distorted, misshapen, malformed, abnormal, mutant, defaced, shapeless" \
--output_path ./output/output.jpeg \
--queue_size 1 
```

### 2. Import Class

```python
>>> import TIFF
>>> args = {
...     "ref_image" : "./images/AdultWoman.jpeg",
...     "positive_prompt": "full body, yellow hair, black jacket, sports clothes, walking on a sidewalk",
...     "negative_prompt": "contortionist, amputee, polydactyly, deformed, distorted, misshapen, malformed, abnormal, mutant, defaced, shapeless",
...     "ref_image": "./images/AdultWoman.jpeg",
...     "output_path" : "./output/output.jpeg",
...     "queue_size": 1
... }
>>> TIFF.main(**args)
```

Example result:

<img width="200" alt="System_result" src="https://github.com/jaechanjo/TIFF/assets/89237860/ec2759fc-a2ce-4b9d-acff-3509fd6de030">

<img width="200" alt="System_video_result" src="https://github.com/jaechanjo/TIFF/assets/89237860/5a776b04-9786-46d8-b386-cc4df0737f91">
