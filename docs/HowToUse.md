## Usage

All commands should be executed within the `TIFF/` subfolder

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
--output_frame_rate 8
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
...     "output_frame_rate": 8
... }
>>> infer.main(**args)
```
