## Usage

All commands should be executed within the `TIFF/` subfolder

### 1. Run Python

```shell
python3 infer3.py \
--video ./videos/motion.mp4 \
--frame-load-cap 16 \
--positive-prompt "A detailed description of the desired output" \
--negative-prompt "A description of undesired elements" \
--ref-image ./images/reference.jpg \
--output ./results/output.png
```

Example command:

```shell
python3 infer3.py \
--video ./videos/motion.mp4 \
--frame-load-cap 16 \
--positive-prompt "A detailed description of the desired output" \
--negative-prompt "A description of undesired elements" \
--ref-image ./images/reference.jpg \
--output ./results/output.png
```

### 2. Import Class

```python
>>> import infer3
>>> args = {
...     "video": "./videos/motion.mp4",
...     "frame_load_cap": 16,
...     "positive_prompt": "A detailed description of the desired output",
...     "negative_prompt": "A description of undesired elements",
...     "ref_image": "./images/reference.jpg",
...     "output": "./results/output.png",
...     "disable_metadata": False
... }
>>> results = infer3.main(**args)
```
