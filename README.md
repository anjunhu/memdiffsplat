### Memorization Evaluation
```
wget https://huggingface.co/datasets/tiange/Cap3D/resolve/main/Cap3D_automated_Objaverse_full.csv
python run_baseline.py
```

### Download Pretrained Models

Note that:
- Pretrained weights will download from HuggingFace and stored in `./out`.
- Other pretrained models (such as CLIP, T5, image VAE, etc.) will be downloaded automatically and stored in your HuggingFace cache directory.
- If you face problems in visiting HuggingFace Hub, you can try to set the environment variable `export HF_ENDPOINT=https://hf-mirror.com`.
- `GSRecon` pretrained weights is NOT really used during inference. Only its rendering function is used for visualization.

```bash
python3 download_ckpt.py --model_type [MODEL_TYPE] [--image_cond]

# `MODEL_TYPE`: choose from "sd15", "pas", "sd35m", "depth", "normal", "canny", "elevest"
# `--image_cond`: add this flag for downloading image-conditioned models
```

For example, to download the `text-cond SD1.5-based DiffSplat`:
```bash
python3 download_ckpt.py --model_type sd15
```
To download the `image-cond PixArt-Sigma-based DiffSplat`:
```bash
python3 download_ckpt.py --model_type pas --image_cond
```

```bash
# DiffSplat (SD1.5)
bash scripts/infer.sh infer_gsdiff_sd.py configs/gsdiff_sd15.yaml gsdiff_gobj83k_sd15__render \
--prompt a_toy_robot --output_video_type gif \
--gpu_id 0 --seed 0 --half_precision

# DiffSplat (PixArt-Sigma)
bash scripts/infer.sh src/infer_gsdiff_pas.py configs/gsdiff_pas.yaml gsdiff_gobj83k_pas_fp16__render \
--prompt a_toy_robot --output_video_type gif \
--gpu_id 0 --seed 0 [--half_precision]

# DiffSplat (SD3.5m)
bash scripts/infer.sh src/infer_gsdiff_sd3.py configs/gsdiff_sd35m_80g.yaml gsdiff_gobj83k_sd35m__render \
--prompt a_toy_robot --output_video_type gif \
--gpu_id 0 --seed 0 [--half_precision]
```


```bash
# DiffSplat (SD1.5)
bash scripts/infer.sh src/infer_gsdiff_sd.py configs/gsdiff_sd15.yaml gsdiff_gobj83k_sd15_image__render \
--rembg_and_center --triangle_cfg_scaling --output_video_type gif --guidance_scale 2 \
--image_path assets/grm/frog.png --elevation 20 --prompt a_frog

# DiffSplat (PixArt-Sigma)
bash scripts/infer.sh src/infer_gsdiff_pas.py configs/gsdiff_pas.yaml gsdiff_gobj83k_pas_fp16_image__render \
--rembg_and_center --triangle_cfg_scaling --output_video_type gif --guidance_scale 2 \
--image_path assets/grm/frog.png --elevation 20 --prompt a_frog

# DiffSplat (SD3.5m)
bash scripts/infer.sh src/infer_gsdiff_sd3.py configs/gsdiff_sd35m_80g.yaml gsdiff_gobj83k_sd35m_image__render \
--rembg_and_center --triangle_cfg_scaling --output_video_type gif --guidance_scale 2 \
--image_path assets/grm/frog.png --elevation 20 --prompt a_frog
```
