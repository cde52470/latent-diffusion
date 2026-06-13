# 實驗重現方式

本 repo 使用 `uv` 管理 Python 3.11 環境，不使用 `requirements.txt`。實驗結果放在 `outputs/experiments/`，報告放在 `docs/ldm_redo_cross_attention_report_zh.md`。

## 環境

本次實驗紀錄的環境如下：

- Python: 3.11.14
- PyTorch: 2.4.1 + CUDA 12.1
- diffusers: 0.30.3
- transformers: 4.36.2
- numpy: 1.26.4
- GPU: NVIDIA GeForce RTX 2070 SUPER

`pyproject.toml` 已固定主要套件版本，並設定 Linux 下使用 PyTorch CUDA 12.1 wheel index。

## 安裝

```bash
uv sync
```

如果機器沒有 NVIDIA GPU，也可以同步同一份環境後用 CPU 跑，但圖片生成會慢很多：

```bash
uv run python scripts/reproduce_experiments.py --device cpu
```

## 重現全部實驗

```bash
uv run python scripts/reproduce_experiments.py
```

這會重新產生：

- `outputs/experiments/001_diffusers_text2img_baseline/`
- `outputs/experiments/002_cross_attention_mechanism/`
- `outputs/experiments/003_attention_shape_observation/`

## 只重現單一實驗

```bash
uv run python scripts/reproduce_experiments.py --experiment baseline
uv run python scripts/reproduce_experiments.py --experiment prompt-comparison
uv run python scripts/reproduce_experiments.py --experiment attention-shapes
```

## 模型來源

實驗使用 Hugging Face 上的模型：

```text
CompVis/ldm-text2im-large-256
```

模型會由 `diffusers` 自動下載到本機 Hugging Face cache。這些 cache 不放進 Git，因為檔案很大，且可以重新下載。

## 實驗設定

主要生成設定：

- seed: 23
- num_inference_steps: 50
- eta: 0.3
- guidance_scale: 6.0
- dtype: CUDA 使用 float16，CPU 使用 float32

baseline prompt：

```text
A painting of a squirrel eating a burger
```

prompt 對照實驗：

```text
A painting of a squirrel eating a burger
A painting of a cat eating a burger
A painting of a squirrel eating pizza
A photo of a squirrel eating a burger
```

## 不放進 Git 的內容

以下是重現時會在本機產生或下載的東西，不應該推到 GitHub：

- `.venv/`
- `models/huggingface/`
- `models/ldm/text2img-large/model.ckpt`
- `latent_diffusion.egg-info/`
- `__pycache__/`

這些都可以透過 `uv sync`、模型下載或 Python 執行過程重新產生。
