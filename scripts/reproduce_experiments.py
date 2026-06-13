"""Reproduce the report experiment artifacts with uv and Python 3.11.

The script recreates the curated artifacts under outputs/experiments:

1. A baseline text-to-image generation.
2. A prompt-factor comparison grid.
3. Cross-attention tensor shape observations for one denoising step.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from diffusers import DiffusionPipeline
from PIL import Image, ImageDraw


MODEL_ID = "CompVis/ldm-text2im-large-256"
PROMPT = "A painting of a squirrel eating a burger"
SEED = 23
NUM_INFERENCE_STEPS = 50
ETA = 0.3
GUIDANCE_SCALE = 6.0
OUTPUT_ROOT = Path("outputs/experiments")


PROMPT_VARIANTS = {
    "baseline": "A painting of a squirrel eating a burger",
    "subject_cat": "A painting of a cat eating a burger",
    "object_pizza": "A painting of a squirrel eating pizza",
    "style_photo": "A photo of a squirrel eating a burger",
}

ATTENTION_MODULES = [
    "down_blocks.0.attentions.0.transformer_blocks.0.attn2",
    "mid_block.attentions.0.transformer_blocks.0.attn2",
    "up_blocks.3.attentions.2.transformer_blocks.0.attn2",
]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_log(path: Path, lines: list[str]) -> None:
    ensure_dir(path.parent)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def environment_metadata() -> dict[str, Any]:
    return {
        "python": ".".join(map(str, __import__("sys").version_info[:3])),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "diffusers": __import__("diffusers").__version__,
        "transformers": __import__("transformers").__version__,
        "numpy": np.__version__,
    }


def load_pipeline(device: str) -> DiffusionPipeline:
    dtype = torch.float16 if device == "cuda" else torch.float32
    pipe = DiffusionPipeline.from_pretrained(MODEL_ID, torch_dtype=dtype)
    return pipe.to(device)


def generator_for(device: str, seed: int) -> torch.Generator:
    generator_device = "cuda" if device == "cuda" else "cpu"
    return torch.Generator(device=generator_device).manual_seed(seed)


def run_baseline(device: str) -> None:
    out_dir = OUTPUT_ROOT / "001_diffusers_text2img_baseline"
    image_path = out_dir / "images" / "baseline.png"
    log_path = out_dir / "logs" / "run.log"
    metadata_dir = out_dir / "metadata"
    ensure_dir(image_path.parent)

    logs = [f"prompt: {PROMPT}", f"device: {device}", "loading pipeline"]
    pipe = load_pipeline(device)
    logs.append("running generation")
    start = time.time()
    image = pipe(
        PROMPT,
        num_inference_steps=NUM_INFERENCE_STEPS,
        eta=ETA,
        guidance_scale=GUIDANCE_SCALE,
        generator=generator_for(device, SEED),
    ).images[0]
    elapsed = time.time() - start
    image.save(image_path)
    logs.extend([f"saved image: {image_path}", f"elapsed_seconds: {elapsed:.2f}"])

    write_log(log_path, logs)
    write_json(metadata_dir / "environment.json", environment_metadata())
    write_json(
        metadata_dir / "generation_config.json",
        {
            "experiment": "001_diffusers_text2img_baseline",
            "model": MODEL_ID,
            "prompt": PROMPT,
            "seed": SEED,
            "num_inference_steps": NUM_INFERENCE_STEPS,
            "eta": ETA,
            "guidance_scale": GUIDANCE_SCALE,
            "dtype": "float16" if device == "cuda" else "float32",
            "device": device,
            "output_image": str(image_path),
            "elapsed_seconds": elapsed,
        },
    )


def make_comparison_grid(images: dict[str, Image.Image], output_path: Path) -> None:
    tile_w, tile_h = next(iter(images.values())).size
    label_h = 32
    grid = Image.new("RGB", (tile_w * 2, (tile_h + label_h) * 2), "white")
    draw = ImageDraw.Draw(grid)

    for idx, (name, image) in enumerate(images.items()):
        x = (idx % 2) * tile_w
        y = (idx // 2) * (tile_h + label_h)
        grid.paste(image.convert("RGB"), (x, y + label_h))
        draw.text((x + 8, y + 8), name, fill=(0, 0, 0))

    ensure_dir(output_path.parent)
    grid.save(output_path)


def run_prompt_comparison(device: str) -> None:
    out_dir = OUTPUT_ROOT / "002_cross_attention_mechanism"
    images_dir = out_dir / "images"
    metadata_dir = out_dir / "metadata"
    log_path = out_dir / "logs" / "run.log"
    ensure_dir(images_dir)

    logs = ["experiment: 002_cross_attention_mechanism", f"device: {device}", "loading pipeline"]
    pipe = load_pipeline(device)
    generated: dict[str, Image.Image] = {}
    prompt_metadata = {}

    start = time.time()
    for name, prompt in PROMPT_VARIANTS.items():
        logs.append(f"generating {name}: {prompt}")
        image = pipe(
            prompt,
            num_inference_steps=NUM_INFERENCE_STEPS,
            eta=ETA,
            guidance_scale=GUIDANCE_SCALE,
            generator=generator_for(device, SEED),
        ).images[0]
        image_path = images_dir / f"{name}.png"
        image.save(image_path)
        generated[name] = image
        prompt_metadata[name] = {"prompt": prompt, "output_image": str(image_path)}

    grid_path = images_dir / "comparison_grid.png"
    make_comparison_grid(generated, grid_path)
    elapsed = time.time() - start
    logs.extend([f"saved comparison grid: {grid_path}", f"elapsed_seconds: {elapsed:.2f}"])

    write_log(log_path, logs)
    write_json(metadata_dir / "environment.json", environment_metadata())
    write_json(metadata_dir / "prompts.json", prompt_metadata)
    write_json(
        metadata_dir / "generation_config.json",
        {
            "experiment": "002_cross_attention_mechanism",
            "model": MODEL_ID,
            "seed": SEED,
            "num_inference_steps": NUM_INFERENCE_STEPS,
            "eta": ETA,
            "guidance_scale": GUIDANCE_SCALE,
            "dtype": "float16" if device == "cuda" else "float32",
            "device": device,
            "prompts": prompt_metadata,
            "comparison_grid": str(grid_path),
            "elapsed_seconds": elapsed,
        },
    )


def shape_list(value: Any) -> list[int] | None:
    return list(value.shape) if hasattr(value, "shape") else None


def split_heads_shape(tensor: torch.Tensor, heads: int) -> list[int]:
    batch, tokens, channels = tensor.shape
    return [batch, heads, tokens, channels // heads]


def run_attention_shapes(device: str) -> None:
    out_dir = OUTPUT_ROOT / "003_attention_shape_observation"
    metadata_dir = out_dir / "metadata"
    log_path = out_dir / "logs" / "run.log"

    logs = ["experiment: 003_attention_shape_observation", f"device: {device}", "loading pipeline"]
    pipe = load_pipeline(device)
    captured: dict[str, dict[str, Any]] = {}
    handles = []
    modules = dict(pipe.unet.named_modules())

    def make_hook(name: str):
        def hook(module: torch.nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any], output: Any) -> None:
            if name in captured:
                return
            hidden_states = args[0] if args else kwargs.get("hidden_states")
            encoder_hidden_states = kwargs.get("encoder_hidden_states")
            if encoder_hidden_states is None and len(args) > 1:
                encoder_hidden_states = args[1]
            if hidden_states is None or encoder_hidden_states is None:
                return

            heads = int(getattr(module, "heads", 1))
            q_linear = module.to_q(hidden_states)
            k_linear = module.to_k(encoder_hidden_states)
            v_linear = module.to_v(encoder_hidden_states)
            captured[name] = {
                "module_name": name,
                "module_class": module.__class__.__name__,
                "heads": heads,
                "input_hidden_states_shape": shape_list(hidden_states),
                "encoder_hidden_states_shape": shape_list(encoder_hidden_states),
                "output_shape": shape_list(output),
                "q_linear_shape": shape_list(q_linear),
                "k_linear_shape": shape_list(k_linear),
                "v_linear_shape": shape_list(v_linear),
                "q_heads_shape": split_heads_shape(q_linear, heads),
                "k_heads_shape": split_heads_shape(k_linear, heads),
                "v_heads_shape": split_heads_shape(v_linear, heads),
                "attention_probability_shape": [
                    int(q_linear.shape[0]),
                    heads,
                    int(q_linear.shape[1]),
                    int(k_linear.shape[1]),
                ],
            }

        return hook

    for module_name in ATTENTION_MODULES:
        if module_name in modules:
            handles.append(modules[module_name].register_forward_hook(make_hook(module_name), with_kwargs=True))

    tokenizer = pipe.tokenizer
    tokens = tokenizer(
        PROMPT,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    token_ids = tokens.input_ids.to(device)
    with torch.no_grad():
        text_embeddings = pipe.text_encoder(token_ids)[0]

    start = time.time()
    pipe(
        PROMPT,
        num_inference_steps=1,
        eta=ETA,
        guidance_scale=GUIDANCE_SCALE,
        generator=generator_for(device, SEED),
    )
    elapsed = time.time() - start

    for handle in handles:
        handle.remove()

    latent_size = int(getattr(pipe.unet.config, "sample_size", 32))
    latent_channels = int(getattr(pipe.unet.config, "in_channels", 4))
    payload = {
        "experiment": "003_attention_shape_observation",
        "model": MODEL_ID,
        "prompt": PROMPT,
        "seed": SEED,
        "num_inference_steps": 1,
        "eta": ETA,
        "guidance_scale": GUIDANCE_SCALE,
        "device": device,
        "tokenizer_max_length_used": int(tokenizer.model_max_length),
        "latent_shape": [1, latent_channels, latent_size, latent_size],
        "token_ids_shape": shape_list(token_ids),
        "tokens_first_20": tokenizer.convert_ids_to_tokens(token_ids[0][:20]),
        "text_embedding_shape": shape_list(text_embeddings),
        "selected_cross_attention_modules": captured,
        "elapsed_seconds": elapsed,
    }

    logs.extend([f"captured_modules: {len(captured)}", f"elapsed_seconds: {elapsed:.2f}"])
    write_log(log_path, logs)
    write_json(metadata_dir / "environment.json", environment_metadata())
    write_json(metadata_dir / "tensor_shapes.json", payload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment",
        choices=["all", "baseline", "prompt-comparison", "attention-shapes"],
        default="all",
        help="Experiment artifact set to regenerate.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Device to use. auto selects cuda when available.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"

    if args.experiment in {"all", "baseline"}:
        run_baseline(device)
    if args.experiment in {"all", "prompt-comparison"}:
        run_prompt_comparison(device)
    if args.experiment in {"all", "attention-shapes"}:
        run_attention_shapes(device)


if __name__ == "__main__":
    main()
