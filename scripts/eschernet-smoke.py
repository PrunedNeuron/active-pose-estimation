#!/usr/bin/env python
"""Minimal EscherNet 6DoF inference smoke test for a 4 GiB CUDA GPU.

Uses the bundled GSO30 elephant example:
  reference: view 000
  target pose / comparison image: view 001

The script tries 256 px first and retries at 128 px only if inference raises a
CUDA out-of-memory error. It saves a horizontal reference/target/generated
comparison under ``results/``.
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw


def parse_args() -> argparse.Namespace:
    thesis_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--eschernet-root",
        type=Path,
        default=thesis_root / "EscherNet",
        help="Path to the EscherNet checkout.",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("ESCHERNET_MODEL", "kxic/eschernet-6dof"),
        help="Hugging Face model ID or local snapshot directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=thesis_root / "results",
    )
    parser.add_argument("--reference-view", type=int, default=0)
    parser.add_argument("--target-view", type=int, default=1)
    parser.add_argument("--resolutions", type=int, nargs="+", default=[256, 128])
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--guidance-scale", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def add_eschernet_imports(root: Path) -> None:
    root = root.resolve()
    six_dof = root / "6DoF"
    required = [
        six_dof / "pipeline_zero1to3.py",
        six_dof / "CN_encoder.py",
        root / "demo/GSO30/elephant/render_mvs_25/model/000.png",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "EscherNet checkout is incomplete or --eschernet-root is wrong:\n  "
            + "\n  ".join(missing)
        )

    # EscherNet ships a customized Diffusers tree inside 6DoF. It must shadow
    # the environment's ordinary diffusers installation.
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(six_dof))


def load_rgba_on_white(path: Path, resolution: int) -> tuple[Image.Image, torch.Tensor]:
    rgba = Image.open(path).convert("RGBA")
    rgba = rgba.resize((resolution, resolution), Image.Resampling.LANCZOS)
    array = np.asarray(rgba, dtype=np.float32) / 255.0
    alpha = array[..., 3:4]
    rgb = array[..., :3] * alpha + (1.0 - alpha)
    image = Image.fromarray(np.uint8(np.clip(rgb * 255.0, 0, 255)), mode="RGB")
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).contiguous()
    tensor = tensor.mul(2.0).sub(1.0).unsqueeze(0)
    return image, tensor


def load_pose(path: Path) -> np.ndarray:
    pose = np.asarray(np.load(path), dtype=np.float32)
    if pose.shape == (3, 4):
        pose = np.concatenate(
            [pose, np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32)],
            axis=0,
        )
    if pose.shape != (4, 4):
        raise ValueError(f"Expected a 3x4 or 4x4 pose in {path}, got {pose.shape}")
    return pose


def pose_tensors(
    reference_pose: np.ndarray,
    target_pose: np.ndarray,
    device: torch.device,
) -> list[list[torch.Tensor]]:
    dtype = torch.float16

    def direct(pose: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(pose).to(device=device, dtype=dtype)[None, None]

    def inverse_transpose(pose: np.ndarray) -> torch.Tensor:
        value = np.linalg.inv(pose).T.astype(np.float32, copy=False)
        return torch.from_numpy(value).to(device=device, dtype=dtype)[None, None]

    pose_in = direct(reference_pose)
    pose_out = direct(target_pose)
    pose_in_inv = inverse_transpose(reference_pose)
    pose_out_inv = inverse_transpose(target_pose)
    return [[pose_out, pose_out_inv], [pose_in, pose_in_inv]]


def load_pipeline(model: str, device: torch.device):
    from CN_encoder import CN_encoder
    from diffusers import DDIMScheduler
    from pipeline_zero1to3 import Zero1to3StableDiffusionPipeline

    dtype = torch.float16
    print(f"Loading EscherNet 6DoF from {model}...", flush=True)
    scheduler = DDIMScheduler.from_pretrained(model, subfolder="scheduler")
    image_encoder = CN_encoder.from_pretrained(
        model,
        subfolder="image_encoder",
        torch_dtype=dtype,
    )
    pipeline = Zero1to3StableDiffusionPipeline.from_pretrained(
        model,
        scheduler=scheduler,
        image_encoder=None,
        safety_checker=None,
        feature_extractor=None,
        torch_dtype=dtype,
    )
    pipeline.image_encoder = image_encoder
    pipeline.set_progress_bar_config(disable=False)
    pipeline.enable_xformers_memory_efficient_attention()
    pipeline.enable_vae_slicing()
    pipeline.to(device)

    for module in (pipeline.unet, pipeline.vae, pipeline.image_encoder):
        module.eval()
        module.requires_grad_(False)

    print("Pipeline ready", flush=True)
    return pipeline


def save_comparison(
    reference: Image.Image,
    target: Image.Image,
    generated: np.ndarray,
    path: Path,
) -> None:
    generated_image = Image.fromarray(
        np.uint8(np.clip(generated * 255.0, 0, 255)),
        mode="RGB",
    )
    panels = [reference, target, generated_image]
    labels = ["reference 000", "target 001", "generated at pose 001"]
    label_height = 24
    canvas = Image.new(
        "RGB",
        (sum(panel.width for panel in panels), panels[0].height + label_height),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    x = 0
    for panel, label in zip(panels, labels):
        canvas.paste(panel, (x, label_height))
        draw.text((x + 6, 5), label, fill="black")
        x += panel.width

    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def run_at_resolution(
    pipeline,
    args: argparse.Namespace,
    device: torch.device,
    resolution: int,
) -> Path:
    model_dir = (
        args.eschernet_root
        / "demo/GSO30/elephant/render_mvs_25/model"
    )
    reference_stem = f"{args.reference_view:03d}"
    target_stem = f"{args.target_view:03d}"

    reference_pil, reference = load_rgba_on_white(
        model_dir / f"{reference_stem}.png",
        resolution,
    )
    target_pil, _ = load_rgba_on_white(
        model_dir / f"{target_stem}.png",
        resolution,
    )
    reference_pose = load_pose(model_dir / f"{reference_stem}.npy")
    target_pose = load_pose(model_dir / f"{target_stem}.npy")

    reference = reference.to(device=device, dtype=torch.float16)
    poses = pose_tensors(reference_pose, target_pose, device)
    generator = torch.Generator(device=device).manual_seed(args.seed)

    torch.cuda.reset_peak_memory_stats(device)
    print(
        f"Generating {resolution}x{resolution}, "
        f"steps={args.steps}, guidance={args.guidance_scale}...",
        flush=True,
    )
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        generated = pipeline(
            input_imgs=reference,
            prompt_imgs=reference,
            poses=poses,
            height=resolution,
            width=resolution,
            T_in=1,
            T_out=1,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.steps,
            generator=generator,
            output_type="numpy",
        ).images[0]

    peak_gib = torch.cuda.max_memory_allocated(device) / 2**30
    output = args.output_dir / f"eschernet-smoke-{resolution}px.png"
    save_comparison(reference_pil, target_pil, generated, output)
    print(f"Peak allocated VRAM: {peak_gib:.3f} GiB", flush=True)
    print(f"Saved: {output}", flush=True)
    return output


def main() -> int:
    args = parse_args()
    args.eschernet_root = args.eschernet_root.resolve()
    args.output_dir = args.output_dir.resolve()
    add_eschernet_imports(args.eschernet_root)

    if not torch.cuda.is_available():
        raise RuntimeError("This smoke test requires a CUDA GPU")

    device = torch.device("cuda:0")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    pipeline = load_pipeline(args.model, device)

    last_oom = None
    for resolution in args.resolutions:
        try:
            output = run_at_resolution(pipeline, args, device, resolution)
            print(f"SUCCESS: {output}", flush=True)
            return 0
        except RuntimeError as error:
            if "out of memory" not in str(error).lower():
                raise
            last_oom = error
            print(f"CUDA OOM at {resolution}px; retrying smaller resolution", flush=True)
            gc.collect()
            torch.cuda.empty_cache()

    raise RuntimeError(
        f"EscherNet exhausted CUDA memory at all requested resolutions: {args.resolutions}"
    ) from last_oom


if __name__ == "__main__":
    raise SystemExit(main())
