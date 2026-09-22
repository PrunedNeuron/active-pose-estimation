#!/usr/bin/env python
"""Minimal ID-Pose -> EscherNet 6DoF integration.

Reads an ID-Pose ``pose.json`` plus the preprocessed anchor/target images that
ID-Pose saved beside it. The predicted relative spherical camera displacement
is converted into a pair of EscherNet-compatible world-to-camera extrinsics,
then EscherNet generates the target view.

This is deliberately an inference adapter: it does not load both large models
at once and therefore remains suitable for a 4 GiB GPU.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw


def parse_args() -> argparse.Namespace:
    thesis_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pose-json",
        type=Path,
        default=(
            thesis_root
            / "id-pose/outputs/lion2-lowmem/lion2_000+001/pose.json"
        ),
        help="ID-Pose pose.json to consume.",
    )
    parser.add_argument(
        "--eschernet-root",
        type=Path,
        default=thesis_root / "EscherNet",
    )
    parser.add_argument(
        "--model",
        default="kxic/eschernet-6dof",
        help="Hugging Face model ID or local snapshot directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=thesis_root / "results/idpose-to-eschernet",
    )
    parser.add_argument(
        "--target-view",
        default=None,
        help="Target view key from pose.json; defaults to its first non-anchor view.",
    )
    parser.add_argument(
        "--anchor-radius",
        type=float,
        default=1.8,
        help="Canonical EscherNet camera radius before applying ID-Pose delta-radius.",
    )
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance-scale", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_smoke_helpers(script_dir: Path):
    path = script_dir / "04-eschernet-smoke.py"
    if not path.exists():
        raise FileNotFoundError(
            f"Required helper script not found: {path}\n"
            "Place 04-eschernet-smoke.py and 06-idpose-to-eschernet.py "
            "in the same scripts directory."
        )
    spec = importlib.util.spec_from_file_location("eschernet_smoke", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import helpers from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def spherical_to_cartesian(spherical: np.ndarray) -> np.ndarray:
    """Match ID-Pose src/utils.py exactly: [theta, azimuth, radius]."""
    theta, azimuth, radius = (float(value) for value in spherical)
    return np.array(
        [
            radius * math.sin(theta) * math.cos(azimuth),
            radius * math.sin(theta) * math.sin(azimuth),
            radius * math.cos(theta),
        ],
        dtype=np.float32,
    )


def eye_to_w2c(
    eye: np.ndarray,
    look_at: np.ndarray | None = None,
    up: np.ndarray | None = None,
) -> np.ndarray:
    """Build the Blender/Z-up extrinsic used by ID-Pose and EscherNet GSO."""
    if look_at is None:
        look_at = np.zeros(3, dtype=np.float32)
    if up is None:
        up = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    backward = eye - look_at
    backward /= np.linalg.norm(backward)
    side = np.cross(backward, up)
    side_norm = np.linalg.norm(side)
    if side_norm < 1e-7:
        raise ValueError("Camera direction is parallel to the Z-up vector")
    side /= side_norm
    camera_up = np.cross(side, backward)

    rotation = np.stack([-side, camera_up, backward]).astype(np.float32)
    extrinsic = np.eye(4, dtype=np.float32)
    extrinsic[:3, :3] = rotation
    extrinsic[:3, 3] = -rotation @ eye
    return extrinsic


def convert_relative_pose(
    relative_spherical: np.ndarray,
    anchor_radius: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # ID-Pose estimates only a relative pose. Global camera orientation is a
    # gauge freedom, so choose a non-singular canonical anchor on +X.
    anchor_spherical = np.array(
        [math.pi / 2.0, 0.0, anchor_radius], dtype=np.float32
    )
    target_spherical = anchor_spherical + relative_spherical
    target_spherical[1] %= 2.0 * math.pi
    if target_spherical[2] <= 0.0:
        raise ValueError(
            "ID-Pose radius delta produces a non-positive target radius: "
            f"{target_spherical[2]}"
        )

    anchor_w2c = eye_to_w2c(spherical_to_cartesian(anchor_spherical))
    target_w2c = eye_to_w2c(spherical_to_cartesian(target_spherical))
    return anchor_spherical, target_spherical, anchor_w2c, target_w2c


def pose_tensors(
    anchor_w2c: np.ndarray,
    target_w2c: np.ndarray,
    device: torch.device,
) -> list[list[torch.Tensor]]:
    dtype = torch.float16

    def direct(matrix: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(matrix).to(device=device, dtype=dtype)[None, None]

    def inverse_transpose(matrix: np.ndarray) -> torch.Tensor:
        value = np.linalg.inv(matrix).T.astype(np.float32)
        return torch.from_numpy(value).to(device=device, dtype=dtype)[None, None]

    return [
        [direct(target_w2c), inverse_transpose(target_w2c)],
        [direct(anchor_w2c), inverse_transpose(anchor_w2c)],
    ]


def psnr(reference: np.ndarray, prediction: np.ndarray) -> float:
    mse = float(np.mean((reference.astype(np.float32) - prediction) ** 2))
    return float("inf") if mse == 0.0 else -10.0 * math.log10(mse)


def labeled(image: Image.Image, text: str) -> Image.Image:
    bar = 28
    panel = Image.new("RGB", (image.width, image.height + bar), "white")
    panel.paste(image, (0, bar))
    ImageDraw.Draw(panel).text((7, 7), text, fill="black")
    return panel


def save_comparison(
    anchor: Image.Image,
    observed: Image.Image,
    generated: Image.Image,
    anchor_view: str,
    target_view: str,
    path: Path,
) -> None:
    panels = [
        labeled(anchor, f"ID-Pose anchor {anchor_view}"),
        labeled(observed, f"observed target {target_view}"),
        labeled(generated, "EscherNet from predicted pose"),
    ]
    canvas = Image.new(
        "RGB",
        (sum(panel.width for panel in panels), panels[0].height),
        "white",
    )
    x = 0
    for panel in panels:
        canvas.paste(panel, (x, 0))
        x += panel.width
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def main() -> int:
    args = parse_args()
    args.pose_json = args.pose_json.resolve()
    args.eschernet_root = args.eschernet_root.resolve()
    args.output_dir = args.output_dir.resolve()

    if not args.pose_json.exists():
        raise FileNotFoundError(f"ID-Pose output not found: {args.pose_json}")
    if not torch.cuda.is_available():
        raise RuntimeError("This integration requires a CUDA GPU")
    if args.resolution % 8:
        raise ValueError("--resolution must be divisible by 8")

    pose_document = json.loads(args.pose_json.read_text(encoding="utf-8"))
    anchor_view = str(pose_document["anchor_vid"])
    observations = pose_document["obs"]
    target_candidates = [key for key in observations if key != anchor_view]
    if not target_candidates:
        raise ValueError("pose.json contains no non-anchor observation")
    target_view = str(args.target_view or target_candidates[0])
    if target_view not in observations:
        raise KeyError(f"Target view {target_view!r} is absent from pose.json")
    if "rel_sph" not in observations[target_view]:
        raise KeyError(f"Target view {target_view!r} has no rel_sph prediction")

    relative_spherical = np.asarray(
        observations[target_view]["rel_sph"], dtype=np.float32
    )
    if relative_spherical.shape != (3,) or not np.all(np.isfinite(relative_spherical)):
        raise ValueError(f"Invalid rel_sph: {relative_spherical!r}")

    (
        anchor_spherical,
        target_spherical,
        anchor_w2c,
        target_w2c,
    ) = convert_relative_pose(relative_spherical, args.anchor_radius)

    image_dir = args.pose_json.parent
    anchor_path = image_dir / f"{anchor_view}.png"
    target_path = image_dir / f"{target_view}.png"
    if not anchor_path.exists() or not target_path.exists():
        raise FileNotFoundError(
            "Expected ID-Pose preprocessed images beside pose.json:\n"
            f"  {anchor_path}\n  {target_path}"
        )

    helpers = load_smoke_helpers(Path(__file__).resolve().parent)
    helpers.add_eschernet_imports(args.eschernet_root)
    device = torch.device("cuda:0")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    anchor_pil, anchor_tensor = helpers.load_rgba_on_white(
        anchor_path, args.resolution
    )
    target_pil, _ = helpers.load_rgba_on_white(target_path, args.resolution)
    anchor_tensor = anchor_tensor.to(device=device, dtype=torch.float16)
    poses = pose_tensors(anchor_w2c, target_w2c, device)

    pipeline = helpers.load_pipeline(args.model, device)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    print(
        f"Generating target {target_view} from anchor {anchor_view}: "
        f"rel_sph={relative_spherical.tolist()}",
        flush=True,
    )
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        prediction = pipeline(
            input_imgs=anchor_tensor,
            prompt_imgs=anchor_tensor,
            poses=poses,
            height=args.resolution,
            width=args.resolution,
            T_in=1,
            T_out=1,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.steps,
            generator=generator,
            output_type="numpy",
        ).images[0]
    elapsed = time.perf_counter() - started
    peak_gib = torch.cuda.max_memory_allocated(device) / 2**30

    prediction = np.asarray(prediction, dtype=np.float32)
    generated_pil = Image.fromarray(
        np.uint8(np.clip(prediction * 255.0, 0, 255)), mode="RGB"
    )
    observed = np.asarray(target_pil, dtype=np.float32) / 255.0
    score = psnr(observed, prediction)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    generated_path = args.output_dir / f"generated-{anchor_view}-to-{target_view}.png"
    comparison_path = args.output_dir / f"comparison-{anchor_view}-to-{target_view}.png"
    report_path = args.output_dir / "report.json"
    generated_pil.save(generated_path)
    save_comparison(
        anchor_pil,
        target_pil,
        generated_pil,
        anchor_view,
        target_view,
        comparison_path,
    )

    report = {
        "status": "success",
        "integration": "ID-Pose rel_sph -> EscherNet 6DoF",
        "pose_json": str(args.pose_json),
        "anchor_view": anchor_view,
        "target_view": target_view,
        "idpose_relative_spherical": {
            "convention": "[delta polar radians, delta azimuth radians, delta radius]",
            "value": relative_spherical.tolist(),
        },
        "canonical_anchor_spherical": anchor_spherical.tolist(),
        "derived_target_spherical": target_spherical.tolist(),
        "anchor_w2c": anchor_w2c.tolist(),
        "target_w2c": target_w2c.tolist(),
        "model": args.model,
        "resolution": args.resolution,
        "steps": args.steps,
        "guidance_scale": args.guidance_scale,
        "seed": args.seed,
        "elapsed_seconds": elapsed,
        "peak_allocated_vram_gib": peak_gib,
        "psnr_db_against_observed_target": score,
        "gpu": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "python": platform.python_version(),
        "outputs": {
            "generated": str(generated_path),
            "comparison": str(comparison_path),
        },
    }
    report_path.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    print(f"SUCCESS: {comparison_path}", flush=True)
    print(f"Elapsed: {elapsed:.2f}s", flush=True)
    print(f"Peak allocated VRAM: {peak_gib:.3f} GiB", flush=True)
    print(f"PSNR against observed target: {score:.2f} dB", flush=True)
    print(f"Report: {report_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
