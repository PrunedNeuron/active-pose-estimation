#!/usr/bin/env python
"""Joint multi-view EscherNet 6DoF validation on the bundled elephant.

This is the follow-up to 04-eschernet-smoke.py. It loads the model once and
asks EscherNet to synthesize several target cameras jointly (T_out > 1), which
exercises its multi-view attention rather than repeating independent T_out=1
calls.

Outputs:
  results/eschernet-multiview/contact-sheet.png
  results/eschernet-multiview/report.json
  results/eschernet-multiview/generated-*.png
"""

from __future__ import annotations

import argparse
import gc
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
        default=thesis_root / "results/eschernet-multiview",
    )
    parser.add_argument("--reference-view", type=int, default=0)
    parser.add_argument(
        "--target-views",
        type=int,
        nargs="+",
        default=[1, 8, 16, 23],
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
            "Place 04-eschernet-smoke.py and 05-eschernet-multiview.py "
            "in the same scripts directory."
        )
    spec = importlib.util.spec_from_file_location("eschernet_smoke", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import helpers from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def stack_poses(
    helpers,
    reference_pose: np.ndarray,
    target_poses: list[np.ndarray],
    device: torch.device,
) -> list[list[torch.Tensor]]:
    dtype = torch.float16
    pose_in = torch.from_numpy(reference_pose).to(device, dtype)[None, None]
    pose_out = torch.from_numpy(np.stack(target_poses)).to(device, dtype)[None]

    input_inverse = np.linalg.inv(reference_pose).T.astype(np.float32)
    output_inverse = np.linalg.inv(np.stack(target_poses)).transpose(0, 2, 1)
    pose_in_inv = torch.from_numpy(input_inverse).to(device, dtype)[None, None]
    pose_out_inv = torch.from_numpy(output_inverse).to(device, dtype)[None]
    return [[pose_out, pose_out_inv], [pose_in, pose_in_inv]]


def camera_description(extrinsic: np.ndarray) -> dict[str, float]:
    # The bundled GSO matrices are world-to-camera extrinsics. Camera position
    # in world coordinates is -R^T t.
    center = -extrinsic[:3, :3].T @ extrinsic[:3, 3]
    radius = float(np.linalg.norm(center))
    azimuth = math.degrees(math.atan2(float(center[0]), float(center[2])))
    elevation = math.degrees(math.asin(float(center[1]) / radius))
    return {
        "azimuth_deg": azimuth,
        "elevation_deg": elevation,
        "radius": radius,
    }


def psnr(reference: np.ndarray, prediction: np.ndarray) -> float:
    mse = float(np.mean((reference.astype(np.float32) - prediction) ** 2))
    return float("inf") if mse == 0.0 else -10.0 * math.log10(mse)


def label_panel(image: Image.Image, label: str) -> Image.Image:
    bar_height = 28
    panel = Image.new("RGB", (image.width, image.height + bar_height), "white")
    panel.paste(image, (0, bar_height))
    ImageDraw.Draw(panel).text((7, 7), label, fill="black")
    return panel


def save_contact_sheet(
    reference: Image.Image,
    targets: list[Image.Image],
    generated: list[Image.Image],
    target_views: list[int],
    descriptions: list[dict[str, float]],
    path: Path,
) -> None:
    rows: list[Image.Image] = []
    for target, prediction, view, description in zip(
        targets, generated, target_views, descriptions
    ):
        pose_label = (
            f"target {view:03d}  az={description['azimuth_deg']:+.1f} "
            f"el={description['elevation_deg']:+.1f}"
        )
        panels = [
            label_panel(reference, "reference 000"),
            label_panel(target, pose_label),
            label_panel(prediction, f"generated {view:03d}"),
        ]
        row = Image.new(
            "RGB",
            (sum(panel.width for panel in panels), panels[0].height),
            "white",
        )
        x = 0
        for panel in panels:
            row.paste(panel, (x, 0))
            x += panel.width
        rows.append(row)

    sheet = Image.new(
        "RGB",
        (rows[0].width, sum(row.height for row in rows)),
        "white",
    )
    y = 0
    for row in rows:
        sheet.paste(row, (0, y))
        y += row.height
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def run_configuration(
    pipeline,
    helpers,
    args: argparse.Namespace,
    device: torch.device,
    target_views: list[int],
    resolution: int,
) -> tuple[list[np.ndarray], dict]:
    model_dir = (
        args.eschernet_root / "demo/GSO30/elephant/render_mvs_25/model"
    )
    reference_stem = f"{args.reference_view:03d}"
    reference_pil, reference = helpers.load_rgba_on_white(
        model_dir / f"{reference_stem}.png", resolution
    )
    reference_pose = helpers.load_pose(model_dir / f"{reference_stem}.npy")

    target_poses = [
        helpers.load_pose(model_dir / f"{view:03d}.npy")
        for view in target_views
    ]
    target_images = [
        helpers.load_rgba_on_white(model_dir / f"{view:03d}.png", resolution)[0]
        for view in target_views
    ]
    poses = stack_poses(helpers, reference_pose, target_poses, device)
    reference = reference.to(device=device, dtype=torch.float16)
    generator = torch.Generator(device=device).manual_seed(args.seed)

    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    print(
        f"Joint generation: resolution={resolution}, T_in=1, "
        f"T_out={len(target_views)}, steps={args.steps}",
        flush=True,
    )
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        output = pipeline(
            input_imgs=reference,
            prompt_imgs=reference,
            poses=poses,
            height=resolution,
            width=resolution,
            T_in=1,
            T_out=len(target_views),
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.steps,
            generator=generator,
            output_type="numpy",
        ).images
    elapsed = time.perf_counter() - started
    peak_gib = torch.cuda.max_memory_allocated(device) / 2**30

    predictions = [np.asarray(image, dtype=np.float32) for image in output]
    ground_truth = [
        np.asarray(image, dtype=np.float32) / 255.0 for image in target_images
    ]
    descriptions = [camera_description(pose) for pose in target_poses]
    scores = [psnr(gt, pred) for gt, pred in zip(ground_truth, predictions)]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    generated_pil: list[Image.Image] = []
    for view, prediction in zip(target_views, predictions):
        image = Image.fromarray(
            np.uint8(np.clip(prediction * 255.0, 0, 255)), mode="RGB"
        )
        image.save(args.output_dir / f"generated-{view:03d}.png")
        generated_pil.append(image)

    contact_sheet = args.output_dir / "contact-sheet.png"
    save_contact_sheet(
        reference_pil,
        target_images,
        generated_pil,
        target_views,
        descriptions,
        contact_sheet,
    )

    report = {
        "status": "success",
        "model": args.model,
        "reference_view": args.reference_view,
        "target_views": target_views,
        "resolution": resolution,
        "steps": args.steps,
        "guidance_scale": args.guidance_scale,
        "seed": args.seed,
        "joint_inference": True,
        "T_in": 1,
        "T_out": len(target_views),
        "elapsed_seconds": elapsed,
        "peak_allocated_vram_gib": peak_gib,
        "mean_psnr_db": float(np.mean(scores)),
        "gpu": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "python": platform.python_version(),
        "views": [
            {"view": view, **description, "psnr_db": score}
            for view, description, score in zip(
                target_views, descriptions, scores
            )
        ],
    }
    return predictions, report


def main() -> int:
    args = parse_args()
    args.eschernet_root = args.eschernet_root.resolve()
    args.output_dir = args.output_dir.resolve()
    helpers = load_smoke_helpers(Path(__file__).resolve().parent)
    helpers.add_eschernet_imports(args.eschernet_root)

    if not torch.cuda.is_available():
        raise RuntimeError("This validation requires a CUDA GPU")
    if args.resolution % 8:
        raise ValueError("--resolution must be divisible by 8")
    if not args.target_views:
        raise ValueError("At least one --target-views value is required")

    device = torch.device("cuda:0")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    pipeline = helpers.load_pipeline(args.model, device)

    # Preserve joint multi-view inference while adapting to small GPUs. With
    # the successful 2.027 GiB T_out=1 run, the first configuration is the
    # expected path; later configurations are OOM fallbacks only.
    configurations: list[tuple[int, list[int]]] = [
        (args.resolution, args.target_views),
    ]
    if len(args.target_views) > 2:
        configurations.append((args.resolution, args.target_views[:2]))
    if args.resolution > 128:
        configurations.append((128, args.target_views))
        if len(args.target_views) > 2:
            configurations.append((128, args.target_views[:2]))

    attempted: set[tuple[int, tuple[int, ...]]] = set()
    last_oom: RuntimeError | None = None
    for resolution, target_views in configurations:
        key = (resolution, tuple(target_views))
        if key in attempted:
            continue
        attempted.add(key)
        try:
            _, report = run_configuration(
                pipeline,
                helpers,
                args,
                device,
                target_views,
                resolution,
            )
            report_path = args.output_dir / "report.json"
            report_path.write_text(
                json.dumps(report, indent=2, allow_nan=False) + "\n",
                encoding="utf-8",
            )
            print(
                f"SUCCESS: joint T_out={report['T_out']} at "
                f"{report['resolution']}px in {report['elapsed_seconds']:.2f}s",
                flush=True,
            )
            print(
                f"Peak allocated VRAM: "
                f"{report['peak_allocated_vram_gib']:.3f} GiB",
                flush=True,
            )
            print(f"Mean PSNR: {report['mean_psnr_db']:.2f} dB", flush=True)
            print(f"Contact sheet: {args.output_dir / 'contact-sheet.png'}")
            print(f"Report: {report_path}")
            return 0
        except RuntimeError as error:
            if "out of memory" not in str(error).lower():
                raise
            last_oom = error
            print(
                f"CUDA OOM at {resolution}px with T_out={len(target_views)}; "
                "trying the next joint configuration",
                flush=True,
            )
            gc.collect()
            torch.cuda.empty_cache()

    raise RuntimeError("All joint multi-view configurations exhausted CUDA memory") from last_oom


if __name__ == "__main__":
    raise SystemExit(main())
