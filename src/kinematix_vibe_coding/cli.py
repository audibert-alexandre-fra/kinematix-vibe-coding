"""Point d'entrée en ligne de commande.

    uv run kinematix run data/squat1/raw-video-cam0.mp4 --height 1.80 --out output
    uv run kinematix batch data --out output           # tailles lues dans data/heights.json
    uv run kinematix evaluate output                   # cohérence multi-vues
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv"}


def find_videos(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in VIDEO_EXT)


def height_for(video: Path, data_root: Path, heights: dict, default: float) -> float:
    """Taille (m) : clé = chemin relatif de la vidéo, sinon sous-dossier, sinon 'default'."""
    rel = video.relative_to(data_root)
    for key in (str(rel), str(rel.with_suffix("")), rel.parts[0]):
        if key in heights:
            return float(heights[key])
    return float(heights.get("default", default))


def output_dir_for(video: Path, data_root: Path, out_root: Path) -> Path:
    rel = video.relative_to(data_root)
    return out_root / rel.parent / rel.stem


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="kinematix", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--out", default="output")
        p.add_argument("--backend", default="instanthmr")
        p.add_argument("--device", default="cpu", help="cpu | coreml | cuda (backend ONNX)")
        p.add_argument("--no-render", action="store_true")
        p.add_argument("--no-anti-sliding", action="store_true")
        p.add_argument("--draw-keypoints", action="store_true")

    p_run = sub.add_parser("run", help="traite une vidéo")
    p_run.add_argument("video")
    p_run.add_argument("--height", type=float, required=True, help="taille du sujet (m)")
    common(p_run)

    p_batch = sub.add_parser("batch", help="traite toutes les vidéos d'un dossier")
    p_batch.add_argument("data", nargs="?", default="data")
    p_batch.add_argument("--heights", default=None, help="JSON des tailles (défaut: <data>/heights.json)")
    p_batch.add_argument("--default-height", type=float, default=1.75)
    common(p_batch)

    p_eval = sub.add_parser("evaluate", help="cohérence des angles entre caméras d'un même mouvement")
    p_eval.add_argument("out", nargs="?", default="output")

    args = parser.parse_args(argv)
    os.environ.setdefault("TQDM_DISABLE", "1")

    if args.cmd == "evaluate":
        from .utils.evaluation import evaluate_output

        evaluate_output(Path(args.out))
        return 0

    from .pipeline import PipelineConfig, run_pipeline

    def cfg(height: float) -> PipelineConfig:
        kw = {"device": args.device} if args.backend == "instanthmr" else {}
        return PipelineConfig(height=height, backend=args.backend, backend_kwargs=kw, render=not args.no_render,
                              anti_sliding=not args.no_anti_sliding, draw_keypoints=args.draw_keypoints)

    if args.cmd == "run":
        video = Path(args.video)
        run_pipeline(video, Path(args.out) / video.stem, cfg(args.height))
        return 0

    data_root = Path(args.data)
    hpath = Path(args.heights) if args.heights else data_root / "heights.json"
    heights = json.loads(hpath.read_text()) if hpath.exists() else {}
    results = {}
    for video in find_videos(data_root):
        h = height_for(video, data_root, heights, args.default_height)
        res = run_pipeline(video, output_dir_for(video, data_root, Path(args.out)), cfg(h))
        results[str(video)] = res.summary
    with open(Path(args.out) / "batch_summary.json", "w") as f:
        json.dump(results, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
