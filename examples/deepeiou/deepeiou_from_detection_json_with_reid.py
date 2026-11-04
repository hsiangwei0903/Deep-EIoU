"""Offline example: detection JSON + video -> ReID embeddings -> DeepEIoU.

Reads pre-computed COCO detections, loads matching RGB video frames, extracts
per-detection embeddings with OSNet/ResNet, then runs DeepEIoU tracking and
writes COCO tracking JSON.

Usage:
    uv run python examples/deepeiou/deepeiou_from_detection_json_with_reid.py \
        --input-json path/to/detections.json \
        --video-path path/to/source.mp4 \
        --output-json path/to/tracking_output.json \
        --with-reid \
        --reid-weights path/to/osnet_or_resnet_weights.pth.tar
"""

from __future__ import annotations

import sys
from pathlib import Path

import click
import torch
from loguru import logger

# Ensure plugin packages are importable when running from the repo root.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import cuvis_ai_deepeiou  # noqa: F401, E402 - triggers shims


def _resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise click.BadParameter("CUDA requested but not available.", param_hint="--device")
    return device


@click.command()
@click.option("--input-json", required=True, type=click.Path(exists=True), help="COCO detection JSON")
@click.option(
    "--video-path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Source RGB video used to compute ReID crops.",
)
@click.option("--output-json", required=True, type=click.Path(), help="Output tracking JSON")
@click.option("--with-reid/--no-reid", default=True, show_default=True, help="Enable/disable ReID")
@click.option(
    "--reid-weights",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Path to ReID weights (.pth.tar). Required when --with-reid is enabled.",
)
@click.option(
    "--backbone",
    type=click.Choice(["osnet", "resnet"]),
    default="osnet",
    show_default=True,
    help="ReID backbone architecture when --with-reid is enabled.",
)
@click.option(
    "--device",
    type=click.Choice(["auto", "cpu", "cuda"]),
    default="auto",
    show_default=True,
    help="Execution device for crop/normalization/extractor nodes.",
)
@click.option(
    "--save-embeddings-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Optional folder to save per-frame embeddings as .npy files.",
)
@click.option(
    "--frame-id-offset",
    default=0,
    type=int,
    show_default=True,
    help="Video frame index = detection image_id + offset.",
)
@click.option("--track-high-thresh", default=0.6, type=float, help="High-confidence threshold")
@click.option("--track-low-thresh", default=0.1, type=float, help="Low-confidence threshold")
@click.option("--new-track-thresh", default=0.7, type=float, help="New track threshold")
@click.option("--track-buffer", default=60, type=int, help="Lost-track buffer (frames)")
@click.option("--match-thresh", default=0.8, type=float, help="Match threshold")
@click.option("--frame-rate", default=30, type=int, help="Video frame rate")
@click.option("--proximity-thresh", default=0.5, type=float, help="Spatial gate for ReID fusion")
@click.option("--appearance-thresh", default=0.25, type=float, help="Embedding distance gate")
def main(
    input_json: str,
    video_path: Path,
    output_json: str,
    with_reid: bool,
    reid_weights: Path | None,
    backbone: str,
    device: str,
    save_embeddings_dir: Path | None,
    frame_id_offset: int,
    track_high_thresh: float,
    track_low_thresh: float,
    new_track_thresh: float,
    track_buffer: int,
    match_thresh: float,
    frame_rate: int,
    proximity_thresh: float,
    appearance_thresh: float,
) -> None:
    """Run DeepEIoU tracking on detection JSON with optional video-based ReID."""
    from cuvis_ai.node.json_reader import DetectionJsonReader
    from cuvis_ai.node.json_writer import ByteTrackCocoJson
    from cuvis_ai.node.numpy_writer import NumpyFeatureWriterNode
    from cuvis_ai.node.preprocessors import BBoxRoiCropNode
    from cuvis_ai.node.pretreatments.standardize import IMAGENET_MEAN, IMAGENET_STD, Standardize
    from cuvis_ai.node.video import VideoIterator

    from cuvis_ai_deepeiou.node import DeepEIoUTrack, OSNetExtractor, ResNetExtractor

    runtime_device = _resolve_device(device)

    if with_reid and reid_weights is None:
        raise click.BadParameter(
            "--reid-weights is required when --with-reid is enabled.",
            param_hint="--reid-weights",
        )
    if save_embeddings_dir is not None and not with_reid:
        raise click.BadParameter(
            "--save-embeddings-dir requires --with-reid.",
            param_hint="--save-embeddings-dir",
        )

    logger.info("Reading detections from {}", input_json)
    reader = DetectionJsonReader(json_path=input_json)
    video_iter = VideoIterator(str(video_path))

    tracker = DeepEIoUTrack(
        track_high_thresh=track_high_thresh,
        track_low_thresh=track_low_thresh,
        new_track_thresh=new_track_thresh,
        track_buffer=track_buffer,
        match_thresh=match_thresh,
        frame_rate=frame_rate,
        with_reid=with_reid,
        proximity_thresh=proximity_thresh,
        appearance_thresh=appearance_thresh,
    )

    writer = ByteTrackCocoJson(
        output_json_path=output_json,
        category_id_to_name={},
        flush_interval=1,
    )

    feature_writer = (
        NumpyFeatureWriterNode(output_dir=str(save_embeddings_dir))
        if save_embeddings_dir is not None
        else None
    )

    crop = None
    normalize = None
    extractor = None
    if with_reid:
        crop = BBoxRoiCropNode(output_size=(256, 128)).to(runtime_device)
        normalize = Standardize(
            stats="given", layout="nchw", mean=IMAGENET_MEAN, std=IMAGENET_STD
        ).to(runtime_device)
        extractor_cls = OSNetExtractor if backbone == "osnet" else ResNetExtractor
        extractor = extractor_cls(model_path=str(reid_weights)).to(runtime_device)
        logger.info(
            "ReID enabled: backbone={}, device={}, weights={}",
            backbone,
            runtime_device,
            reid_weights,
        )

    frame_count = 0
    reid_frames = 0
    fallback_frames = 0
    warned_hw_mismatch = False

    try:
        while True:
            det = reader.forward()
            frame_id = int(det["frame_id"][0].item())
            embeddings = None

            if with_reid:
                assert crop is not None and normalize is not None and extractor is not None

                video_frame_idx = frame_id + frame_id_offset
                if video_frame_idx < 0 or video_frame_idx >= len(video_iter):
                    raise click.ClickException(
                        f"Frame mapping out of range: detection image_id={frame_id}, "
                        f"offset={frame_id_offset}, video_idx={video_frame_idx}, "
                        f"video_frames={len(video_iter)}."
                    )

                frame_data = video_iter.get_frame(video_frame_idx)
                frame_bgr = frame_data["image"]
                if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
                    raise click.ClickException(
                        f"Invalid frame at index {video_frame_idx}: expected HxWx3 image."
                    )

                det_h = int(det["orig_hw"][0, 0].item())
                det_w = int(det["orig_hw"][0, 1].item())
                img_h, img_w = int(frame_bgr.shape[0]), int(frame_bgr.shape[1])
                if (
                    not warned_hw_mismatch
                    and det_h > 0
                    and det_w > 0
                    and (det_h != img_h or det_w != img_w)
                ):
                    logger.warning(
                        "JSON orig_hw={}x{} differs from video frame={}x{} (first warning only).",
                        det_h,
                        det_w,
                        img_h,
                        img_w,
                    )
                    warned_hw_mismatch = True

                frame_rgb = (
                    torch.from_numpy(frame_bgr[..., ::-1].copy())
                    .to(torch.float32)
                    .div(255.0)
                    .unsqueeze(0)
                    .to(runtime_device)
                )
                bboxes_dev = det["bboxes"].to(runtime_device)

                crops = crop.forward(images=frame_rgb, bboxes=bboxes_dev)["crops"]
                normalized = normalize.forward(cube=crops)["cube"]
                embeddings = extractor.forward(crops=normalized)["embeddings"]

                n_dets = int(det["bboxes"].shape[1])
                n_emb = int(embeddings.shape[1])
                if n_emb != n_dets:
                    logger.warning(
                        "frame {}: embeddings count {} != detections {}; fallback to EIoU-only",
                        frame_id,
                        n_emb,
                        n_dets,
                    )
                    embeddings = None
                    fallback_frames += 1
                else:
                    reid_frames += 1
                    if feature_writer is not None:
                        feature_writer.forward(features=embeddings, frame_id=det["frame_id"])

            tracker_kwargs = {
                "bboxes": det["bboxes"],
                "category_ids": det["category_ids"],
                "confidences": det["confidences"],
            }
            if embeddings is not None:
                tracker_kwargs["embeddings"] = embeddings

            track_out = tracker.forward(**tracker_kwargs)
            writer.forward(
                frame_id=det["frame_id"],
                bboxes=track_out["bboxes"],
                category_ids=track_out["category_ids"],
                confidences=track_out["confidences"],
                track_ids=track_out["track_ids"],
                orig_hw=det["orig_hw"],
            )
            frame_count += 1
    except StopIteration:
        pass
    finally:
        writer.close()

    logger.info("Processed {} frames -> {}", frame_count, output_json)
    if with_reid:
        logger.info("ReID frames={} fallback_frames={}", reid_frames, fallback_frames)
    if feature_writer is not None:
        logger.info("Saved embeddings -> {}", save_embeddings_dir)


if __name__ == "__main__":
    main()
