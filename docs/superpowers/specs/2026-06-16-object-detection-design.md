# Object Detection for FunscriptFlow

**Date:** 2026-06-16
**Status:** Draft

## Overview

Add YOLO-based object detection to the existing optical flow pipeline to track the penis and face, using their relative motion as the primary signal source instead of full-frame optical flow. This eliminates the "scripts all motion, no idle periods" limitation.

## Prerequisites

- A pre-trained YOLOv8n ONNX model with `penis` and `face` classes (training is a one-time external step, not runtime)
- Existing NSFW object detection dataset (e.g., Niche's dataset) or manual annotation of training images
- Training done offline; only the exported `detector.onnx` is shipped with the app

```
Video → decord → frames (batch)
                       │
                 YOLO model (ONNX)
                       │
                 penis + face boxes
                       │
      ┌────────────────▼────────────────────────┐
      │  Dual-box relative motion signal        │
      │  ┌──────────────────────────────────┐   │
      │  │ 1. center distance (d)           │   │
      │  │ 2. signal = (d - d_prev) / d     │   │
      │  │ 3. temporal smoothing (3-frame)  │   │
      │  └──────────────────────────────────┘   │
      │  (when dual-box)                        │
      │  OR                                      │
      │  Region-constrained radial flow          │
      │  (when single-box — penis only)          │
      └────────────────┬────────────────────────┘
                       │
   Integration → Detrend → Normalize → Keyframes → funscript
```

## Detection Model

- **Model:** YOLOv8n (nano)
- **Format:** ONNX (no PyTorch dependency; `onnxruntime` pip package)
- **Input:** 256×256 grayscale frames (matches existing downsampled pipeline)
- **Inference:** CPU-only, ~20-30ms per frame
- **Classes:** `penis` (required), `face` (optional)
- **Training:** Fine-tune last layers on existing NSFW dataset + COCO-pretrained face weights

**Shipping:** `detector.onnx` bundled in dist folder. If missing, app runs in legacy mode.

**Loading:** Once at startup. Kept in memory for duration of session.

## Signal Computation

### Dual-box mode (both penis and face detected)

```
distance = euclidean(center_penis, center_face)
raw_signal = (d_current - d_previous) / d_current
```

- Positive signal = boxes moving apart (outward stroke)
- Negative signal = boxes moving together (inward stroke)
- Normalized by distance for scale invariance
- Signal is signed — integrates naturally into existing pipeline which then detrends/normalizes

### Single-box mode (penis only, face missing)

Signal = weighted radial motion constrained to penis bounding box. Same math as current `radial_motion_weighted`, just cropped to the box region.

### Smoothing

3-frame moving average on box center positions before computing distance delta, to reduce YOLO jitter.

## Fallback Hierarchy

| Situation | Behavior |
|-----------|----------|
| Both boxes detected | Dual-box distance signal |
| Penis only (face off-frame) | Single-box region flow |
| Neither / penis missing <1s | Full-frame legacy flow |
| Neither / penis missing >1s | Idle (decay output to 50 over ~0.5s) |

**Hysteresis:**
- Require 3 consecutive confident frames to exit idle
- Require 5 consecutive confident dual-box frames to switch from single-box to dual-box mode
- Prevents rapid flipping between modes

## Idle Detection

State enters when:
- Penis box missing > 1 second
- OR face present but penis missing > 1 second (face alone is noise)

State exits when:
- Penis box reappears with confidence > 0.4

During idle: output position decays linearly to 50 over ~0.5 seconds.

Stationary subject (boxes present but distance unchanging): signal naturally trends to 0; detrending + normalization push output toward 50. No special case needed.

## Error Handling

| Scenario | Behavior |
|----------|----------|
| `detector.onnx` missing at startup | Log warning; run legacy mode |
| ONNX model fails to parse | Log warning; run legacy mode |
| YOLO inference crashes mid-processing | Use last known box, count toward grace timer |
| Box flip/flutter (rapid on/off) | Hysteresis layer prevents mode switching |
| Subject near frame edge (partial box) | Use box if >50% expected area; reject otherwise |
| Very closeup (boxes overlap, distance <10px) | Treat as single-box from merged region |
| Multiple penises detected | Take highest-confidence box |
| Video has no penis visible (wrong content) | Grace timer → idle → entire video outputs flat 50 |

## GUI Changes

- **New checkbox:** "Enable detection" in Advanced Settings (default: on)
  - Tooltip: "Uses computer vision to track subject motion. Disable to use pure optical flow."
- **Status indicator** in progress area during processing:
  - "Detector: dual-box" / "Detector: single-box" / "Detector: idle"

## Testing

| Test | Method |
|------|--------|
| ONNX model loads without error | Unit test |
| Dual-box distance signal computed correctly | Unit test with synthetic boxes |
| Fallback to single-box when face missing | Unit test |
| Idle decay when both boxes missing >1s | Unit test |
| Funscript output format unchanged | Integration test |
| Performance: <2x slowdown vs legacy | Benchmark on sample video |

## Files Changed

- `FunscriptFlow.pyw` — add `Detector` class, modify `precompute_flow_info` to accept boxes, modify `process_video` to run detection pass, modify signal computation, add GUI checkbox
- `requirements.txt` — add `onnxruntime`
- `.github/workflows/build.yml` — bundle `detector.onnx` in dist
