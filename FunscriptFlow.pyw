#!/usr/bin/env python3

import sys, os, traceback, json, argparse  # stdlib only, safe to import

STUB_LOG = os.path.join(os.path.dirname(sys.argv[0]) if sys.argv else ".", "startup.log")
EXE_DIR = os.path.dirname(sys.argv[0]) if sys.argv and os.path.dirname(sys.argv[0]) else "."

# Disable GPU providers to avoid DLL init failures from missing DirectX
os.environ["ORT_DISABLE_DIRECTML"] = "1"

# Nuitka puts DLLs in the exe directory — add to PATH so native library loaders find them
os.environ["PATH"] = EXE_DIR + os.pathsep + os.environ.get("PATH", "")
if hasattr(os, 'add_dll_directory'):
    try:
        os.add_dll_directory(EXE_DIR)
    except Exception:
        pass

HAS_ONNX = False
try:
    import gc
    import math, threading, concurrent.futures
    import numpy as np
    import cv2
    from decord import VideoReader, cpu
    import tkinter as tk
    from tkinter import filedialog, messagebox
    import tkinter.ttk as ttk
except Exception:
    try:
        with open(STUB_LOG, "w") as f:
            traceback.print_exc(file=f)
    except Exception:
        pass
    try:
        import tkinter as tk
        from tkinter import messagebox
        messagebox.showerror("Import Error", traceback.format_exc())
    except Exception:
        pass
    sys.exit(1)


# ---------- Localization Strings ----------
def load_strings(filename="strings.json"):
    defaults = {
        "app_title": "Funscript Flow",
        "select_videos": "Select Videos",
        "select_folder": "Select Folder",
        "no_files_selected": "No files selected",
        "vr_mode": "VR Mode",
        "vr_mode_tooltip": ("Use this to improve accuracy for VR videos."),
        "overall_progress": "Overall Progress:",
        "current_video_progress": "Current Video Progress:",
        "advanced_settings": "Advanced Settings",
        "threads": "Threads:",
        "detrend_window": "Detrend window (sec):",
        "norm_window": "Norm window (sec):",
        "batch_size": "Batch size (frames):",
        "face_inversion": "Enable face-based inversion",
        "show_preview": "Show Preview",
        "show_advanced": "Show Advanced Settings",
        "overwrite_files": "Overwrite existing files",
        "run": "Run",
        "cancel": "Cancel",
        "readme": "Readme",
        "config_saved": "Config saved to {config_path}",
        "config_load_error": "Error loading config: {error}",
        "no_files_warning": "Please select one or more video files or a folder.",
        "cancelled_by_user": "Processing cancelled by user.",
        "batch_processing_complete": "Batch processing complete.",
        "funscript_saved": "Funscript saved: {output_path}",
        "skipping_file_exists": "Skipping {video_path}: {output_path} exists.",
        "log_error": "ERROR: Could not write output: {error}",
        "found_files": "Found {n} file(s).",
        "processing_file": "--- Processing file {current}/{total}: {video_path} ---",
        "processing_completed_with_errors": "Processing completed with errors. See run.log for details.",
        "face_inversion_tooltip": "Uses face detection to try to determine the angle of motion, and adjust direction accordingly.",
        "pov_mode_tooltip": "Use this to improve stability for POV videos.",
    }
    try:
        with open(filename, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return defaults

STRINGS = load_strings()

# ---------- Tooltip Implementation ----------
class ToolTip:
    def __init__(self, widget, text="widget info"):
        self.widget = widget
        self.text = text
        self.tipwindow = None
        widget.bind("<Enter>", self.enter)
        widget.bind("<Leave>", self.leave)
    def enter(self, event=None):
        self.showtip()
    def leave(self, event=None):
        self.hidetip()
    def showtip(self):
        if self.tipwindow or not self.text:
            return
        x, y, cx, cy = self.widget.bbox("insert")
        x = x + self.widget.winfo_rootx() + 25
        y = y + cy + self.widget.winfo_rooty() + 25
        self.tipwindow = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry("+%d+%d" % (x, y))
        label = tk.Label(tw, text=self.text, justify=tk.LEFT,
                         background="#ffffe0", relief=tk.SOLID, borderwidth=1,
                         font=("tahoma", "8", "normal"))
        label.pack(ipadx=1)
    def hidetip(self):
        if self.tipwindow:
            self.tipwindow.destroy()
        self.tipwindow = None

def max_divergence(flow):
    div = np.gradient(flow[..., 0], axis=0) + np.gradient(flow[..., 1], axis=1)
    y, x = np.unravel_index(np.argmax(np.abs(div)), div.shape)
    return x, y, div[y, x]


def radial_motion_weighted(flow, center, is_cut, pov_mode=False, balance_global=True):
    if(is_cut):
        return 0.0
    h, w, _ = flow.shape
    y, x = np.indices((h, w))
    dx = x - center[0]
    dy = y - center[1]

    dot = flow[..., 0] * dx + flow[..., 1] * dy

    if(pov_mode or not balance_global):
        return np.mean(dot)
    
    weighted_dot = np.where(x > center[0], dot * (w - x) / w, dot * x / w)
    weighted_dot = np.where(y > center[1], weighted_dot * (h - y) / h, weighted_dot * y / h)

    return np.mean(weighted_dot)


class Detector:
    NUDENET_PENIS_CLASSES = {4, 14}
    NUDENET_FACE_CLASSES = {1, 12}
    CUSTOM_PENIS_CLASSES = {0}
    CUSTOM_FACE_CLASSES = {1}

    def __init__(self, model_path="detector.onnx"):
        self.model = None
        self.enabled = False
        self.input_size = (640, 640)
        self.input_name = "images"
        self.use_nudenet = "nudenet" in model_path.lower() or "320n" in model_path.lower() or "640m" in model_path.lower()
        if not os.path.exists(model_path):
            return
        try:
            # Replace pip-installed ORT DLLs with official Microsoft builds
            _ort_src = os.path.join(EXE_DIR, "ort")
            _ort_dst = os.path.join(EXE_DIR, "onnxruntime", "capi")
            if os.path.isdir(_ort_src) and os.path.isdir(_ort_dst):
                import shutil
                for _f in os.listdir(_ort_src):
                    if _f.endswith(".dll"):
                        _src = os.path.join(_ort_src, _f)
                        _dst = os.path.join(_ort_dst, _f)
                        try:
                            shutil.copy2(_src, _dst)
                        except Exception:
                            pass
            import onnxruntime as ort
            self.model = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
            self.input_name = self.model.get_inputs()[0].name
            self.input_size = self.model.get_inputs()[0].shape[2:]
            self.enabled = True
        except Exception:
            self.model = None
            self.enabled = False
            try:
                import traceback as _tb
                with open(STUB_LOG, "a") as _f:
                    _f.write(f"--- Detector init failed for {model_path} ---\n")
                    _f.write(f"EXE_DIR DLLs: {len([x for x in os.listdir(EXE_DIR) if x.endswith('.dll')])}\n")
                    _f.write(f"ort/ exists: {os.path.isdir(os.path.join(EXE_DIR, 'ort'))}\n")
                    _f.write(f"capi/ exists: {os.path.isdir(os.path.join(EXE_DIR, 'onnxruntime', 'capi'))}\n")
                    _tb.print_exc(file=_f)
            except Exception:
                pass

    def detect(self, frame_gray):
        if not self.enabled or not self.model:
            return None, None
        try:
            h, w = frame_gray.shape
            img = cv2.resize(frame_gray, self.input_size)
            img = img.astype(np.float32) / 255.0
            if len(img.shape) == 2:
                img = np.stack([img, img, img], axis=-1)
            img = np.transpose(img, (2, 0, 1))
            img = np.expand_dims(img, axis=0)
            outputs = self.model.run(None, {self.input_name: img})[0]
            penis_box = None
            face_box = None
            penis_classes = self.NUDENET_PENIS_CLASSES if self.use_nudenet else self.CUSTOM_PENIS_CLASSES
            face_classes = self.NUDENET_FACE_CLASSES if self.use_nudenet else self.CUSTOM_FACE_CLASSES
            for det in outputs[0]:
                x1, y1, x2, y2, conf, cls = det[:6]
                x1, y1 = int(x1 * w / self.input_size[0]), int(y1 * h / self.input_size[1])
                x2, y2 = int(x2 * w / self.input_size[0]), int(y2 * h / self.input_size[1])
                cls = int(cls)
                if conf < 0.4:
                    continue
                box = (max(0, x1), max(0, y1), min(w, x2), min(h, y2), conf)
                if cls in penis_classes:
                    if penis_box is None or conf > penis_box[4]:
                        penis_box = box
                elif cls in face_classes:
                    if face_box is None or conf > face_box[4]:
                        face_box = box
            return penis_box, face_box
        except Exception:
            return None, None


def flow_in_box(flow, box):
    x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
    x1, y1 = int(max(0, x1)), int(max(0, y1))
    x2, y2 = int(min(flow.shape[1], x2)), int(min(flow.shape[0], y2))
    if x2 <= x1 or y2 <= y1:
        return 0.0
    region = flow[y1:y2, x1:x2]
    return np.mean(np.abs(region[..., 0] + region[..., 1]))


def compute_detection_signal(flow_info, detections, pair_idx, center, mode_state, pov_mode, balance_global, effective_fps):
    signal = 0.0
    cut = flow_info["cut"]
    
    if not detections or pair_idx + 1 >= len(detections):
        mode_state["current_mode"] = "legacy"
        if not cut:
            signal = radial_motion_weighted(flow_info["flow"], center, False, pov_mode, balance_global)
        return signal, mode_state

    det0 = detections[pair_idx]
    det1 = detections[pair_idx + 1]
    if det0 is None or det1 is None:
        mode_state["idle_count"] = mode_state.get("idle_count", 0) + 1
        if mode_state.get("idle_count", 0) > int(effective_fps):
            mode_state["current_mode"] = "idle"
            return 0.0, mode_state
        if not cut:
            signal = radial_motion_weighted(flow_info["flow"], center, False, pov_mode, balance_global)
        mode_state["current_mode"] = "legacy"
        return signal, mode_state

    penis0, face0 = det0
    penis1, face1 = det1
    penis_present = penis0 is not None and penis1 is not None
    face_present = face0 is not None and face1 is not None
    both_present = penis_present and face_present

    if both_present:
        mode_state["dual_streak"] = mode_state.get("dual_streak", 0) + 1
    else:
        mode_state["dual_streak"] = 0

    if mode_state.get("dual_streak", 0) >= 5:
        d0 = _box_distance(penis0, face0)
        d1 = _box_distance(penis1, face1)
        if d0 > 0:
            signal = (d1 - d0) / d0
        else:
            signal = 0.0
        mode_state["current_mode"] = "dual-box"
        mode_state["idle_count"] = 0
        return signal, mode_state

    if penis_present:
        if mode_state.get("idle_count", 0) >= 3:
            mode_state["idle_count"] = 0
        signal = flow_in_box(flow_info["flow"], penis1)
        mode_state["current_mode"] = "single-box"
        mode_state["idle_count"] = 0
        return signal, mode_state

    mode_state["idle_count"] = mode_state.get("idle_count", 0) + 1
    if mode_state.get("idle_count", 0) > int(effective_fps):
        mode_state["current_mode"] = "idle"
        return 0.0, mode_state

    mode_state["current_mode"] = "legacy"
    if not cut:
        signal = radial_motion_weighted(flow_info["flow"], center, False, pov_mode, balance_global)
    return signal, mode_state


def _box_distance(box1, box2):
    cx1 = (box1[0] + box1[2]) / 2
    cy1 = (box1[1] + box1[3]) / 2
    cx2 = (box2[0] + box2[2]) / 2
    cy2 = (box2[1] + box2[3]) / 2
    return math.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2)


def precompute_flow_info(p0, p1, config):
    cut_threshold = config.get("cut_threshold", 7)
    
    flow = cv2.calcOpticalFlowFarneback(p0, p1, None,
                                        0.5, 3, 15, 3, 5, 1.2, 0)
    if(config.get("pov_mode")):
        divergence_max = (p0.shape[1] // 2, p0.shape[0] - 1, 0)
    else:
        divergence_max = max_divergence(flow)
    pos_center = divergence_max[0:2]
    val_pos = divergence_max[2]
        
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    mean_mag = np.mean(mag)
    is_cut = mean_mag > cut_threshold

    return {
        "flow": flow,
        "pos_center": pos_center,
        "val_pos": val_pos,
        "cut": is_cut
    }

def fetch_frames(video_path, chunk, params):
    frames_gray = []
    try:
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=params["threads"], width=512 if params.get("vr_mode") else 256, height=512 if params.get("vr_mode") else 256)
        batch_frames = vr.get_batch(chunk).asnumpy()
    except Exception as e:
        return frames_gray
    vr = None
    gc.collect()

    for f in batch_frames:
        if params.get("vr_mode"):
            h, w, _ = f.shape
            gray = cv2.cvtColor(f[h // 2:, :w // 2], cv2.COLOR_RGB2GRAY)
        else:
            gray = cv2.cvtColor(f, cv2.COLOR_RGB2GRAY)
        frames_gray.append(gray)

    return frames_gray

# ---------- Main Processing Function ----------
def process_video(video_path, params, log_func, progress_callback=None, cancel_flag=None, preview_callback=None):
    error_occurred = False
    base, _ = os.path.splitext(video_path)
    output_path = base + ".funscript"
    if os.path.exists(output_path) and not params["overwrite"]:
        log_func(f"Skipping: output file exists ({output_path})")
        return error_occurred

    try:
        log_func(f"Processing video: {video_path}")
        vr = VideoReader(video_path, ctx=cpu(0), width=1024, height=1024, num_threads=params["threads"])
    except Exception as e:
        log_func(f"ERROR: Unable to open video at {video_path}: {e}")
        return True

    try:
        total_frames = len(vr)
        fps = vr.get_avg_fps()
    except Exception as e:
        log_func(f"ERROR: Unable to read video properties: {e}")
        return True

    step = max(1, int(math.ceil(fps / 30.0)))
    effective_fps = fps / step
    indices = list(range(0, total_frames, step))
    log_func(f"FPS: {fps:.2f}; downsampled to ~{effective_fps:.2f} fps; {len(indices)} frames selected.")

    bracket_size = int(params.get("batch_size", 3000.0))

    detector = None
    detection_enabled = params.get("detection_enabled", True)
    exe_dir = os.path.dirname(sys.argv[0]) if sys.argv else "."
    model_candidates = [
        os.path.join(exe_dir, "detector.onnx"),
        os.path.join(exe_dir, "320n.onnx"),
        os.path.join(exe_dir, "640m.onnx"),
    ]
    detector_model = None
    for candidate in model_candidates:
        if os.path.exists(candidate):
            detector_model = candidate
            break
    if detector_model is None:
        detector_model = model_candidates[0]  # default, will fail gracefully
    if detection_enabled:
        detector = Detector(detector_model)
        if detector.enabled:
            log_func("Detection: enabled")
        else:
            log_func("Detection: model not found or failed to load, using legacy mode")
    else:
        log_func("Detection: disabled")

    mode_state = {"dual_streak": 0, "idle_count": 0, "current_mode": "legacy"}

    final_flow_list = []

    next_batch = None
    fetch_thread = None
    final_con_list = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=params["threads"]) as ex:
     for chunk_start in range(0, len(indices), bracket_size):
        if cancel_flag and cancel_flag():
            log_func("User bailed.")
            return error_occurred

        chunk = indices[chunk_start:chunk_start + bracket_size]
        frame_indices = chunk[:-1]
        if len(chunk) < 2:
            continue

        if fetch_thread:
            fetch_thread.join()
            frames_gray = next_batch if next_batch is not None else fetch_frames(video_path, chunk, params)
            next_batch = None
        else:
            frames_gray = fetch_frames(video_path, chunk, params)

        if not frames_gray:
            log_func(f"ERROR: Unable to fetch frames for chunk {chunk_start} - skipping.")
            continue
        if chunk_start + bracket_size < len(indices):
            next_chunk = indices[chunk_start + bracket_size:chunk_start + 2 * bracket_size]
            def fetch_and_store():
                nonlocal next_batch
                next_batch = fetch_frames(video_path, next_chunk, params)

            fetch_thread = threading.Thread(target=fetch_and_store)
            fetch_thread.start()

        pairs = list(zip(frames_gray[:-1], frames_gray[1:]))

        flow_futures = [ex.submit(precompute_flow_info, p[0], p[1], params) for p in pairs]
        precomputed = [f.result() for f in flow_futures]

        final_con_list.extend([abs(info["val_pos"] * 10) for info in precomputed])

        final_centers = []
        for j, info in enumerate(precomputed):
            center_list = [info["pos_center"]]
            for i in range(1, 7):
                if j - i >= 0:
                    center_list.append(precomputed[j - i]["pos_center"])
                if j + i < len(precomputed):
                    center_list.append(precomputed[j + i]["pos_center"])
            center_list = np.array(center_list)
            center = np.mean(center_list, axis=0)
            final_centers.append(center)

        # Run detection on batch if enabled
        detections = []
        if detector and detector.enabled:
            for f in frames_gray:
                detections.append(detector.detect(f))

        # Compute signals using detection when available
        for j, info in enumerate(precomputed):
            signal, mode_state = compute_detection_signal(
                info, detections, j, final_centers[j],
                mode_state, params.get("pov_mode", False),
                params.get("balance_global", True), effective_fps)
            final_flow_list.append((signal, info["cut"], frame_indices[j]))

        # Display detection mode status in log
        if detector and detector.enabled and detections:
            modes = {"dual-box": 0, "single-box": 0, "legacy": 0, "idle": 0}
            # Sample mode from last computed state
            modes[mode_state.get("current_mode", "legacy")] = 1

        if progress_callback:
            prog = min(100, int(100 * (chunk_start + len(chunk)) / len(indices)))
            progress_callback(prog)

    cum_flow = [0]
    time_stamps = [final_flow_list[0][2]]

    for i in range(1, len(final_flow_list)):
        flow_prev, cut_prev, t_prev = final_flow_list[i - 1]
        flow_curr, cut_curr, t_curr = final_flow_list[i]

        if cut_curr:
            cum_flow.append(0)
        else:
            mid_flow = (flow_prev + flow_curr) / 2
            cum_flow.append(cum_flow[-1] + mid_flow)

        time_stamps.append(t_curr)

    cum_flow = [(cum_flow[i] + cum_flow[i-1]) / 2 if i > 0 else cum_flow[i] for i in range(len(cum_flow))]

    detrend_win = int(params["detrend_window"] * effective_fps)
    disc_threshold = 1000

    detrended_data = np.zeros_like(cum_flow)
    weight_sum = np.zeros_like(cum_flow)

    disc_indices = np.where(np.abs(np.diff(cum_flow)) > disc_threshold)[0] + 1
    segment_boundaries = [0] + list(disc_indices) + [len(cum_flow)]

    overlap = detrend_win // 2

    for i in range(len(segment_boundaries) - 1):
        seg_start = segment_boundaries[i]
        seg_end = segment_boundaries[i + 1]
        seg_length = seg_end - seg_start

        if seg_length < 5:
            detrended_data[seg_start:seg_end] = cum_flow[seg_start:seg_end] - np.mean(cum_flow[seg_start:seg_end])
            continue
        if seg_length <= detrend_win:
            segment = cum_flow[seg_start:seg_end]
            x = np.arange(len(segment))
            trend = np.polyfit(x, segment, 1)
            detrended_segment = segment - np.polyval(trend, x)
            weights = np.hanning(len(segment))
            detrended_data[seg_start:seg_end] += detrended_segment * weights
            weight_sum[seg_start:seg_end] += weights
        else:
            for start in range(seg_start, seg_end - overlap, overlap):
                end = min(start + detrend_win, seg_end)
                segment = cum_flow[start:end]
                x = np.arange(len(segment))
                trend = np.polyfit(x, segment, 1)
                detrended_segment = segment - np.polyval(trend, x)
                weights = np.hanning(len(segment))
                detrended_data[start:end] += detrended_segment * weights
                weight_sum[start:end] += weights

    detrended_data /= np.maximum(weight_sum, 1e-6)

    smoothed_data = np.convolve(detrended_data, [1/16, 1/4, 3/8, 1/4, 1/16], mode='same')
    norm_win = int(params["norm_window"] * effective_fps)
    if norm_win % 2 == 0:
        norm_win += 1
    half_norm = norm_win // 2
    norm_rolling = np.empty_like(smoothed_data)
    for i in range(len(smoothed_data)):
        start_idx = max(0, i - half_norm)
        end_idx = min(len(smoothed_data), i + half_norm + 1)
        local_window = smoothed_data[start_idx:end_idx]
        local_min = local_window.min()
        local_max = local_window.max()
        if local_max - local_min == 0:
            norm_rolling[i] = 50
        else:
            norm_rolling[i] = (smoothed_data[i] - local_min) / (local_max - local_min) * 100

    if(params["keyframe_reduction"]):
        key_indices = [0]
        for i in range(1, len(norm_rolling) - 1):
            d1 = norm_rolling[i] - norm_rolling[i - 1]
            d2 = norm_rolling[i + 1] - norm_rolling[i]
            
            if (d1 < 0) != (d2 < 0):
                key_indices.append(i)
        key_indices.append(len(norm_rolling) - 1)
    else:
        key_indices = range(len(norm_rolling))
    actions = []
    for ki in key_indices:  
        try:
            timestamp_ms = int(((time_stamps[ki]) / fps) * 1000)
            pos = int(round(norm_rolling[ki]))
            actions.append({"at": timestamp_ms, "pos": 100-pos})
        except Exception as e:
            log_func(f"Error computing action at segment index {ki}: {e}")
            error_occurred = True

    log_func(f"Keyframe reduction: {len(actions)} actions computed.")
    funscript = {"version": "1.0", "actions": actions}
    try:
        with open(output_path, "w") as f:
            json.dump(funscript, f, indent=2)
        log_func(STRINGS["funscript_saved"].format(output_path=output_path))
    except Exception as e:
        log_func(STRINGS["log_error"].format(error=str(e)))
        error_occurred = True
    return error_occurred

# ---------- Preview Helper ----------
def convert_frame_to_photo(frame):
    try:
        rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        retval, buffer = cv2.imencode('.png', rgb)
        if not retval:
            return None
        img_data = buffer.tobytes()
        return tk.PhotoImage(data=img_data)
    except Exception:
        return None

# ---------- GUI Code ----------
def disable_widgets_except(widget, exceptions):
    if widget not in exceptions:
        try:
            widget.configure(state="disabled")
        except tk.TclError:
            pass
    for child in widget.winfo_children():
        disable_widgets_except(child, exceptions)

def enable_widgets(widget):
    try:
        widget.configure(state="normal")
    except tk.TclError:
        pass
    for child in widget.winfo_children():
        enable_widgets(child)

class App:
    def __init__(self, master):
        self.master = master
        master.title(STRINGS["app_title"])
        try:
            icon = tk.PhotoImage(file="icon.png")
            master.iconphoto(False, icon)
        except Exception:
            pass
        self.files = []
        self.cancel_event = threading.Event()
        self.error_occurred = False
        self.preview_window = None
        self.preview_label = None
        self.show_preview = tk.BooleanVar(value=False)
        self.show_adv = tk.BooleanVar(value=False)
        
        top_frame = tk.Frame(master)
        top_frame.pack(fill=tk.X, padx=5, pady=5)
        btn_sel_files = tk.Button(top_frame, text=STRINGS["select_videos"], command=self.select_files)
        btn_sel_files.pack(side=tk.LEFT, padx=2)
        btn_sel_folder = tk.Button(top_frame, text=STRINGS["select_folder"], command=self.select_folder)
        btn_sel_folder.pack(side=tk.LEFT, padx=2)
        self.lbl_files = tk.Label(top_frame, text=STRINGS["no_files_selected"])
        self.lbl_files.pack(side=tk.LEFT, padx=10)
        btn_readme = tk.Button(top_frame, text=STRINGS["readme"], command=self.show_readme)
        btn_readme.pack(side=tk.RIGHT, padx=2)
        
        mode_frame = tk.Frame(master)
        mode_frame.pack(fill=tk.X, padx=5, pady=2)
        self.vr_mode = tk.BooleanVar(value=False)
        chk_vr = tk.Checkbutton(mode_frame, text=STRINGS["vr_mode"], variable=self.vr_mode)
        chk_vr.pack(side=tk.LEFT, padx=2)
        ToolTip(chk_vr, STRINGS["vr_mode_tooltip"])
        chk_preview = tk.Checkbutton(mode_frame, text=STRINGS["show_preview"], variable=self.show_preview)
        self.pov_mode = tk.BooleanVar(value=False)
        chk_pov = tk.Checkbutton(mode_frame, text="POV Mode", variable=self.pov_mode)
        chk_pov.pack(side=tk.LEFT, padx=2)
        self.balance_global = tk.BooleanVar(value=True)
        chk_balance = tk.Checkbutton(mode_frame, text="Balance Global Motion", variable=self.balance_global)
        chk_balance.pack(side=tk.LEFT, padx=2)
        self.detection_enabled = tk.BooleanVar(value=True)
        chk_detect = tk.Checkbutton(mode_frame, text="Enable Detection", variable=self.detection_enabled)
        chk_detect.pack(side=tk.LEFT, padx=2)
        ToolTip(chk_detect, "Uses computer vision to track subject motion. Disable to use pure optical flow.")
        ToolTip(chk_balance, "If enabled, the script will try to cancel out camera motion. Disable it for scenes with no camera movement.")
        ToolTip(chk_pov, STRINGS["pov_mode_tooltip"])
        
        adv_toggle_frame = tk.Frame(master)
        adv_toggle_frame.pack(fill=tk.X, padx=5, pady=2)
        chk_adv = tk.Checkbutton(adv_toggle_frame, text=STRINGS["show_advanced"] if "show_advanced" in STRINGS else "Show Advanced Settings", variable=self.show_adv, command=self.toggle_advanced)
        chk_adv.pack(side=tk.LEFT, padx=2)
        
        prog_frame = tk.Frame(master)
        prog_frame.pack(fill=tk.X, padx=5, pady=5)
        tk.Label(prog_frame, text=STRINGS["overall_progress"]).pack(anchor=tk.W)
        self.overall_progress = ttk.Progressbar(prog_frame, orient="horizontal", mode="determinate", maximum=100)
        self.overall_progress.pack(fill=tk.X, padx=5, pady=2)
        tk.Label(prog_frame, text=STRINGS["current_video_progress"]).pack(anchor=tk.W)
        self.video_progress = ttk.Progressbar(prog_frame, orient="horizontal", mode="determinate", maximum=100)
        self.video_progress.pack(fill=tk.X, padx=5, pady=2)
        self.lbl_detect_status = tk.Label(prog_frame, text="Detector: N/A")
        self.lbl_detect_status.pack(anchor=tk.W, padx=5)
        
        self.adv_frame = tk.LabelFrame(master, text=STRINGS["advanced_settings"])
        self.adv_frame.pack(fill=tk.X, padx=5, pady=5)
        self.params = {}
        num_cores = os.cpu_count()
        self.create_param(self.adv_frame, STRINGS["threads"], "threads", str(num_cores), "Number of threads used for optical flow computation.")
        self.create_param(self.adv_frame, STRINGS["detrend_window"], "detrend_window", "1.5", "Controls the aggressiveness of drift removal. See readme for detail.  Recommended: 1-10, higher values for more stable cameras.")
        
        self.create_param(self.adv_frame, STRINGS["norm_window"], "norm_window", "4", "Time window to calibrate motion range (seconds). Shorter values amplify motion, but also cause artifacts in long thrusts.")
        self.create_param(self.adv_frame, STRINGS["batch_size"], "batch_size", "3000", "Number of frames to process per batch (Higher values will be faster, but also take more RAM).")
        
        self.keyframe_reduction = tk.BooleanVar(value=True)
        chk_keyframe = tk.Checkbutton(self.adv_frame, text=STRINGS["chk_keyframe"] if "chk_keyframe" in STRINGS else "Enable keyframe reduction", variable=self.keyframe_reduction)
        chk_keyframe.pack(anchor=tk.W, padx=5, pady=2)
        
        self.overwrite = tk.BooleanVar(value=False)
        chk_overwrite = tk.Checkbutton(self.adv_frame, text=STRINGS["overwrite_files"], variable=self.overwrite)
        chk_overwrite.pack(anchor=tk.W, padx=5, pady=2)
        
        btn_frame = tk.Frame(master)
        btn_frame.pack(padx=5, pady=5)
        btn_run = tk.Button(btn_frame, text=STRINGS["run"], command=self.run_batch)
        btn_run.pack(side=tk.LEFT, padx=5)
        self.btn_cancel = tk.Button(btn_frame, text=STRINGS["cancel"], command=self.cancel_run)
        self.btn_cancel.pack(side=tk.LEFT, padx=5)
        
        self.log_file = None
        self.error_occurred = False
        self.load_config()
        self.toggle_advanced()
    
    def create_param(self, parent, label_text, key, default, tooltip_text):
        frm = tk.Frame(parent)
        frm.pack(fill=tk.X, padx=5, pady=2)
        lbl = tk.Label(frm, text=label_text, width=25, anchor=tk.W)
        lbl.pack(side=tk.LEFT)
        entry = tk.Entry(frm)
        entry.insert(0, default)
        entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.params[key] = entry
        ToolTip(entry, tooltip_text)
    
    def select_files(self):
        files = filedialog.askopenfilenames(title=STRINGS["select_videos"],
                    filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv *.m4v"), ("All Files", "*.*")])
        if files:
            self.files = list(files)
            self.lbl_files.config(text=f"{len(self.files)} file(s) selected")
        else:
            self.files = []
            self.lbl_files.config(text=STRINGS["no_files_selected"])
    
    def select_folder(self):
        folder = filedialog.askdirectory(title=STRINGS["select_folder"])
        if folder:
            video_exts = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}
            found = []
            for root, dirs, files in os.walk(folder):
                for f in files:
                    ext = os.path.splitext(f)[1].lower()
                    if ext in video_exts:
                        found.append(os.path.join(root, f))
            self.files = found
            self.lbl_files.config(text=f"{len(self.files)} file(s) found in folder")
    
    def show_readme(self):
        try:
            with open("readme.txt", "r", encoding="utf-8") as f:
                content = f.read()
        except Exception as e:
            content = f"Error reading readme.txt: {e}"
        win = tk.Toplevel(self.master)
        win.title("Readme")
        txt = tk.Text(win, wrap=tk.WORD)
        txt.insert(tk.END, content)
        txt.pack(fill=tk.BOTH, expand=True)
    
    def update_preview(self, frame):
        photo = convert_frame_to_photo(frame)
        if photo is None:
            return
        if self.preview_window is None:
            self.preview_window = tk.Toplevel(self.master)
            self.preview_window.title("Preview")
            self.preview_label = tk.Label(self.preview_window)
            self.preview_label.pack()
        self.preview_label.configure(image=photo)
        self.preview_label.image = photo
    
    def save_config(self):
        config = { key: self.params[key].get() for key in self.params }
        config["overwrite"] = self.overwrite.get()
        config["vr_mode"] = self.vr_mode.get()
        config["pov_mode"] = self.pov_mode.get()
        config["balance_global"] = self.balance_global.get()
        config["detection_enabled"] = self.detection_enabled.get()
        
        config_path = "config.json"
        try:
            with open(config_path, "w") as f:
                json.dump(config, f, indent=2)
            messagebox.showinfo("Config Saved", STRINGS["config_saved"].format(config_path=config_path))
        except Exception as e:
            messagebox.showerror("Error", f"Could not save config: {e}")
    
    def load_config(self):
        config_path = "config.json"
        if os.path.exists(config_path):
            try:
                with open(config_path, "r") as f:
                    config = json.load(f)
                for key in self.params:
                    if key in config:
                        self.params[key].delete(0, tk.END)
                        self.params[key].insert(0, str(config[key]))
                self.overwrite.set(config.get("overwrite", False))
                self.vr_mode.set(config.get("vr_mode", False))
                self.pov_mode.set(config.get("pov_mode", False))
                self.balance_global.set(config.get("balance_global", True))
                self.detection_enabled.set(config.get("detection_enabled", True))
                
            except Exception as e:
                messagebox.showwarning("Config Load", STRINGS["config_load_error"].format(error=str(e)))
    
    def toggle_advanced(self):
        if self.show_adv.get():
            self.adv_frame.pack(fill=tk.X, padx=5, pady=5)
        else:
            self.adv_frame.forget()
    
    def update_video_progress(self, prog):
        self.master.after(0, lambda: self.video_progress.configure(value=prog))
    
    def update_overall_progress(self, prog):
        self.master.after(0, lambda: self.overall_progress.configure(value=prog))
    
    def cancel_run(self):
        self.cancel_event.set()
    
    def log(self, msg):
        self.log_file.write(msg + "\n")
        self.log_file.flush()
        
    def run_batch(self):
        if not self.files:
            messagebox.showwarning("No files", STRINGS["no_files_warning"])
            return
        try:
            settings = {
                "threads": int(self.params["threads"].get()),
                "detrend_window": float(self.params["detrend_window"].get()),
                
                "norm_window": float(self.params["norm_window"].get()),
                "batch_size": int(self.params["batch_size"].get()),
                "overwrite": self.overwrite.get(),
                "keyframe_reduction": self.keyframe_reduction.get()
            }
            settings["vr_mode"] = self.vr_mode.get()
            settings["pov_mode"] = self.pov_mode.get()
            settings["balance_global"] = self.balance_global.get()
            settings["detection_enabled"] = self.detection_enabled.get()
        except Exception as e:
            messagebox.showerror("Parameter Error", f"Invalid parameters: {e}")
            return
        
        self.cancel_event.clear()
        try:
            log_path = None
            if os.name == "posix":
                log_path = "/tmp"
            else:
                if not os.getenv("APPDATA"):
                    messagebox.showerror("Log Error", "APPDATA environment variable not set.")
                    return
                log_path = os.path.join(os.getenv("APPDATA"), "FunscriptFlow")
            os.makedirs(log_path, exist_ok=True)
            log_filename = os.path.join(log_path, "run.log")
            self.log_file = open(log_filename, "w", encoding="utf-8")
        except Exception as e:
            messagebox.showerror("Log Error", f"Cannot open log file: {e}")
            return
        self.overall_progress.configure(value=0)
        self.video_progress.configure(value=0)
        if settings.get("detection_enabled", True):
            self.lbl_detect_status.config(text="Detector: enabled")
        else:
            self.lbl_detect_status.config(text="Detector: disabled")
        disable_widgets_except(self.master, [self.btn_cancel])
        total_files = len(self.files)
        self.error_occurred = False
        def worker():
            for idx, video in enumerate(self.files):
                if self.cancel_event.is_set():
                    self.log(STRINGS["cancelled_by_user"])
                    break
                self.update_video_progress(0)
                err = process_video(video, settings, self.log,
                              progress_callback=lambda prog: self.update_video_progress(prog),
                              cancel_flag=lambda: self.cancel_event.is_set(),
                              preview_callback=lambda frame: self.update_preview(frame) if self.show_preview.get() else None)
                if err:
                    self.error_occurred = True
                overall = int(100 * (idx + 1) / total_files)
                self.update_overall_progress(overall)
            self.log(STRINGS["batch_processing_complete"])
            self.log_file.close()
            enable_widgets(self.master)
            if self.error_occurred:
                if messagebox.askyesno("Run Finished", STRINGS["processing_completed_with_errors"] + "\nWould you like to open the log?"):
                    open_log(log_filename)
            else:
                if messagebox.askyesno("Run Finished", "Batch processing complete.\nSee run.log for details.\nWould you like to open the log?"):
                    open_log(log_filename)
        threading.Thread(target=worker, daemon=True).start()

def open_log(log_filename):
    os.startfile(log_filename)
    
# ---------- Headless Mode ----------
def run_headless(input_path, settings):
    log_filename = "run.log"
    try:
        logf = open(log_filename, "w")
    except Exception as e:
        print(f"Error opening log file: {e}")
        return
    def log_func(msg):
        logf.write(msg + "\n")
        logf.flush()
        print(msg)
    if os.path.isdir(input_path):
        video_exts = {".mp4", ".avi", ".mov", ".mkv"}
        files = []
        for root, dirs, files_in in os.walk(input_path):
            for f in files_in:
                ext = os.path.splitext(f)[1].lower()
                if ext in video_exts:
                    files.append(os.path.join(root, f))
    else:
        files = [input_path]
    if not files:
        print("No video files found.")
        logf.write("No video files found.\n")
        logf.close()
        return
    total_files = len(files)
    log_func(STRINGS["found_files"].format(n=total_files))
    for idx, video in enumerate(files):
        log_func(STRINGS["processing_file"].format(current=idx+1, total=total_files, video_path=video))
        process_video(video, settings, log_func, progress_callback=lambda prog: print(f"Video progress: {prog}%"))
    log_func(STRINGS["batch_processing_complete"])
    logf.close()
    print("Done. See run.log for details.")

# ---------- Main ----------
if __name__ == '__main__':
    try:
        parser = argparse.ArgumentParser(description="Optical Flow to Funscript")
        parser.add_argument("input", nargs="?", help="Input video file or folder")
        parser.add_argument("--threads", type=int, default=8, help="Number of threads (default: 8)")
        parser.add_argument("--detrend_window", type=float, default=2.0, help="Detrend window in seconds (default: 2.0)")
        parser.add_argument("--norm_window", type=float, default=3.0, help="Normalization window in seconds (default: 3.0)")
        parser.add_argument("--batch_size", type=int, default=3000, help="Batch size in frames (default: 3000)")
        parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files")
        parser.add_argument("--vr_mode", action="store_true", help="Enable VR Mode (if not set, non-VR mode is used)")
        parser.add_argument("--pov_mode", action="store_true", help="Enable POV Mode (improves stability for POV videos)")
        parser.add_argument("--disable_keyframe_reduction", action="store_false", help="Disable keyframe reduction")
        args = parser.parse_args()
        settings = {
            "threads": args.threads,
            "detrend_window": args.detrend_window,
            "norm_window": args.norm_window,
            "batch_size": args.batch_size,
            "overwrite": args.overwrite,
            "vr_mode": args.vr_mode,
            "pov_mode": args.pov_mode,
            "keyframe_reduction": not args.disable_keyframe_reduction
        }
        if args.input:
            run_headless(args.input, settings)
        else:
            root = tk.Tk()
            app = App(root)
            root.mainloop()
    except Exception:
        err_log = os.path.join(os.path.dirname(sys.argv[0]), "startup.log")
        with open(err_log, "w") as f:
            traceback.print_exc(file=f)
        try:
            messagebox.showerror("Startup Error", traceback.format_exc())
        except Exception:
            pass
