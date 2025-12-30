import os, re, shutil
from datetime import datetime
import numpy as np
import cv2
import imageio.v3 as iio
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display, clear_output

def read_video(path):

    # Read a video (avi/mp4) and return grayscale frames
    cap = cv2.VideoCapture(path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:               
            break
        # MP4 videos get inverted grayscale
        invertir = path.lower().endswith(".mp4")
        if invertir:
            frames.append(255 - cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        else:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    cap.release()

    return np.stack(frames, axis=0)


def save_config_txt(conf_path, payload: dict):
    # aquí aseguramos que exista la carpeta donde se guarda el .txt 
    os.makedirs(os.path.dirname(conf_path), exist_ok=True)

    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if os.path.exists(conf_path):
        stem = os.path.splitext(os.path.basename(conf_path))[0]
        archive_dir = os.path.join(os.path.dirname(conf_path), "archive", stem)

        # aquí aseguramos que exista la carpeta de historial 
        os.makedirs(archive_dir, exist_ok=True)

        existing = [f for f in os.listdir(archive_dir) if f.startswith("conf") and f.endswith(".txt")]
        nums = []
        for f in existing:
            m = re.match(r"conf(\d+)_", f)
            if m:
                nums.append(int(m.group(1)))
        n = (max(nums) + 1) if nums else 1

        archived = os.path.join(archive_dir, f"conf{n}_{datetime.now().strftime('%Y%m%d-%H%M%S')}.txt")
        shutil.move(conf_path, archived)

    lines = [f"# saved_at: {stamp}"]
    for k, v in payload.items():
        lines.append(f"{k} = {v}")

    with open(conf_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def video_crop_threshold_ui(videos_dir="videos",conf_dir="conf_",selection_min=1,selection_max=19,
                            default_selection=1 ,cmap="gray"):
    
    state = {"frames": None,"video_path": None,"frames_modificado": None,"cache": {},"lock": False,
        "ref_frame": None}
    files = []
    # Read videos available (MOESM pattern)
    if os.path.isdir(videos_dir): 
        for fn in os.listdir(videos_dir):
            if re.search(r"_MOESM\d+_ESM\.(avi|mp4)$", fn, flags=re.IGNORECASE):
                files.append(fn)
    files.sort()
    # Map number with video filename
    moesm_map = {} 
    for fn in files:
        m = re.search(r"_MOESM(\d+)_ESM\.(avi|mp4)$", fn, flags=re.IGNORECASE)
        if m:
            moesm_map[int(m.group(1))] = fn
    # Map selection index with label
    options = []  
    for sel in range(selection_min, selection_max + 1):
        moesm = sel + 1
        fn = moesm_map.get(moesm, None)
        label = f"{sel:02d}  →  MOESM{moesm}   ({fn if fn else 'NO ENCONTRADO'})"
        options.append((label, sel))
    # Widgets 
    w_select = widgets.Dropdown(options=options, value=default_selection, description="Video",layout=widgets.Layout(width="760px"))
    w_frame = widgets.IntSlider(value=0, min=0, max=0, step=1, description="Frame",continuous_update=False, layout=widgets.Layout(width="760px"))
    w_xrng = widgets.IntRangeSlider(value=[0, 1], min=0, max=1, step=1, description="X",continuous_update=False, layout=widgets.Layout(width="760px"))
    w_yrng = widgets.IntRangeSlider(value=[0, 1], min=0, max=1, step=1, description="Y",continuous_update=False, layout=widgets.Layout(width="760px"))
    w_x0 = widgets.BoundedIntText(value=0, min=0, max=1, description="x0", layout=widgets.Layout(width="190px"))
    w_x1 = widgets.BoundedIntText(value=1, min=0, max=1, description="x1", layout=widgets.Layout(width="190px"))
    w_y0 = widgets.BoundedIntText(value=0, min=0, max=1, description="y0", layout=widgets.Layout(width="190px"))
    w_y1 = widgets.BoundedIntText(value=1, min=0, max=1, description="y1", layout=widgets.Layout(width="190px"))
    w_thr = widgets.IntSlider(value=10, min=0, max=255, step=1, description="Threshold",continuous_update=False, layout=widgets.Layout(width="760px"))
    w_save = widgets.Button(description="Save configuration", button_style="success")
    w_reset = widgets.Button(description="Reset", button_style="")
    w_find_ref = widgets.Button(description="Frame 4CHECk", button_style="warning")
    w_load = widgets.Button(description="Load last configuration", button_style="info")
    w_status = widgets.HTML(value="")
    out = widgets.Output()
    # Read current parameters
    def _current_params():
        x0, x1 = w_xrng.value
        y0, y1 = w_yrng.value
        x0, x1 = sorted([int(x0), int(x1)])
        y0, y1 = sorted([int(y0), int(y1)])
        if x1 == x0: x1 = min(w_xrng.max, x0 + 1)
        if y1 == y0: y1 = min(w_yrng.max, y0 + 1)
        return x0, x1, y0, y1, int(w_thr.value), int(w_frame.value)

    # Render: izquierda original con ROI; derecha recorte ya umbralizado
    def _render():
        frames = state["frames"]
        x0, x1, y0, y1, thr, fid = _current_params()

        fr = frames[fid]  # frame original

        # aquí aplicamos umbral inline (en vez de _apply_threshold)
        fr_thr = np.where(fr < thr, 0, fr)

        crop = fr_thr[y0:y1, x0:x1]  # recorte

        with out:
            clear_output(wait=True)

            fig = plt.figure(figsize=(10.5, 5))
            ax1 = fig.add_subplot(1, 2, 1)
            ax2 = fig.add_subplot(1, 2, 2)

            ax1.imshow(fr, cmap=cmap)
            ax1.add_patch(plt.Rectangle((x0, y0), x1-x0, y1-y0, fill=False, linewidth=2, edgecolor="red"))
            title_ref = f" | ref={state['ref_frame']}" if state["ref_frame"] is not None else ""
            ax1.set_title(f"ORIGINAL — Frame {fid}{title_ref}")
            ax1.set_xlabel("x"); ax1.set_ylabel("y")

            ax2.imshow(crop, cmap=cmap)
            ax2.set_title(f"Recorte (umbral={thr})  y[{y0}:{y1}] x[{x0}:{x1}]")
            ax2.axis("off")

            plt.tight_layout()
            plt.show()

            print(f"Slice: frames[:, {y0}:{y1}, {x0}:{x1}]")
            print(f"Umbral: frame_thr = np.where(frame < {thr}, 0, frame)")

    # Sync sliders <-> inputs numéricos
    def _sync_from_slider_to_text():
        if state["lock"]:
            return
        state["lock"] = True
        try:
            x0, x1 = w_xrng.value
            y0, y1 = w_yrng.value
            w_x0.value, w_x1.value = int(min(x0, x1)), int(max(x0, x1))
            w_y0.value, w_y1.value = int(min(y0, y1)), int(max(y0, y1))
        finally:
            state["lock"] = False

    def _sync_from_text_to_slider():
        if state["lock"]:
            return
        state["lock"] = True
        try:
            x0, x1 = int(w_x0.value), int(w_x1.value)
            y0, y1 = int(w_y0.value), int(w_y1.value)
            w_xrng.value = (min(x0, x1), max(x0, x1))
            w_yrng.value = (min(y0, y1), max(y0, y1))
        finally:
            state["lock"] = False

    # Ajustar límites según el video (rangos, defaults X=50% y Y=80%, umbral por dtype)
    def _update_bounds_from_frames():
        frames = state["frames"]
        T, H, W = frames.shape

        w_frame.max = T - 1
        w_frame.value = min(w_frame.value, T - 1)

        w_xrng.min, w_xrng.max = 0, W - 1
        w_yrng.min, w_yrng.max = 0, H - 1

        w_xrng.value = (0, max(1, int(0.5 * (W - 1))))
        w_yrng.value = (0, max(1, int(0.8 * (H - 1))))

        for w in (w_x0, w_x1):
            w.min, w.max = 0, W - 1
        for w in (w_y0, w_y1):
            w.min, w.max = 0, H - 1

        _sync_from_slider_to_text()

        dt = frames.dtype
        if np.issubdtype(dt, np.integer):
            info = np.iinfo(dt)
            w_thr.min = int(info.min)
            w_thr.max = int(info.max)
        else:
            w_thr.min = int(np.floor(np.min(frames)))
            w_thr.max = int(np.ceil(np.max(frames)))

        w_thr.value = int(w_thr.min + 0.3 * (w_thr.max - w_thr.min))
        state["ref_frame"] = None

    # Cargar video desde dropdown (con cache)
    def _load_selected_video(sel_value):
        moesm = int(sel_value) + 1
        fn = moesm_map.get(moesm, None)
        
        path = os.path.join(videos_dir, fn)
        state["video_path"] = path
        w_status.value = f"<b>Video:</b> {path}"

        if path in state["cache"]:
            state["frames"] = state["cache"][path]
        else:
            state["frames"] = read_video(path)
            state["cache"][path] = state["frames"]

        _update_bounds_from_frames()
        _render()

    # Frame con más “blancos” (pixeles >= umbral) para usar como referencia
    def on_find_ref(_):
        frames = state["frames"]
        thr = int(w_thr.value)

        T = frames.shape[0]
        areas = np.empty(T, dtype=np.int64)

        chunk = 50
        for i in range(0, T, chunk):
            block = frames[i:i+chunk]
            areas[i:i+chunk] = np.count_nonzero(block >= thr, axis=(1, 2))

        ref = int(np.argmax(areas))
        state["ref_frame"] = ref
        w_frame.value = ref

        w_status.value = (f"<b style='color:#d80'>Ref frame:</b> {ref} "
            f"(área blanca máx con thr={thr}: {areas[ref]} píxeles)")
        _render()

    # Save: aplicar umbral + recorte a TODO el video y guardar config en txt
    def on_save(_):
        frames = state["frames"]
        x0, x1, y0, y1, thr, fid = _current_params()

        # aquí aplicamos umbral al video completo (inline)
        fr_thr = np.where(frames < thr, 0, frames)

        # aquí recortamos todo el video
        frames_mod = fr_thr[:, y0:y1, x0:x1]
        state["frames_modificado"] = frames_mod

        video_base = os.path.splitext(os.path.basename(state["video_path"]))[0]
        conf_path = os.path.join(conf_dir, f"{video_base}.txt")

        payload = dict(video_path=state["video_path"],selected_frame_id=fid,ref_frame_max_white=state["ref_frame"],
            threshold=thr,x0=x0, x1=x1, y0=y0, y1=y1,original_shape=str(frames.shape),output_shape=str(frames_mod.shape),
            slice=f"frames[:, {y0}:{y1}, {x0}:{x1}]")

        save_config_txt(conf_path, payload)

        w_status.value = (f"<b style='color:#070'>Guardado:</b> {conf_path}"
            f"<br><b>frames_modificado</b> listo (shape={frames_mod.shape}).")

    def on_reset(_):
        if state["frames"] is None:
            return
        _update_bounds_from_frames()
        _render()

    def on_select_change(change):
        _load_selected_video(change["new"])

    def on_any_change(change):
        _sync_from_slider_to_text()
        _render()

    def on_text_change(change):
        _sync_from_text_to_slider()
        _render()


    # leer la ultima configuracion   
    def _read_conf_txt(conf_path):
        data = {}
        with open(conf_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    k, v = line.split("=", 1)
                    data[k.strip()] = v.strip()
        return data

    def on_load_last(_):
        # necesitamos un video cargado
        if state["video_path"] is None:
            w_status.value = "<b style='color:#b00'>No hay video cargado todavía</b>"
            return
    
        video_base = os.path.splitext(os.path.basename(state["video_path"]))[0]
        conf_path = os.path.join(conf_dir, f"{video_base}.txt")
    
        if not os.path.exists(conf_path):
            w_status.value = f"<b style='color:#b00'>No hay configuraciones guardadas para:</b> {video_base}"
            return
    
        conf = _read_conf_txt(conf_path)
    
        # valores esperados
        try:
            # seteo directo (inputs numéricos)
            w_x0.value = int(conf.get("x0", w_x0.value))
            w_x1.value = int(conf.get("x1", w_x1.value))
            w_y0.value = int(conf.get("y0", w_y0.value))
            w_y1.value = int(conf.get("y1", w_y1.value))
            w_thr.value = int(conf.get("threshold", w_thr.value))
    
            # frame guardado (si existe)
            if "selected_frame_id" in conf and conf["selected_frame_id"] != "None":
                w_frame.value = int(conf["selected_frame_id"])
    
            # sincroniza sliders desde los inputs + render
            _sync_from_text_to_slider()
            _render()
    
            w_status.value = f"<b style='color:#070'>Cargada última configuración:</b> {conf_path}"
    
        except Exception as e:
            w_status.value = f"<b style='color:#b00'>No pude cargar la config:</b> {e}"

            
    # Conexión de eventos
    w_select.observe(on_select_change, names="value")
    for w in (w_frame, w_xrng, w_yrng, w_thr):
        w.observe(on_any_change, names="value")
    for w in (w_x0, w_x1, w_y0, w_y1):
        w.observe(on_text_change, names="value")

    w_save.on_click(on_save)
    w_reset.on_click(on_reset)
    w_find_ref.on_click(on_find_ref)
    w_load.on_click(on_load_last)


    # Layout final
    box_inputs = widgets.HBox([w_x0, w_x1, w_y0, w_y1])
    box_buttons = widgets.HBox([w_save, w_reset, w_find_ref, w_load])
    ui = widgets.VBox([w_select, w_frame, w_xrng, w_yrng, box_inputs, w_thr, box_buttons, w_status, out])
    display(ui)

    _load_selected_video(w_select.value)
    return state