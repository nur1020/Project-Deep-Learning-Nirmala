import streamlit as st
import os
import gdown
import cv2
import numpy as np
from PIL import Image
import tempfile
import time
from pathlib import Path
import sys

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="AWAN Detector",
    page_icon="🌤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Kelas awan + cuaca yang diprediksi
CLOUD_WEATHER_MAP = {
    "Altocumulus":  {"cuaca": "Berawan Sedang",           "icon": "⛅",   "warna": "#6495ED"},
    "Nimbostratus": {"cuaca": "Hujan Lebat",              "icon": "🌧️",  "warna": "#2F4F4F"},
    "Cumulus":      {"cuaca": "Cerah",                    "icon": "⛅",   "warna": "#32CD32"},
    "Cumulonimbus": {"cuaca": "Badai / Hujan Deras",     "icon": "⛈️",   "warna": "#DC143C"},
}

def download_models():
    os.makedirs("models", exist_ok=True)
    
    files = {
        "models/Model_FasterRCNN_AWAN_best.pth": "1B7JoXIF5KUOEbRvZqzGkpvStyjGk-RXJ",
        "models/Model_SSD_AWAN_best.pth":         "1ojMR1qJhhMkSKI_3ELG9vyHay2ERNn5E",
        "models/Model_YOLO_AWAN_best.pt":         "1MfZyBQzqlsBbbcGlfeG78ItSdH5n5oJU",
    }
    
    for path, file_id in files.items():
        if not os.path.exists(path):
            with st.spinner(f"⬇️ Mendownload {path}..."):
                gdown.download(f"https://drive.google.com/uc?id={file_id}", path, quiet=False)

download_models()
# ─────────────────────────────────────────────
# LOAD MODEL (cached)
# ─────────────────────────────────────────────
@st.cache_resource
def load_model(model_name):
    """Load model sesuai pilihan. Returnvalue: model object."""
    try:
        if model_name == "YOLO":
            from ultralytics import YOLO
            model = YOLO("models/Model_YOLO_AWAN_best.pt")
            return ("yolo", model)

        elif model_name == "SSD":
            import torch
            import torchvision
            from torchvision.models.detection import ssd300_vgg16
            num_classes = 5   # +1 background
            model = ssd300_vgg16(weights=None)
            # Sesuaikan head jika perlu
            checkpoint = torch.load("models/Model_SSD_AWAN_best.pth",
                                    map_location=torch.device("cpu"))
            state = checkpoint.get("model_state_dict", checkpoint)
            model.load_state_dict(state, strict=False)
            model.eval()
            return ("ssd", model)

        elif model_name == "Faster R-CNN":
            import torch
            from torchvision.models.detection import fasterrcnn_resnet50_fpn
            num_classes = 5 # 4 kelas + background
            model = fasterrcnn_resnet50_fpn(weights=None, num_classes=num_classes)
            checkpoint = torch.load("models/Model_FasterRCNN_AWAN_best.pth",
                                    map_location=torch.device("cpu"))
            state = checkpoint.get("model_state_dict", checkpoint)
            model.load_state_dict(state, strict=False)
            model.eval()
            return ("fasterrcnn", model)

    except Exception as e:
        st.error(f"❌ Gagal load model {model_name}: {e}")
        return None


# ─────────────────────────────────────────────
# INFERENCE
# ─────────────────────────────────────────────
def predict_image(model_tuple, img_bgr, conf_threshold=0.4):
    """
    Jalankan inferensi. Return list of dicts:
    [{"label": str, "conf": float, "box": [x1,y1,x2,y2]}, ...]
    """
    if model_tuple is None:
        return []

    model_type, model = model_tuple
    results = []

    try:
        if model_type == "yolo":
            res = model.predict(img_bgr, conf=conf_threshold, verbose=False)
            for r in res:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    conf   = float(box.conf[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    label  = model.names.get(cls_id, f"Class {cls_id}")
                    results.append({"label": label, "conf": conf,
                                    "box": [x1, y1, x2, y2]})

        elif model_type in ("ssd", "fasterrcnn"):
            import torch
            import torchvision.transforms.functional as F

            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            tensor  = F.to_tensor(img_rgb).unsqueeze(0)

            with torch.no_grad():
                preds = model(tensor)[0]

            class_names = list(CLOUD_WEATHER_MAP.keys())
            boxes  = preds["boxes"].numpy()
            labels = preds["labels"].numpy()
            scores = preds["scores"].numpy()

            for box, lbl, score in zip(boxes, labels, scores):
                if score >= conf_threshold:
                    idx   = int(lbl) - 1  # 0 = background
                    name  = class_names[idx] if 0 <= idx < len(class_names) else f"Class {lbl}"
                    x1, y1, x2, y2 = map(int, box.tolist())
                    results.append({"label": name, "conf": float(score),
                                    "box": [x1, y1, x2, y2]})
    except Exception as e:
        st.warning(f"⚠️ Error saat prediksi: {e}")

    return results


def draw_results(img_bgr, detections):
    """Gambar bounding box + label ke gambar."""
    img = img_bgr.copy()
    for det in detections:
        label = det["label"]
        conf  = det["conf"]
        x1, y1, x2, y2 = det["box"]
        info  = CLOUD_WEATHER_MAP.get(label, {"cuaca": "Unknown", "icon": "❓"})

        # Warna kotak (BGR)
        color = (0, 200, 100)

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        text = f"{label} {conf:.0%}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(img, text, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return img


# ─────────────────────────────────────────────
# UI HELPER
# ─────────────────────────────────────────────
def show_detection_cards(detections):
    if not detections:
        st.info("Tidak ada awan terdeteksi. Coba turunkan threshold atau arahkan kamera ke langit.")
        return

    st.markdown("### 🔍 Hasil Deteksi")
    for det in detections:
        label = det["label"]
        conf  = det["conf"]
        info  = CLOUD_WEATHER_MAP.get(label, {"cuaca": "-", "icon": "❓", "warna": "#888"})

        with st.container():
            col1, col2 = st.columns([1, 4])
            with col1:
                st.markdown(f"<h1 style='text-align:center'>{info['icon']}</h1>",
                            unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
**Jenis Awan:** `{label}`  
**Cuaca Diprediksi:** {info['cuaca']}  
**Kepercayaan:** {conf:.1%}
""")
                st.progress(conf)
            st.divider()


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/4052/4052984.png", width=80)
    st.title("🌤️ AWAN Detector")
    st.markdown("Deteksi jenis awan & prediksi cuaca secara otomatis.")
    st.divider()

    model_choice = st.selectbox(
        "🤖 Pilih Model",
        ["YOLO", "SSD", "Faster R-CNN"],
        help="YOLO = tercepat, Faster R-CNN = paling akurat"
    )

    conf_threshold = st.slider(
        "🎯 Confidence Threshold", 0.1, 0.95, 0.40, 0.05,
        help="Semakin tinggi = lebih selektif"
    )

    st.divider()
    st.markdown("**Kelas Awan yang Didukung:**")
    for name, info in CLOUD_WEATHER_MAP.items():
        st.markdown(f"{info['icon']} {name}")


# ─────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────
with st.spinner(f"⏳ Loading model {model_choice}..."):
    model_tuple = load_model(model_choice)

if model_tuple is None:
    st.error("Model gagal dimuat. Pastikan file model ada di folder `models/`.")
    st.stop()

st.success(f"✅ Model **{model_choice}** siap digunakan!")

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab_cam, tab_img, tab_vid = st.tabs(["📷 Kamera Live", "🖼️ Upload Gambar", "🎬 Upload Video"])


# ══════════════════════════════════════════════
# TAB 1 – KAMERA LIVE
# ══════════════════════════════════════════════
with tab_cam:
    st.markdown("### 📷 Deteksi Realtime via Kamera")
    st.info("Arahkan kamera HP ke langit — deteksi berjalan otomatis!")

    from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
    import av

    class CloudDetector(VideoProcessorBase):
        def __init__(self):
            self.conf_threshold = conf_threshold
            self.model_tuple = model_tuple

        def recv(self, frame):
            img_bgr = frame.to_ndarray(format="bgr24")

            # Deteksi
            dets = predict_image(self.model_tuple, img_bgr, self.conf_threshold)

            # Gambar bounding box
            annotated = draw_results(img_bgr, dets)

            return av.VideoFrame.from_ndarray(annotated, format="bgr24")

    webrtc_streamer(
        key="cloud-detector",
        video_processor_factory=CloudDetector,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

# ══════════════════════════════════════════════
# TAB 2 – UPLOAD GAMBAR
# ══════════════════════════════════════════════
with tab_img:
    st.markdown("### 🖼️ Deteksi dari Gambar")
    uploaded_imgs = st.file_uploader(
        "Upload gambar awan (JPG / PNG / WEBP)",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True
    )

    if uploaded_imgs:
        for up_img in uploaded_imgs:
            st.markdown(f"---\n#### 📁 {up_img.name}")
            img_pil = Image.open(up_img).convert("RGB")
            img_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

            with st.spinner("🔍 Mendeteksi..."):
                dets = predict_image(model_tuple, img_bgr, conf_threshold)

            annotated = draw_results(img_bgr, dets)
            col1, col2 = st.columns(2)
            with col1:
                st.image(img_pil, caption="Gambar Asli", use_container_width=True)
            with col2:
                st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                         caption="Hasil Deteksi", use_container_width=True)

            show_detection_cards(dets)
    else:
        st.markdown(
            "<div style='text-align:center;padding:60px;background:#f0f2f6;"
            "border-radius:12px;font-size:36px'>🖼️<br>"
            "<small style='font-size:16px'>Upload gambar di atas</small></div>",
            unsafe_allow_html=True
        )


# ══════════════════════════════════════════════
# TAB 3 – UPLOAD VIDEO
# ══════════════════════════════════════════════
with tab_vid:
    st.markdown("### 🎬 Deteksi dari Video")
    uploaded_vid = st.file_uploader(
        "Upload video (MP4 / AVI / MOV)",
        type=["mp4", "avi", "mov", "mkv"]
    )

    if uploaded_vid:
        # Simpan ke file temp
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_vid.read())
        tfile.close()

        st.video(tfile.name)

        if st.button("▶️ Mulai Analisis Video"):
            cap = cv2.VideoCapture(tfile.name)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps          = cap.get(cv2.CAP_PROP_FPS) or 25
            process_every = max(1, int(fps // 2))  # Proses 2 frame/detik

            st.markdown(f"**Total frame:** {total_frames} | **FPS:** {fps:.1f}")

            progress_bar = st.progress(0)
            vid_window   = st.empty()
            info_window  = st.empty()

            all_dets     = []
            frame_idx    = 0
            last_dets    = []

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % process_every == 0:
                    last_dets = predict_image(model_tuple, frame, conf_threshold)
                    all_dets.extend(last_dets)

                annotated = draw_results(frame, last_dets)
                rgb        = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                vid_window.image(rgb, channels="RGB", use_container_width=True)

                if total_frames > 0:
                    progress_bar.progress(min(frame_idx / total_frames, 1.0))

                if last_dets:
                    top  = last_dets[0]
                    info = CLOUD_WEATHER_MAP.get(top["label"],
                                                  {"icon": "❓", "cuaca": "-"})
                    info_window.info(
                        f"{info['icon']} Frame {frame_idx}: "
                        f"**{top['label']}** — {info['cuaca']} "
                        f"({top['conf']:.0%})"
                    )

                frame_idx += 1

            cap.release()
            os.unlink(tfile.name)
            progress_bar.progress(1.0)
            st.success("✅ Analisis video selesai!")

            # Ringkasan
            if all_dets:
                from collections import Counter
                label_counts = Counter(d["label"] for d in all_dets)
                st.markdown("### 📊 Ringkasan Deteksi Video")
                for lbl, cnt in label_counts.most_common():
                    info = CLOUD_WEATHER_MAP.get(lbl, {"icon": "❓", "cuaca": "-"})
                    st.markdown(f"- {info['icon']} **{lbl}**: {cnt} deteksi — _{info['cuaca']}_")
    else:
        st.markdown(
            "<div style='text-align:center;padding:60px;background:#f0f2f6;"
            "border-radius:12px;font-size:36px'>🎬<br>"
            "<small style='font-size:16px'>Upload video di atas</small></div>",
            unsafe_allow_html=True
        )
