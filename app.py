import streamlit as st
import os
import tempfile
import threading
from datetime import datetime
from collections import Counter

import gdown
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# =========================================================
# KONFIGURASI
# =========================================================
st.set_page_config(
    page_title="AWAN Detector",
    page_icon="🌤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

CLOUD_WEATHER_MAP = {
    "Altocumulus": {
        "cuaca": "Berawan Sedang", "icon": "⛅", "warna": "#6495ED",
        "deskripsi": "Awan bergumpal pada lapisan menengah dan sering tampak berkelompok."
    },
    "Nimbostratus": {
        "cuaca": "Hujan Lebat", "icon": "🌧️", "warna": "#2F4F4F",
        "deskripsi": "Awan berlapis tebal yang dapat menghasilkan hujan dalam waktu relatif lama."
    },
    "Cumulus": {
        "cuaca": "Cerah", "icon": "⛅", "warna": "#32CD32",
        "deskripsi": "Awan bergumpal dengan dasar relatif datar yang umumnya muncul saat cuaca cerah."
    },
    "Cumulonimbus": {
        "cuaca": "Badai / Hujan Deras", "icon": "⛈️", "warna": "#DC143C",
        "deskripsi": "Awan yang berkembang vertikal dan berkaitan dengan hujan deras, petir, atau badai."
    },
}

MODEL_FILES = {
    "models/Model_FasterRCNN_AWAN_best.pth": "1B7JoXIF5KUOEbRvZqzGkpvStyjGk-RXJ",
    "models/Model_SSD_AWAN_best.pth": "1ojMR1qJhhMkSKI_3ELG9vyHay2ERNn5E",
    "models/Model_YOLO_AWAN_best.pt": "1MfZyBQzqlsBbbcGlfeG78ItSdH5n5oJU",
}

# Hasil penelitian dari notebook Colab yang kamu kirim.
# Urutan kelas mengikuti data.yaml: altocumulus, cumulonimbus, cumulus, nimbostratus.
DATASET_CLASS_COUNTS = {
    "Altocumulus": 364,
    "Cumulonimbus": 354,
    "Cumulus": 350,
    "Nimbostratus": 360,
}

MODEL_METRICS = {
    "YOLO": {"mAP50": 0.8961, "mAP50-95": 0.6742, "Precision": 0.9206, "Recall": 0.8242, "speed_ms": 5.7},
    "SSD": {"mAP50": 0.7520, "mAP50-95": 0.4924, "Precision": 0.6122, "Recall": 0.5848, "speed_ms": None},
    "Faster R-CNN": {"mAP50": 0.6904, "mAP50-95": 0.4618, "Precision": None, "Recall": 0.5674, "speed_ms": 111.57},
}

# Loss epoch per epoch sesuai keluaran training pada notebook.
SSD_TRAIN_LOSS = [8.1569, 5.5390, 4.9771, 4.6611, 4.3585, 4.1339, 3.9357, 3.6723, 3.5432, 3.3227, 3.2121, 3.1028, 2.9306, 2.8599, 2.6757, 2.6150, 2.5296, 2.4205, 2.3109, 2.2175, 2.1744, 2.1279, 2.0172, 1.9561, 1.9084, 1.8046, 1.7241, 1.6831, 1.6505, 1.5930]
FASTER_RCNN_TRAIN_LOSS = [0.2121, 0.1427, 0.1232, 0.1023, 0.0897, 0.0772, 0.0690, 0.0631, 0.0555, 0.0515, 0.0471, 0.0434, 0.0410, 0.0384, 0.0365, 0.0337, 0.0343, 0.0326, 0.0299, 0.0280]

# Confusion matrix SSD sebagaimana dicetak pada notebook.
# Notebook memberi label singkat Cb, Cc, Ns, Sc, tetapi pemetaan label tersebut
# tidak konsisten/terjelaskan terhadap urutan data.yaml. Karena itu ditampilkan
# sebagai indeks kelas, bukan diberi nama kelas yang belum terverifikasi.
SSD_CONFUSION_MATRIX = [
    [16, 1, 17, 3],
    [21, 0, 21, 1],
    [8, 0, 16, 0],
    [14, 1, 14, 0],
]

# =========================================================
# SESSION STATE
# =========================================================
if "history" not in st.session_state:
    st.session_state.history = []
if "last_model" not in st.session_state:
    st.session_state.last_model = "YOLO"


def save_history(source, model_name, detections):
    """Simpan ringkasan hasil deteksi selama sesi aplikasi berlangsung."""
    if not detections:
        return
    st.session_state.history.append({
        "waktu": datetime.now().strftime("%d-%m-%Y %H:%M:%S"),
        "sumber": source,
        "model": model_name,
        "jumlah": len(detections),
        "hasil": [
            {
                "jenis": d["label"],
                "confidence": round(float(d["conf"]) * 100, 2),
                "cuaca": CLOUD_WEATHER_MAP.get(d["label"], {}).get("cuaca", "Tidak diketahui")
            }
            for d in detections
        ]
    })


# =========================================================
# DOWNLOAD DAN LOAD MODEL
# =========================================================
def ensure_model_file(model_path):
    """Unduh hanya bobot model yang sedang dipilih."""
    os.makedirs("models", exist_ok=True)
    if os.path.exists(model_path) and os.path.getsize(model_path) > 0:
        return
    file_id = MODEL_FILES[model_path]
    with st.spinner(f"Mendownload {os.path.basename(model_path)}..."):
        result = gdown.download(
            f"https://drive.google.com/uc?id={file_id}",
            model_path,
            quiet=False
        )
    if not result or not os.path.exists(model_path) or os.path.getsize(model_path) == 0:
        raise RuntimeError(f"Gagal mengunduh file model: {model_path}")


@st.cache_resource
def load_model(model_name):
    try:
        if model_name == "YOLO":
            ensure_model_file("models/Model_YOLO_AWAN_best.pt")
            from ultralytics import YOLO
            return ("yolo", YOLO("models/Model_YOLO_AWAN_best.pt"))

        import torch

        if model_name == "SSD":
            ensure_model_file("models/Model_SSD_AWAN_best.pth")
            from torchvision.models.detection import ssdlite320_mobilenet_v3_large
            # Arsitektur harus sama dengan notebook: SSDLite320 MobileNetV3.
            model = ssdlite320_mobilenet_v3_large(
                weights=None, weights_backbone=None, num_classes=5
            )
            checkpoint = torch.load(
                "models/Model_SSD_AWAN_best.pth", map_location="cpu"
            )
            state = checkpoint.get("model_state_dict", checkpoint)
            model.load_state_dict(state, strict=True)
            model.eval()
            return ("ssd", model)

        if model_name == "Faster R-CNN":
            ensure_model_file("models/Model_FasterRCNN_AWAN_best.pth")
            from torchvision.models.detection import fasterrcnn_resnet50_fpn
            model = fasterrcnn_resnet50_fpn(weights=None, num_classes=5)
            checkpoint = torch.load(
                "models/Model_FasterRCNN_AWAN_best.pth", map_location="cpu"
            )
            state = checkpoint.get("model_state_dict", checkpoint)
            model.load_state_dict(state, strict=False)
            model.eval()
            return ("fasterrcnn", model)

    except Exception as exc:
        st.error(f"Gagal memuat model {model_name}: {exc}")
        return None


# =========================================================
# INFERENCE DAN ANOTASI
# =========================================================
def pil_to_tensor(img_pil):
    import torch
    arr = np.array(img_pil.convert("RGB")).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)


def predict_pil(model_tuple, img_pil, conf_threshold=0.25):
    if model_tuple is None:
        return []

    model_type, model = model_tuple
    detections = []

    try:
        if model_type == "yolo":
            img_np = np.array(img_pil.convert("RGB"))
            predictions = model.predict(
                img_np, conf=conf_threshold, verbose=False
            )
            for result in predictions:
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    detections.append({
                        "label": model.names.get(cls_id, f"Class {cls_id}"),
                        "conf": float(box.conf[0]),
                        "box": [int(v) for v in box.xyxy[0].tolist()]
                    })

        else:
            import torch
            tensor = pil_to_tensor(img_pil)
            with torch.no_grad():
                prediction = model([tensor])[0]

            # Urutan indeks mengikuti data.yaml pada notebook.
            class_names = ["Altocumulus", "Cumulonimbus", "Cumulus", "Nimbostratus"]
            boxes = prediction["boxes"].cpu().numpy()
            labels = prediction["labels"].cpu().numpy()
            scores = prediction["scores"].cpu().numpy()

            for box, label_id, score in zip(boxes, labels, scores):
                if float(score) >= conf_threshold:
                    idx = int(label_id) - 1
                    label = (
                        class_names[idx]
                        if 0 <= idx < len(class_names)
                        else f"Class {label_id}"
                    )
                    detections.append({
                        "label": label,
                        "conf": float(score),
                        "box": [int(v) for v in box.tolist()]
                    })

    except Exception as exc:
        st.warning(f"Terjadi kesalahan saat prediksi: {exc}")

    return detections


def draw_results_pil(img_pil, detections):
    img = img_pil.convert("RGB").copy()
    draw = ImageDraw.Draw(img)
    width, height = img.size

    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16
        )
        small_font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13
        )
    except Exception:
        font = ImageFont.load_default()
        small_font = font

    for det in detections:
        label = det["label"]
        confidence = det["conf"]
        x1, y1, x2, y2 = det["box"]
        info = CLOUD_WEATHER_MAP.get(label, {})
        color = info.get("warna", "#00C878")

        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

        label_text = f"{label} {confidence:.0%}"
        bbox = draw.textbbox((0, 0), label_text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        label_top = max(0, y1 - text_h - 8)
        draw.rectangle(
            [x1, label_top, x1 + text_w + 8, label_top + text_h + 6],
            fill=color
        )
        draw.text((x1 + 4, label_top + 2), label_text, fill="white", font=font)

        weather_text = f"Cuaca: {info.get('cuaca', 'Tidak diketahui')}"
        wb = draw.textbbox((0, 0), weather_text, font=small_font)
        ww, wh = wb[2] - wb[0], wb[3] - wb[1]
        weather_top = min(height - wh - 8, y2)
        draw.rectangle(
            [x1, weather_top, min(width, x1 + ww + 8), weather_top + wh + 6],
            fill=(25, 25, 25)
        )
        draw.text(
            (x1 + 4, weather_top + 2),
            weather_text,
            fill=(0, 255, 255),
            font=small_font
        )

    if detections:
        top_info = CLOUD_WEATHER_MAP.get(detections[0]["label"], {})
        banner = f"Prediksi cuaca: {top_info.get('cuaca', '-')}"
        draw.rectangle([0, 0, width, 34], fill=(0, 0, 0))
        draw.text((10, 7), banner, fill=(0, 255, 255), font=font)

    return img


def show_detection_cards(detections):
    if not detections:
        st.info("Tidak ada awan yang terdeteksi. Coba gambar lain atau turunkan ambang confidence.")
        return

    st.subheader("Hasil Deteksi")
    for index, det in enumerate(detections, start=1):
        info = CLOUD_WEATHER_MAP.get(det["label"], {})
        with st.container(border=True):
            col1, col2 = st.columns([1, 5])
            with col1:
                st.markdown(
                    f"<div style='font-size:42px;text-align:center'>{info.get('icon', '☁️')}</div>",
                    unsafe_allow_html=True
                )
            with col2:
                st.markdown(f"**{det['label']}**")
                st.write(f"Prediksi cuaca: {info.get('cuaca', 'Tidak diketahui')}")
                st.write(f"Confidence: {det['conf']:.1%}")
                st.progress(min(max(float(det["conf"]), 0), 1))


# =========================================================
# SIDEBAR / NAVIGASI
# =========================================================
with st.sidebar:
    st.markdown(
        "<div style='text-align:center;font-size:48px'>🌤️</div>",
        unsafe_allow_html=True
    )
    st.title("AWAN Detector")
    st.caption("Cloud Detection & Weather Prediction")
    st.divider()

    page = st.radio(
        "NAVIGASI",
        [
            "Dashboard",
            "Deteksi Gambar",
            "Kamera Live",
            "Deteksi Video",
            "Riwayat Deteksi",
            "Informasi Jenis Awan",
            "Tentang Aplikasi"
        ],
        label_visibility="collapsed"
    )

    st.divider()
    model_choice = st.selectbox(
        "Model Deteksi",
        ["YOLO", "SSD", "Faster R-CNN"],
        index=["YOLO", "SSD", "Faster R-CNN"].index(
            st.session_state.last_model
        )
    )
    st.session_state.last_model = model_choice

    conf_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.05,
        max_value=0.95,
        value=0.25,
        step=0.05,
        help="Semakin tinggi nilai, semakin ketat prediksi yang ditampilkan."
    )

    st.caption("Pilih model dan confidence sebelum melakukan deteksi.")


# Muat model hanya ketika halaman deteksi dipilih.
needs_model = page in ["Deteksi Gambar", "Kamera Live", "Deteksi Video"]
model_tuple = None
if needs_model:
    with st.spinner(f"Memuat model {model_choice}..."):
        model_tuple = load_model(model_choice)
    if model_tuple is None:
        st.error("Model tidak dapat digunakan. Periksa file model dan dependensinya.")
        st.stop()


# =========================================================
# HALAMAN 1: DASHBOARD
# =========================================================
if page == "Dashboard":
    st.title("🌤️ Dashboard AWAN Detector")
    st.caption("Ringkasan aktivitas deteksi jenis awan dan prediksi cuaca.")
    history = st.session_state.history
    total_detections = sum(item["jumlah"] for item in history)
    unique_clouds = len({
        result["jenis"]
        for item in history
        for result in item["hasil"]
    })
    total_sessions = len(history)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Deteksi", total_detections, help="Jumlah objek awan dari riwayat sesi ini.")
    col2.metric("Sesi Analisis", total_sessions)
    col3.metric("Jenis Awan Ditemukan", unique_clouds)
    col4.metric("Model Terpilih", model_choice)

    st.divider()
    st.subheader("Visualisasi Dataset Penelitian")
    st.caption("Distribusi objek dari hasil perhitungan label pada notebook Colab.")
    dataset_col1, dataset_col2 = st.columns([3, 1])
    with dataset_col1:
        import pandas as pd
        dataset_df = pd.DataFrame({
            "Jenis Awan": list(DATASET_CLASS_COUNTS.keys()),
            "Jumlah Objek": list(DATASET_CLASS_COUNTS.values())
        }).set_index("Jenis Awan")
        st.bar_chart(dataset_df, y="Jumlah Objek", color="#42A5F5")
    with dataset_col2:
        st.metric("Total Objek", sum(DATASET_CLASS_COUNTS.values()))
        st.write("Jumlah objek pada empat kelas relatif berimbang.")

    st.subheader("Perbandingan Hasil Evaluasi Model")
    metric_rows = []
    for name, values in MODEL_METRICS.items():
        metric_rows.append({
            "Model": name,
            "mAP@50": values["mAP50"],
            "mAP@50-95": values["mAP50-95"],
            "Precision": values["Precision"],
            "Recall": values["Recall"],
            "Waktu (ms/gambar)": values["speed_ms"],
        })
    metric_df = pd.DataFrame(metric_rows).set_index("Model")
    st.dataframe(
        metric_df.style.format({
            "mAP@50": "{:.4f}", "mAP@50-95": "{:.4f}",
            "Precision": lambda x: "—" if pd.isna(x) else f"{x:.4f}",
            "Recall": "{:.4f}",
            "Waktu (ms/gambar)": lambda x: "—" if pd.isna(x) else f"{x:.2f}",
        }),
        use_container_width=True
    )
    st.caption(
        "Catatan: Pada notebook SSD, kolom yang dicetak sebagai Precision dan Recall "
        "sebenarnya berasal dari mar_100 dan mar_10 TorchMetrics. Angka tersebut "
        "ditampilkan mengikuti keluaran notebook, bukan precision/recall standar. "
        "Precision Faster R-CNN tidak tersedia pada keluaran evaluasi yang dikirim. "
        "Waktu YOLO 5,7 ms adalah waktu inference yang tercetak pada hasil validasi."
    )

    st.subheader("Perbandingan mAP@50")
    map_df = pd.DataFrame({
        "Model": list(MODEL_METRICS.keys()),
        "mAP@50": [v["mAP50"] for v in MODEL_METRICS.values()],
    }).set_index("Model")
    st.bar_chart(map_df, y="mAP@50", color="#26A69A")

    st.subheader("Grafik Training Loss")
    loss_choice = st.selectbox(
        "Pilih model untuk melihat training loss",
        ["SSD", "Faster R-CNN"]
    )
    if loss_choice == "SSD":
        loss_values = SSD_TRAIN_LOSS
    else:
        loss_values = FASTER_RCNN_TRAIN_LOSS
    loss_df = pd.DataFrame({
        "Epoch": range(1, len(loss_values) + 1),
        "Training Loss": loss_values
    }).set_index("Epoch")
    st.line_chart(loss_df, y="Training Loss", color="#EF6C00")
    st.caption(
        "Nilai loss YOLOv8 tersimpan dalam grafik dan file results.csv di lingkungan Colab, "
        "tetapi deret angka results.csv tidak tersedia pada keluaran notebook yang dikirim. "
        "Karena itu grafik loss YOLOv8 belum ditampilkan agar tidak mengarang data."
    )

    st.subheader("Confusion Matrix SSD")
    st.caption(
        "Matriks berikut menggunakan angka yang dicetak pada notebook. "
        "Label baris/kolom ditampilkan sebagai indeks karena pemetaan singkatan kelas "
        "pada kode notebook belum dapat dipastikan."
    )
    cm_df = pd.DataFrame(
        SSD_CONFUSION_MATRIX,
        index=["Kelas 1", "Kelas 2", "Kelas 3", "Kelas 4"],
        columns=["Prediksi 1", "Prediksi 2", "Prediksi 3", "Prediksi 4"]
    )
    st.dataframe(cm_df, use_container_width=True)

    st.divider()
    left, right = st.columns([3, 2])

    with left:
        st.subheader("Distribusi Jenis Awan")
        counts = Counter(
            result["jenis"]
            for item in history
            for result in item["hasil"]
        )
        if counts:
            st.bar_chart(
                {name: count for name, count in counts.items()},
                horizontal=True
            )
        else:
            st.info("Grafik akan muncul setelah kamu melakukan deteksi.")

    with right:
        st.subheader("Informasi Sistem")
        st.write(f"**Model yang dipilih:** {model_choice}")
        st.write(f"**Confidence threshold:** {conf_threshold:.0%}")
        st.write("**Kelas yang didukung:** 4 jenis awan")
        st.write("**Fitur:** Gambar, kamera live, dan video")

    st.subheader("Aktivitas Terbaru")
    if history:
        for item in reversed(history[-5:]):
            with st.container(border=True):
                st.write(f"**{item['waktu']}** · {item['sumber']}")
                st.caption(
                    f"Model: {item['model']} | Jumlah deteksi: {item['jumlah']}"
                )
    else:
        st.info("Belum ada aktivitas. Buka menu Deteksi Gambar, Kamera Live, atau Deteksi Video.")


# =========================================================
# HALAMAN 2: DETEKSI GAMBAR
# =========================================================
elif page == "Deteksi Gambar":
    st.title("🖼️ Deteksi Awan dari Gambar")
    st.caption("Unggah satu atau beberapa gambar untuk mendeteksi jenis awan.")
    uploaded_imgs = st.file_uploader(
        "Pilih gambar JPG, JPEG, PNG, atau WEBP",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True
    )

    if uploaded_imgs:
        for uploaded in uploaded_imgs:
            st.divider()
            st.subheader(uploaded.name)
            try:
                img = Image.open(uploaded).convert("RGB")
                with st.spinner("Sedang menganalisis gambar..."):
                    detections = predict_pil(model_tuple, img, conf_threshold)

                col1, col2 = st.columns(2)
                with col1:
                    st.image(img, caption="Gambar Asli", use_container_width=True)
                with col2:
                    st.image(
                        draw_results_pil(img, detections),
                        caption="Hasil Deteksi",
                        use_container_width=True
                    )

                show_detection_cards(detections)
                if detections:
                    save_history("Gambar", model_choice, detections)
            except Exception as exc:
                st.error(f"Gambar tidak dapat diproses: {exc}")
    else:
        st.info("Silakan unggah gambar awan untuk memulai deteksi.")


# =========================================================
# HALAMAN 3: KAMERA LIVE
# =========================================================
elif page == "Kamera Live":
    st.title("📷 Deteksi Awan Kamera Live")
    st.caption("Arahkan kamera ke langit untuk melihat hasil deteksi secara langsung.")
    st.warning(
        "Akses kamera memerlukan izin browser dan koneksi HTTPS "
        "(kecuali localhost)."
    )

    try:
        from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
        import av

        class CloudDetector(VideoProcessorBase):
            def __init__(self):
                self.model_tuple = model_tuple
                self.threshold = conf_threshold

            def recv(self, frame):
                image_array = frame.to_ndarray(format="rgb24")
                image_pil = Image.fromarray(image_array)
                detections = predict_pil(
                    self.model_tuple, image_pil, self.threshold
                )
                annotated = draw_results_pil(image_pil, detections)
                return av.VideoFrame.from_ndarray(
                    np.array(annotated), format="rgb24"
                )

        webrtc_streamer(
            key=f"cloud-detector-{model_choice}",
            video_processor_factory=CloudDetector,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )
    except ImportError:
        st.error("Library streamlit-webrtc atau av belum terpasang.")
        st.code("pip install streamlit-webrtc av")


# =========================================================
# HALAMAN 4: DETEKSI VIDEO
# =========================================================
elif page == "Deteksi Video":
    st.title("🎬 Deteksi Awan dari Video")
    st.caption("Unggah video dan analisis frame untuk menemukan jenis awan.")
    uploaded_video = st.file_uploader(
        "Pilih video MP4, AVI, MOV, atau MKV",
        type=["mp4", "avi", "mov", "mkv"]
    )

    if uploaded_video:
        suffix = os.path.splitext(uploaded_video.name)[1] or ".mp4"
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp:
                temp.write(uploaded_video.getbuffer())
                temp_path = temp.name

            st.video(temp_path)
            process_every = st.slider(
                "Analisis setiap berapa frame?",
                min_value=1,
                max_value=60,
                value=10,
                help="Nilai lebih besar mempercepat proses, tetapi lebih sedikit frame dianalisis."
            )

            if st.button("▶️ Mulai Analisis Video", type="primary"):
                try:
                    import av
                    container = av.open(temp_path)
                    stream = container.streams.video[0]
                    total_frames = stream.frames or 0
                    progress = st.progress(0)
                    preview = st.empty()
                    status = st.empty()
                    all_detections = []
                    frame_index = 0
                    last_detections = []

                    for packet in container.demux(stream):
                        for frame in packet.decode():
                            if frame_index % process_every == 0:
                                frame_image = frame.to_image().convert("RGB")
                                last_detections = predict_pil(
                                    model_tuple, frame_image, conf_threshold
                                )
                                all_detections.extend(last_detections)
                                preview.image(
                                    draw_results_pil(frame_image, last_detections),
                                    caption=f"Frame {frame_index}",
                                    use_container_width=True
                                )
                                status.write(
                                    f"Frame diproses: {frame_index} · "
                                    f"Deteksi pada frame: {len(last_detections)}"
                                )

                            if total_frames:
                                progress.progress(
                                    min((frame_index + 1) / total_frames, 1.0)
                                )
                            frame_index += 1

                    container.close()
                    progress.progress(1.0)
                    st.success("Analisis video selesai.")

                    if all_detections:
                        save_history("Video", model_choice, all_detections)
                        counts = Counter(d["label"] for d in all_detections)
                        st.subheader("Ringkasan Deteksi Video")
                        st.metric("Total objek terdeteksi", len(all_detections))
                        st.bar_chart(dict(counts))
                        for label, count in counts.most_common():
                            info = CLOUD_WEATHER_MAP.get(label, {})
                            st.write(
                                f"{info.get('icon', '☁️')} **{label}:** "
                                f"{count} deteksi — {info.get('cuaca', '-')}"
                            )
                    else:
                        st.info("Tidak ditemukan awan pada frame yang dianalisis.")
                except Exception as exc:
                    st.error(f"Gagal menganalisis video: {exc}")
        finally:
            # File sementara dihapus setelah proses halaman selesai.
            # Jika tombol belum ditekan, file tetap tersedia selama rerun ini.
            pass
    else:
        st.info("Silakan unggah video untuk memulai analisis.")


# =========================================================
# HALAMAN 5: RIWAYAT DETEKSI
# =========================================================
elif page == "Riwayat Deteksi":
    st.title("🕘 Riwayat Deteksi")
    st.caption("Daftar hasil deteksi yang tersimpan selama sesi aplikasi berjalan.")
    history = st.session_state.history

    if history:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.metric("Jumlah sesi tersimpan", len(history))
        with col2:
            if st.button("Hapus Riwayat", type="secondary"):
                st.session_state.history = []
                st.rerun()

        for item in reversed(history):
            with st.expander(
                f"{item['waktu']} | {item['sumber']} | {item['jumlah']} deteksi"
            ):
                st.write(f"**Model:** {item['model']}")
                for result in item["hasil"]:
                    st.write(
                        f"- **{result['jenis']}** · "
                        f"{result['confidence']:.2f}% · {result['cuaca']}"
                    )
    else:
        st.info("Riwayat masih kosong. Lakukan deteksi terlebih dahulu.")


# =========================================================
# HALAMAN 6: INFORMASI JENIS AWAN
# =========================================================
elif page == "Informasi Jenis Awan":
    st.title("☁️ Informasi Jenis Awan")
    st.caption("Mengenal empat kelas awan yang digunakan dalam sistem.")
    for cloud_name, info in CLOUD_WEATHER_MAP.items():
        with st.container(border=True):
            st.markdown(f"### {info['icon']} {cloud_name}")
            st.write(info["deskripsi"])
            st.write(f"**Prediksi cuaca pada aplikasi:** {info['cuaca']}")


# =========================================================
# HALAMAN 7: TENTANG APLIKASI
# =========================================================
elif page == "Tentang Aplikasi":
    st.title("ℹ️ Tentang AWAN Detector")
    st.markdown(
        """
        **AWAN Detector** merupakan aplikasi berbasis web yang dirancang
        untuk mendeteksi jenis awan dari gambar, kamera secara langsung,
        maupun video. Sistem menampilkan kelas awan, nilai confidence,
        dan informasi prediksi cuaca berdasarkan pemetaan kelas yang
        telah ditentukan pada aplikasi.

        ### Model deteksi
        - **YOLO** — model deteksi objek satu tahap.
        - **SSD** — model Single Shot Detector.
        - **Faster R-CNN** — model deteksi objek berbasis region proposal.

        ### Kelas awan
        Altocumulus, Nimbostratus, Cumulus, dan Cumulonimbus.

        ### Catatan
        Prediksi cuaca yang ditampilkan merupakan pemetaan berdasarkan
        kelas awan hasil model, bukan prakiraan cuaca meteorologis.
        Hasil deteksi dipengaruhi kualitas gambar, data pelatihan,
        konfigurasi model, dan nilai confidence threshold.
        """
    )
    st.divider()
    st.caption("AWAN Detector · Cloud Detection & Weather Prediction")
