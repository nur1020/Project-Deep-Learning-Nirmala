**Perbandingan Model YOLOv8, SSD MobileNetV3, dan Faster R-CNN dalam Identifikasi Jenis Awan untuk Analisis Cuaca**

Aplikasi dan penelitian ini digunakan untuk mendeteksi jenis awan (Altocumulus, Cumulonimbus, Cumulus, Nimbostratus) dari citra langit atau video, sekaligus memprediksi kondisi cuaca secara cerdas dengan menganalisis kombinasi jenis awan dan tingkat kecerahan (brightness) gambar.

**Link Aplikasi / Demo**

Anda dapat menjalankan notebook inferensi langsung melalui:
🔗 [https://github.com/nur1020/Project-Deep-Learning-Nirmala]

**Tim Pengembang / Peneliti**

1. Nirmala
   
**Deskripsi Proyek**

Identifikasi jenis awan merupakan elemen krusial dalam meteorologi untuk memprediksi cuaca jangka pendek. Proyek ini bertujuan untuk:
1. Mengidentifikasi 4 jenis awan utama penyusun cuaca: Altocumulus, Cumulonimbus, Cumulus, dan Nimbostratus.
2. Membandingkan performa tiga arsitektur Object Detection mutakhir: YOLOv8, SSD MobileNetV3 Large, dan Faster R-CNN ResNet50 FPN.
3. Membangun logika cerdas berbasis Computer Vision untuk mengklasifikasikan cuaca akhir (Cerah, Mendung, Hujan, Hujan Lebat/Badai) menggunakan threshold kecerahan citra.
   
**Teknologi yang Digunakan**

Machine Learning & Computer Vision

- Python 3.8+

- PyTorch & Torchvision — Framework utama untuk SSD MobileNetV3 dan Faster R-CNN.

- Ultralytics — Framework utama untuk pelatihan dan inferensi YOLOv8.

- OpenCV (cv2) — Pemrosesan citra, manipulasi video, deteksi blur (Laplacian), dan tingkat kecerahan.

- TorchMetrics — Evaluasi model (Mean Average Precision / mAP).

Data Processing & Visualisasi

- Roboflow — Manajemen dataset dan API penarikan data.

- Matplotlib & Seaborn — Visualisasi distribusi data, metrik loss, dan Confusion Matrix.

- Pandas & NumPy — Manipulasi matriks dan analisis log hasil (CSV).

**Struktur Proyek**

awan-analisis-cuaca/
│
├── dataset_baru/                # Dataset bersih setelah filtering (Train, Valid, Test)
├── runs/detect/YOLO_AWAN/       # Log hasil training, weights, dan grafik YOLOv8
├── hasil_evaluasi/              # Confusion matrix dan grafik loss dari ketiga model
├── SSD_Hasil_Bebas/             # Output inferensi gambar & video dari SSD
├── YOLOv8_Hasil_Bebas/          # Output inferensi gambar & video dari YOLOv8
├── Model_YOLO_AWAN_best.pt      # Model YOLOv8 terlatih
├── Model_SSD_AWAN_best.pth      # Model SSD terlatih
├── Model_FasterRCNN_AWAN_best.pth # Model Faster R-CNN terlatih
└── identifikasi_awan_cuaca.ipynb # Notebook utama (Data prep, Training, Inference)

**Metodologi**

1. Data Wrangling & Assessing
Data Gathering: Mengunduh dataset deteksi awan dari workspace Roboflow menggunakan API Key. Dataset berisi anomali cuaca dengan format anotasi YOLO (.txt).

Data Assessing: Mengekstrak informasi dari data.yaml dan menghitung distribusi bounding box untuk 4 kelas awan guna mengecek keseimbangan dataset.

2. Data Cleaning (Kualitas Citra)
Menggunakan OpenCV untuk menyeleksi gambar secara otomatis sebelum pelatihan:

Deteksi Blur: Menghitung varians dari Laplacian filter (cv2.Laplacian(gray).var()). Gambar dengan skor di bawah ambang batas (10-20) dihapus karena terlalu buram.

Deteksi Gelap: Menghitung rata-rata piksel grayscale. Gambar di bawah ambang batas (< 40) dihapus karena tidak informatif.

3. Data Preprocessing & Split
Melakukan redistribusi (resplit) dataset menjadi rasio baru: 75% Train, 12.5% Validation, 12.5% Test untuk memastikan evaluasi model lebih objektif.

YOLOv8: Gambar di-resize ke dimensi 640x640.

SSD MobileNetV3: Gambar di-resize ke 320x320, konversi anotasi dari format YOLO (titik tengah) ke piksel absolut [xmin, ymin, xmax, ymax], dan penambahan augmentasi Horizontal Flip serta Brightness Adjustment.

Faster R-CNN: Gambar di-resize ke resolusi tinggi 800x800 dengan standarisasi tensor.

4. Modelling
Melatih tiga model dengan pendekatan berbeda:

YOLOv8n: Dilatih selama 50 epochs dengan batch size 16. Sangat optimal untuk deteksi real-time.

SSD MobileNetV3 Large: Dilatih selama 30 epochs menggunakan optimizer SGD (lr=0.002). Arsitektur ringan yang cocok untuk mobile/edge device.

Faster R-CNN (ResNet50 FPN): Dilatih selama 20 epochs menggunakan SGD. Model Two-Stage dengan fokus pada tingkat presisi tinggi.

5. Inferensi Cerdas (Aturan Prediksi Cuaca)
Model tidak hanya mengeluarkan bounding box jenis awan, tetapi memprosesnya lebih lanjut menjadi klasifikasi cuaca final menggunakan aturan gabungan awan + kecerahan:

Cumulonimbus + Brightness < 100 ➡️ Hujan Lebat (Jika kecerahan > 100 ➡️ Potensi Hujan/Awan Badai).

Nimbostratus ➡️ Hujan terus-menerus.

Altocumulus ➡️ Mendung.

Cumulus + Brightness > 130 ➡️ Cerah (Jika kecerahan < 130 ➡️ Berawan).

**Evaluasi Model**

Ketiga model dievaluasi secara komprehensif menggunakan set data pengujian (Test Dataset). Metrik yang digunakan meliputi:

Mean Average Precision (mAP@50 dan mAP@50-95) menggunakan modul torchmetrics.

Confusion Matrix absolut dan ternormalisasi.

Precision & Recall batas atas (mar_100).

(Catatan: Anda dapat menambahkan tabel hasil akhir perbandingan mAP dan kecepatan waktu inferensi dari ketiga model di bagian ini setelah mendapatkan angka pastinya dari Colab).

**Penyimpanan Model**

Model yang telah dilatih (best weights) disimpan ke dalam Google Drive agar dapat dipanggil kembali tanpa perlu melatih ulang:

- Model_YOLO_AWAN_best.pt

- Model_SSD_AWAN_best.pth

- Model_FasterRCNN_AWAN_best.pth

**Dependencies**

torch>=2.0.0
torchvision>=0.15.0
ultralytics>=8.0.0
roboflow>=1.1.0
torchmetrics>=1.2.0
opencv-python>=4.8.0
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
PyYAML>=6.0

**Lisensi**

Proyek ini dibuat untuk keperluan akademis dan penelitian.
