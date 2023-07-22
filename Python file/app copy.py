import cv2
import face_recognition
import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np

# Fungsi untuk melakukan prediksi pada gambar
def predict(image):
    xb = image.unsqueeze(0)
    xb = xb.to(device)
    preds = model(xb)
    return preds[0]

# Inisialisasi kamera
video_capture = cv2.VideoCapture(0)

# Load model yang sudah dilatih sebelumnya
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = 'model.pth'  # Load model yang sudah dilatih sebelumnya (ganti dengan model yang sesuai)
to_device = T.ToTensor()

# Loop untuk mengambil gambar dari kamera
while True:
    # Ambil frame dari kamera
    ret, frame = video_capture.read()

    # Ubah frame menjadi format yang dapat diproses oleh model
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_pil = Image.fromarray(frame_rgb)
    image_tensor = to_device(frame_pil, device)

    # Lakukan prediksi pada gambar
    prediction = predict(image_tensor)

    # Konversi hasil prediksi menjadi array numpy
    prediction_np = prediction.cpu().detach().numpy()
    
    # Tampilkan hasil prediksi pada layar
    result_str = "Predictions:\n"
    classes = ["alvian", "chen", "nicho", "rendie", "helm_merah", "helm_kuning", "helm_orange", "helm_putih", "vest", "gloves"]
    for i, class_name in enumerate(classes):
        percent = round(prediction_np[i] * 100, 2)
        result_str += f"{class_name}: {percent}%\n"

    cv2.putText(frame, result_str, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow('Object and Face Detector', frame)

    # Tekan tombol 'q' untuk keluar dari loop dan menghentikan kamera
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Stop kamera dan tutup jendela kamera
video_capture.release()
cv2.destroyAllWindows()
