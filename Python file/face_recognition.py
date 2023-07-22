import cv2
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import face_recognition
import torch
import torchvision.transforms as T

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Load model yang sudah dilatih sebelumnya dari file "model.pth"
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = torch.load("model.pth", map_location=device)
model.eval()

# Definisikan transformasi data yang sesuai untuk gambar wajah
to_device = T.ToTensor()

# Fungsi untuk melakukan prediksi pada gambar wajah
def predict_single(image):
    xb = image.unsqueeze(0)
    xb = xb.to(device)
    preds = model(xb)
    return preds[0]

# Fungsi untuk mendeteksi dan mengenali wajah
def recognize_face(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        face_roi = gray_frame[y:y + h, x:x + w]

        # Lakukan proses pengenalan wajah di sini
        # Convert gambar wajah menjadi PyTorch Tensor
        image_tensor = to_device(Image.fromarray(face_roi), device)

        # Lakukan prediksi pada gambar wajah menggunakan model yang sudah dilatih
        prediction = predict_single(image_tensor)

        # Ambil indeks kelas dengan probabilitas tertinggi
        pred_class_index = torch.argmax(prediction).item()

        # Daftar nama kelas (misalnya, alvian, chen, nicho, rendie)
        classes = ["alvian", "chen", "nicho", "rendie"]

        # Ambil nama kelas yang sesuai dengan indeks
        label = classes[pred_class_index]

        # Tampilkan label di atas wajah yang dikenali
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

# Fungsi untuk menangkap gambar dari kamera dan mengupdate canvas
def capture_frame():
    _, frame = camera.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = recognize_face(frame)
    photo = ImageTk.PhotoImage(image=Image.fromarray(frame))
    canvas.create_image(0, 0, image=photo, anchor=tk.NW)
    canvas.image = photo
    root.after(10, capture_frame)

# Fungsi untuk mengaktifkan deteksi wajah
def detect_faces():
    messagebox.showinfo("Info", "Pengenalan Wajah Dimulai!")

    # Inisialisasi kamera
    video_capture = cv2.VideoCapture(0)

    while True:
        # Ambil frame dari kamera
        ret, frame = video_capture.read()

        # Lakukan face detection menggunakan face_recognition
        face_locations = face_recognition.face_locations(frame)

        # Tandai wajah dengan persegi panjang
        for top, right, bottom, left in face_locations:
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        # Tampilkan frame dengan wajah yang telah ditandai
        cv2.imshow('Face Detection', frame)

        # Tekan tombol 'q' untuk keluar dari loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Inisialisasi GUI tkinter
root = tk.Tk()
root.title("Face Recognition App")
root.geometry("1170x2532")  # Ukuran iPhone 12 Pro

# Buat canvas untuk menampilkan gambar dari kamera
canvas = tk.Canvas(root, width=1170, height=2532)
canvas.pack()

# Tombol untuk memulai deteksi wajah
detect_button = tk.Button(root, text="Detect", command=detect_faces)
detect_button.pack()

# Inisialisasi kamera
camera = cv2.VideoCapture(0)

# Mulai menangkap gambar dan mendeteksi wajah
capture_frame()

root.mainloop()