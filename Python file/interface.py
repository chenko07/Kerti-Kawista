import cv2
import face_recognition
from PIL import Image, ImageDraw

class FaceRecognitionCamera:
    def __init__(self):
        # Load model and face encodings
        self.model_path = "/Users/marchelandrianshevchenko/Documents/DETEKSI KEAMANAN KONSTRUKSI/model.pth"  # Ganti dengan path model yang sesuai
        self.known_face_encodings, self.known_face_names = self.load_known_faces()

    def load_known_faces(self):
        # Load data wajah yang sudah dikenal
        known_face_encodings = []
        known_face_names = ["alvian", "chen", "nicho", "rendie"]  # Ganti dengan nama kelas yang sesuai

        # Implementasi kode untuk memuat model wajah yang sudah dikenal
        # ...

        return known_face_encodings, known_face_names

    def recognize_faces(self, frame):
        # Proses pengenalan wajah pada frame kamera
        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
            name = "Unknown"

            if True in matches:
                first_match_index = matches.index(True)
                name = self.known_face_names[first_match_index]

            # Tandai wajah dengan nama
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (0, 255, 0), 1)

        return frame

    def run(self):
        video_capture = cv2.VideoCapture(0)

        while True:
            ret, frame = video_capture.read()
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Lakukan face recognition
            recognized_frame = self.recognize_faces(rgb_frame)

            # Tampilkan hasil kamera realtime
            cv2.imshow('Face Recognition', recognized_frame)

            # Tekan 'q' untuk keluar dari loop
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        video_capture.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    face_recognition_app = FaceRecognitionCamera()
    face_recognition_app.run()
