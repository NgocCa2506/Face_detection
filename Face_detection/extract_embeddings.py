import os
import cv2
import mediapipe as mp
from deepface import DeepFace
import numpy as np
import pickle

dataset_dir = "dataset"
embedding_data = []

mp_face = mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.7)

def extract_face(img):
    results = mp_face.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if results.detections:
        for det in results.detections:
            bbox = det.location_data.relative_bounding_box
            h, w, _ = img.shape
            x, y, w_box, h_box = int(bbox.xmin*w), int(bbox.ymin*h), int(bbox.width*w), int(bbox.height*h)
            face = img[y:y+h_box, x:x+w_box]
            return face
    return None

for label in os.listdir(dataset_dir):
    label_dir = os.path.join(dataset_dir, label)
    for file in os.listdir(label_dir):
        path = os.path.join(label_dir, file)
        img = cv2.imread(path)
        face = extract_face(img)
        if face is not None:
            try:
                embedding = DeepFace.represent(face, model_name="Facenet")[0]["embedding"]
                embedding_data.append((embedding, label))
            except Exception as e:
                print(f"❌ Lỗi với ảnh {file}: {e}")

# Lưu dữ liệu embedding
with open("embeddings.pkl", "wb") as f:
    pickle.dump(embedding_data, f)

print("✅ Đã trích xuất embedding.")
