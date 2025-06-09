import cv2
import mediapipe as mp
from deepface import DeepFace
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import threading
import time
import requests

# Load SVM model và LabelEncoder
with open("svm_model.pkl", "rb") as f:
    model, le = pickle.load(f)

# Load embeddings
with open("embeddings.pkl", "rb") as f:
    data = pickle.load(f)
    known_embeddings = np.array([item[0] for item in data])
    known_labels = [item[1] for item in data]

# Mediapipe face detection
mp_face = mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.7)

cap = cv2.VideoCapture(0)

# Biến dùng cho threading và xử lý ổn định
lock = threading.Lock()
current_name = "Detecting..."
current_bbox = None
processing = False

# Gửi HTTP sang ESP32
esp32_url = "http://192.168.1.5/update"  # ⚠️ Thay IP này bằng IP của ESP32 bạn

# Kiểm soát ổn định
stable_name = None
last_name = None
stable_start_time = None
sent = False

def extract_face(img):
    results = mp_face.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if results.detections:
        det = results.detections[0]
        bbox = det.location_data.relative_bounding_box
        h, w, _ = img.shape
        x, y, w_box, h_box = int(bbox.xmin * w), int(bbox.ymin * h), int(bbox.width * w), int(bbox.height * h)
        face = img[y:y + h_box, x:x + w_box]
        return face, (x, y, w_box, h_box)
    return None, None

def recognition_thread(frame):
    global current_name, current_bbox, processing
    try:
        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        face, bbox = extract_face(small_frame)
        if face is not None:
            face = cv2.resize(face, (160, 160))
            embedding = DeepFace.represent(face, model_name="Facenet", enforce_detection=False)[0]["embedding"]
            sims = cosine_similarity([embedding], known_embeddings)[0]
            max_sim = np.max(sims)
            threshold = 0.7
            if max_sim < threshold:
                name = "Unknown"
            else:
                pred = model.predict([embedding])[0]
                name = le.inverse_transform([pred])[0]
            # Chuyển bbox lên ảnh gốc
            x, y, w_box, h_box = [int(v * 2) for v in bbox]
            with lock:
                current_name = name
                current_bbox = (x, y, w_box, h_box)
        else:
            with lock:
                current_name = "No face"
                current_bbox = None
    except Exception as e:
        print(f"❌ Lỗi: {e}")
        with lock:
            current_name = "Error"
            current_bbox = None
    finally:
        processing = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if not processing:
        processing = True
        threading.Thread(target=recognition_thread, args=(frame.copy(),)).start()

    # Hiển thị kết quả và kiểm tra ổn định
    with lock:
        name = current_name
        bbox = current_bbox

        # Kiểm tra nhận diện ổn định
        if name not in ["Detecting...", "No face", "Error"]:
            if name == last_name:
                if stable_start_time is None:
                    stable_start_time = time.time()
                elif time.time() - stable_start_time >= 2 and not sent:
                    try:
                        print(f"✅ Gửi tên '{name}' sang ESP32")
                        requests.post(esp32_url, data=name)
                        sent = True
                    except Exception as e:
                        print(f"❌ Lỗi gửi dữ liệu: {e}")
            else:
                last_name = name
                stable_start_time = time.time()
                sent = False
        else:
            last_name = None
            stable_start_time = None
            sent = False

        # Vẽ kết quả lên khung hình
        if bbox is not None:
            x, y, w_box, h_box = bbox
            color = (0, 0, 255) if name == "Unknown" else (255, 0, 0)
            cv2.rectangle(frame, (x, y), (x + w_box, y + h_box), (0, 255, 0), 2)
            cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        else:
            cv2.putText(frame, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.imshow("Face Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
