import cv2
import os
import mediapipe as mp

# Nhập tên người
label = input("Input name: ").strip()
save_dir = f"dataset/{label}"
os.makedirs(save_dir, exist_ok=True)

# Khởi tạo MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.7)

cap = cv2.VideoCapture(0)
count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Phát hiện khuôn mặt
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb_frame)

    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            x = int(bboxC.xmin * iw)
            y = int(bboxC.ymin * ih)
            w = int(bboxC.width * iw)
            h = int(bboxC.height * ih)

            x, y = max(x, 0), max(y, 0)
            face_img = frame[y:y + h, x:x + w]

            # Hiển thị vùng mặt và toàn ảnh
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.imshow("Detected Face", face_img)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('s'):
                save_path = f"{save_dir}/{label}_{count}.jpg"
                cv2.imwrite(save_path, face_img)
                print(f"✅ Đã lưu: {save_path}")
                count += 1
            elif key == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                exit()

    cv2.imshow("Webcam", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
