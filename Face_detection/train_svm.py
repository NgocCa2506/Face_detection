import pickle
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import numpy as np

# Load embeddings
with open("embeddings.pkl", "rb") as f:
    data = pickle.load(f)

X = [item[0] for item in data]
y = [item[1] for item in data]

le = LabelEncoder()
y_encoded = le.fit_transform(y)

model = SVC(probability=True)
model.fit(X, y_encoded)

# Dự đoán trên chính tập train để in accuracy
y_pred = model.predict(X)
acc = accuracy_score(y_encoded, y_pred)
print(f"✅ Mô hình SVM đã được huấn luyện và lưu.")
print(f"📊 Độ chính xác trên tập huấn luyện: {acc*100:.2f}%")

with open("svm_model.pkl", "wb") as f:
    pickle.dump((model, le), f)
