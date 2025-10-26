from insightface.app import FaceAnalysis
import cv2
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models

# ------------------ QDRANT SETUP ------------------
client = QdrantClient(url="http://localhost:6333")
collection_name = "Faces"

# ------------------ INSIGHTFACE SETUP ------------------
MODEL_ROOT = "C:/Users/User/Desktop/InsightFace"
app = FaceAnalysis(
    name="buffalo_s",
    root=MODEL_ROOT,
    providers=['CPUExecutionProvider']
)
app.prepare(ctx_id=0, det_size=(640, 640))

# ------------------ CAPTURE MULTIPLE PHOTOS ------------------
cap = cv2.VideoCapture(0)  # default camera
print("Press 'Space' to take a photo. Capture 5 photos for better accuracy.")

embeddings = []
num_photos = 5
captured = 0

while captured < num_photos:
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imshow("Camera", frame)
    key = cv2.waitKey(1)
    if key & 0xFF == ord(' '):  # Press space to capture
        img = frame.copy()
        faces = app.get(img)
        if faces:
            embeddings.append(faces[0].embedding)
            captured += 1
            print(f"Captured photo {captured}/{num_photos}")
        else:
            print("No face detected, try again.")
    elif key & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        exit()

cap.release()
cv2.destroyAllWindows()

if not embeddings:
    print("No faces captured. Exiting.")
    exit()

# ------------------ COMPUTE AVERAGE EMBEDDING ------------------
avg_embedding = np.mean(embeddings, axis=0).tolist()
print(f"Average embedding calculated (length {len(avg_embedding)})")

# ------------------ UPSERT INTO QDRANT ------------------
user_id = input("Enter your ID: ")
user_name = input("Enter your name: ")

# Auto-increment point ID
point_id = client.count(collection_name).count + 1

client.upsert(
    collection_name=collection_name,
    points=[
        models.PointStruct(
            id=point_id,
            vector=avg_embedding,
            payload={"ID": user_id, "Name": user_name}
        )
    ],
    wait=True
)
print(f"\n{user_name} ({user_id}) inserted successfully into collection '{collection_name}'.")

# ------------------ VERIFY ------------------
retrieved_point = client.retrieve(collection_name=collection_name, ids=[point_id])
print("\nVerified inserted point:")
print(retrieved_point)
