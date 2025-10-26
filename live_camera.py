from insightface.app import FaceAnalysis
import cv2
import numpy as np
import json

# -------------------------------
# Paths
MODEL_ROOT = "C:/Users/User/Desktop/InsightFace"
DATABASE_PATH = "C:/Users/User/Desktop/InsightFace/faces.json"  # your JSON file
# -------------------------------

# Initialize InsightFace
app = FaceAnalysis(name="buffalo_s", root=MODEL_ROOT, providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

# Load your face embeddings database
with open(DATABASE_PATH, "r") as f:
    database = json.load(f)  # expects format {"name": [embedding_list]}

# Convert embeddings to numpy arrays
for key in database:
    database[key] = np.array(database[key], dtype=np.float32)

# Function to compute cosine similarity
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Threshold for matching
SIMILARITY_THRESHOLD = 0.6  # you can adjust this

# Open your PC camera
cap = cv2.VideoCapture(0)  # 0 = default camera

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    faces = app.get(frame)

    for face in faces:
        box = face.bbox.astype(int)
        embedding = face.embedding

        # Compare with database
        match_name = "Unknown"
        max_sim = 0
        for name, db_embedding in database.items():
            sim = cosine_similarity(embedding, db_embedding)
            if sim > max_sim and sim > SIMILARITY_THRESHOLD:
                max_sim = sim
                match_name = name

        # Draw bounding box + name
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        cv2.putText(frame, f"{match_name}", (box[0], box[1]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow("Face Recognition", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
