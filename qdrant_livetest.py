from insightface.app import FaceAnalysis
import cv2
from qdrant_client import QdrantClient
import numpy as np
from datetime import datetime
import time

# ------------------ QDRANT SETUP ------------------
client = QdrantClient(url="http://localhost:6333")

faces_collection = "Faces"
logs_collection = "logs"

# ------------------ INSIGHTFACE SETUP ------------------
MODEL_ROOT = "C:/Users/User/Desktop/InsightFace"
app = FaceAnalysis(
    name="buffalo_s",
    root=MODEL_ROOT,
    providers=['CPUExecutionProvider']
)
app.prepare(ctx_id=0, det_size=(640, 640))

# ------------------ PARAMETERS ------------------
THRESHOLD = 0.7
NUM_FRAMES = 5
DETECTION_TIME = 5  # seconds total
SAMPLE_INTERVAL = DETECTION_TIME / NUM_FRAMES

cap = cv2.VideoCapture(0)
FONT = cv2.FONT_HERSHEY_SIMPLEX

print("Starting live face recognition. Press 'q' to quit.")

# ------------------ STATE ------------------
face_samples = []
last_sample_time = 0
collecting = False
granted_user_id = None
granted_user_name = None
access_granted = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    faces = app.get(frame)

    # ---------------- MULTIPLE FACES CASE ----------------
    if len(faces) > 1:
        # If access was already granted -> intruder detected
        if granted_user_id is not None:
            cv2.putText(frame, "ACCESS DENIED - Intruder detected", (30, 30), FONT, 0.7, (0, 0, 255), 2)
            access_granted = False
            granted_user_id = None
            granted_user_name = None
        else:
            for face in faces:
                box = face.bbox.astype(int)
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
            cv2.putText(frame, "UNAUTHORIZED multiple faces", (30, 30), FONT, 0.7, (0, 0, 255), 2)
        collecting = False
        face_samples = []

    # ---------------- SINGLE FACE CASE ----------------
    elif len(faces) == 1:
        face = faces[0]
        box = face.bbox.astype(int)
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)

        if granted_user_id is not None:
            # If already granted, just keep showing until face leaves
            cv2.putText(frame, f"ACCESS GRANTED: {granted_user_name}", (30, 30),
                        FONT, 0.8, (0, 255, 0), 2)
        else:
            # Start collecting if not already
            if not collecting:
                collecting = True
                face_samples = []
                last_sample_time = time.time()

            # Collect embeddings every SAMPLE_INTERVAL
            if time.time() - last_sample_time >= SAMPLE_INTERVAL and len(face_samples) < NUM_FRAMES:
                face_samples.append(face.embedding)
                last_sample_time = time.time()

            cv2.putText(frame, f"Collecting samples: {len(face_samples)}/{NUM_FRAMES}",
                        (30, 30), FONT, 0.7, (255, 255, 0), 2)

            # Once NUM_FRAMES collected, do recognition
            if len(face_samples) == NUM_FRAMES:
                avg_embedding = np.mean(np.array(face_samples), axis=0).tolist()
                user_id = "unknown"
                user_name = "unknown"
                success = False

                try:
                    results = client.search(
                        collection_name=faces_collection,
                        query_vector=avg_embedding,
                        limit=1
                    )
                    if results:
                        match = results[0]
                        similarity = match.score
                        if similarity >= THRESHOLD:
                            success = True
                            payload = match.payload
                            user_id = payload.get("ID", "unknown")
                            user_name = payload.get("Name", "unknown")
                except Exception as e:
                    print("Qdrant search error:", e)

                if success:
                    access_granted = True
                    granted_user_id = user_id
                    granted_user_name = user_name
                    cv2.putText(frame, f"ACCESS GRANTED", (box[0], box[1] - 10),
                                FONT, 0.9, (0, 255, 0), 2)

                    # log once
                    log_point = {
                        "id": int(datetime.now().timestamp()),
                        "vector": [0.0] * 128,
                        "payload": {
                            "user_id": user_id,
                            "user_name": user_name,
                            "timestamp": datetime.now().isoformat(),
                            "action": "Entered-through-main-gate"
                        }
                    }
                    try:
                        client.upsert(collection_name=logs_collection, points=[log_point])
                    except Exception as e:
                        print("Failed to log entry:", e)
                else:
                    access_granted = False
                    cv2.putText(frame, "ACCESS DENIED", (box[0], box[1] - 10),
                                FONT, 0.9, (0, 0, 255), 2)

                # Reset collection for next cycle
                collecting = False
                face_samples = []

    # ---------------- NO FACE CASE ----------------
    else:
        if granted_user_id is not None:
            # User left → reset
            granted_user_id = None
            granted_user_name = None
            access_granted = False
        cv2.putText(frame, "No face detected", (30, 30), FONT, 0.6, (200, 200, 200), 1)
        collecting = False
        face_samples = []

    cv2.imshow("Face Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
