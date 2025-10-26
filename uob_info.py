import tensorflow_hub as hub
import tensorflow as tf
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, VectorParams, Distance
import re
import numpy as np # Import numpy for safe vector conversion

# --- 1. Data to be Uploaded (The new sentences) ---
uob_text = """
The University of Balamand (UOB) is a private, non-profit, independent Lebanese institution of Higher Education.

UOB was founded in 1988 by the Orthodox Patriarchate of Antioch and All the East upon the direction of Patriarch Ignatius IV.

The University's main campus is located in Balamand, El-Koura, Lebanon, on a scenic hill overlooking the Mediterranean Sea and the city of Tripoli.

UOB also has three other campuses across Lebanon: Dekwaneh (Beirut), Souk El Gharb, and Beino (Akkar).

The main campus occupies a suburban area of approximately 550,000 square meters.

The University is affiliated with the Greek Orthodox Church of Antioch but operates as a quasi-secular institution.

UOB welcomes students from all faiths and national or ethnic origins without discrimination based on religion, gender, or physical handicap.

The University commits itself to Christian-Muslim understanding, openness, tolerance, and compassion.

The University offers curricula leading to degrees in more than 71 undergraduate majors.

UOB maintains an academic framework largely following the American model of higher education.

The primary language of instruction at the University is English.

The Lebanese Academy of Fine Arts (ALBA), which predated the university, conducts its programs mainly in French.

The degrees granted by the University of Balamand are recognized worldwide.

The UOB includes 10 Faculties and one Institute, such as the Faculty of Engineering, Faculty of Medicine, and Académie Libanaise des Beaux-Arts (ALBA).

The University system includes five libraries, with the main one being the Issam Fares Library Learning Center.
"""

# Split paragraph into sentences
sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', uob_text) if s]

# --- 2. Setup and Connection ---
# Load USE model
# NOTE: Ensure the model is loaded safely. If model is large, this takes time.
model_PATH = r"C:\Users\User\Desktop\insightface\models\universal-sentence-encoder"
model = hub.load(model_PATH)

# Connect to Qdrant
client = QdrantClient(host="localhost", port=6333)

# Define collection parameters
collection_name = "uob_info"
# Determine the vector size from the loaded model (e.g., USE is 512 dimensions)
# This is a critical step for collection creation
vector_dim = model(["Test sentence"])[0].shape[0]


# --- 3. Check Collection Status and Set Starting ID ---

try:
    # Get collection information
    collection_info = client.get_collection(collection_name=collection_name)
    # Get the current number of points in the collection (the highest used ID is likely count-1)
    start_id = collection_info.points_count
    print(f"Collection '{collection_name}' found. Starting new IDs from: {start_id}")

except:
    # Collection does not exist, so create it
    print(f"Collection '{collection_name}' not found. Creating a new one...")
    client.recreate_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_dim, distance=Distance.COSINE)
    )
    start_id = 0
    print(f"Collection created. Starting new IDs from: {start_id}")


# --- 4. Prepare Points with Sequential IDs ---
points = []
for i, sentence in enumerate(sentences):
    # Calculate the new, unique ID
    point_id = start_id + i

    # Create the vector
    vector = model([sentence])[0].numpy().tolist()

    # Create the PointStruct
    points.append(PointStruct(id=point_id, vector=vector, payload={"text": sentence}))


# --- 5. Upload Points to Qdrant ---
# Using the 'wait=True' parameter ensures the operation completes before the script finishes
client.upsert(collection_name=collection_name, points=points, wait=True)

print(f"Successfully uploaded {len(sentences)} new sentences to Qdrant, starting from ID {start_id}!")