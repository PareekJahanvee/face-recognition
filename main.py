import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.models import Model
import numpy as np
import cv2
import os
import time
import chromadb
from tqdm import tqdm
from mtcnn import MTCNN

#settings
DATASET_PATH = "./dataset"
CHROMA_DB_PATH = "./new_chroma_db_store"  #storage path
COLLECTION_NAME = "face_embeddings_resnet_new"  # collection name
EMBEDDING_DIM = 2048
RECOGNITION_THRESHOLD = 0.05  # For cosine similarity

#global variables
face_detector = None
embedding_model = None
embeddings_collection = None

# Initialize ResNet50 model for embeddings
def initialize_embedding_model():
    print("Log: Initializing ResNet50 embedding model...")
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    x = base_model.output
    embeddings_output = GlobalAveragePooling2D()(x)
    model = Model(inputs=base_model.input, outputs=embeddings_output)
    print("Log: ResNet50 embedding model initialized.")
    return model

# Initialize MTCNN face detector
def initialize_face_detector():
    print("Log: Initializing MTCNN face detector...")
    detector = MTCNN()
    print("Log: MTCNN face detector initialized.")
    return detector

#Get face embedding from an image file
def get_face_embedding(image_path):
    global face_detector, embedding_model
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image {image_path}")
        return None, None
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faces = face_detector.detect_faces(img_rgb)
    if len(faces) == 0:
        print(f"Warning: No face detected in {image_path}")
        return None, None
    #bounding box area
    face = max(faces, key=lambda f: f['box'][2] * f['box'][3])
    x, y, w, h = face['box']
    # Ensure coordinates are non-negative and within image bounds
    x, y = max(0, x), max(0, y)
    face_roi = img[y:y+h, x:x+w]
    if face_roi.shape[0] == 0 or face_roi.shape[1] == 0:
        print(f"Warning: Invalid face ROI in {image_path}")
        return None, None
    face_roi_resized = cv2.resize(face_roi, (224, 224))
    face_roi_normalized = preprocess_input(np.expand_dims(face_roi_resized, axis=0))
    embedding = embedding_model.predict(face_roi_normalized, verbose=0)[0]
    return embedding, face_roi

# Generate and store embeddings
def generate_and_store_embeddings():
    global embeddings_collection
    print("\nLog: Starting embedding generation and storage process...")
    user_names = []
    embeddings_list = []
    metadata_list = []
    chroma_ids = []
    users = [d for d in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, d))]
    if not users:
        print(f"Error: No user folders found in {DATASET_PATH}.")
        return
    for user_name in tqdm(users, desc="Processing users for embeddings"):
        user_dir = os.path.join(DATASET_PATH, user_name)
        for img_name in os.listdir(user_dir):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(user_dir, img_name)
                start_time = time.time()
                embedding, _ = get_face_embedding(image_path)
                embedding_gen_time = time.time() - start_time
                print(f"Log: Embedding gen time for {image_path}: {embedding_gen_time:.4f} seconds")
                if embedding is not None:
                    embeddings_list.append(embedding.tolist())
                    metadata_list.append({"user_name": user_name, "image_name": img_name})
                    chroma_ids.append(f"{user_name}_{os.path.splitext(img_name)[0]}")
                else:
                    print(f"Warning: Skipped {image_path} due to no face detected or error.")
    if embeddings_list:
        start_time_db_add = time.time()
        embeddings_collection.add(
            embeddings=embeddings_list,
            metadatas=metadata_list,
            ids=chroma_ids
        )
        db_add_time = time.time() - start_time_db_add
        print(f"Log: Time to add {len(embeddings_list)} embeddings to ChromaDB: {db_add_time:.4f} seconds")
        print(f"Log: Successfully added {len(embeddings_list)} embeddings to ChromaDB collection '{COLLECTION_NAME}'.")
    else:
        print("Warning: No embeddings generated. Check dataset or face detection.")

# Real-time face recognition from video
def recognize_face_from_video():
    global face_detector, embedding_model, embeddings_collection
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    print("\nLog: Starting real-time face recognition (Press 'q' to quit)...")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame.")
            break
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = face_detector.detect_faces(frame_rgb)
        for face in faces:
            x, y, w, h = face['box']
            x, y = max(0, x), max(0, y)
            face_roi = frame[y:y+h, x:x+w]
            if face_roi.shape[0] == 0 or face_roi.shape[1] == 0:
                continue
            face_roi_resized = cv2.resize(face_roi, (224, 224))
            face_roi_normalized = preprocess_input(np.expand_dims(face_roi_resized, axis=0))
            start_time_embedding = time.time()
            current_face_embedding = embedding_model.predict(face_roi_normalized, verbose=0)[0]
            embedding_gen_time_live = time.time() - start_time_embedding
            print(f"Log: Live embedding gen time: {embedding_gen_time_live:.4f} seconds")
            start_time_matching = time.time()
            results = embeddings_collection.query(
                query_embeddings=[current_face_embedding.tolist()],
                n_results=3  # Get top 3 matches for debugging
            )
            matching_time = time.time() - start_time_matching
            print(f"Log: Matching time in ChromaDB: {matching_time:.4f} seconds")
            print(f"Log: Recognition threshold: {RECOGNITION_THRESHOLD}")
            label = "Unknown"
            color = (0, 0, 255)
            if results and results['ids'] and results['distances']:
                for i, (dist, meta, id) in enumerate(zip(results['distances'][0], results['metadatas'][0], results['ids'][0])):
                    print(f"Match {i+1}: {meta['user_name']} (Distance: {dist:.4f})")
                distance = results['distances'][0][0]
                matched_user_name = results['metadatas'][0][0]['user_name']
                if distance > RECOGNITION_THRESHOLD:  # Cosine similarity
                    label = f"Match: {matched_user_name} ({distance:.2f})"
                    color = (0, 255, 0)
                else:
                    label = f"Unknown ({distance:.2f})"
                    color = (0, 165, 255)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.imshow('Face Recognition System - ResNet50 + ChromaDB', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    print("\nLog: Video stream stopped.")

# --- Main Execution ---
if __name__ == "__main__":
    print("--- Facial Recognition System Start ---")
    try:
        embedding_model = initialize_embedding_model()
        face_detector = initialize_face_detector()
        print(f"Log: Initializing ChromaDB client at {CHROMA_DB_PATH}...")
        chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        embeddings_collection = chroma_client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}  # Use cosine similarity
        )
        print(f"Log: ChromaDB collection '{COLLECTION_NAME}' ready. Current count: {embeddings_collection.count()} embeddings.")
    except Exception as e:
        print(f"Initialization Error: {e}")
        exit()
    if embeddings_collection.count() == 0:
        print("\nLog: No embeddings found in ChromaDB. Generating from dataset...")
        generate_and_store_embeddings()
    else:
        print("\nLog: Embeddings already exist in ChromaDB. Skipping generation.")
        print("To regenerate, delete the 'new_chroma_db_store' folder and re-run.")
    recognize_face_from_video()
    print("\n--- Facial Recognition System End ---")