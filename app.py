import streamlit as st
import cv2
import numpy as np
import tempfile
from keras_facenet import FaceNet
from PIL import Image
import pickle
import os
import urllib.request

# === Setup Paths ===
AGE_BUCKETS = ['(0-3)', '(4-8)', '(9-14)', '(15-20)', '(20-24)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
GENDERS = ['Male', 'Female']
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

# Get the absolute path to the directory where this script lives
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load Age and Gender models (assumed to be in the same directory as this script)
age_net = cv2.dnn.readNetFromCaffe(
    os.path.join(BASE_DIR, 'age_deploy.prototxt'),
    os.path.join(BASE_DIR, 'age_net.caffemodel')
)

gender_net = cv2.dnn.readNetFromCaffe(
    os.path.join(BASE_DIR, 'gender_deploy.prototxt'),
    os.path.join(BASE_DIR, 'gender_net.caffemodel')
)

# Load the FaceNet embedder
embedder = FaceNet()

# Load the KNN model (knn_model.pkl is in the same directory)
with open(os.path.join(BASE_DIR, 'knn_model.pkl'), "rb") as f:
    knn = pickle.load(f)

# Load Haar Cascade for face detection (uses OpenCV's built-in path)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# === Inference Logic ===
def predict_faces(image):
    results = []
    faces = face_cascade.detectMultiScale(image, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        face_img = image[y:y+h, x:x+w]
        blob = cv2.dnn.blobFromImage(face_img, 1, (227, 227),
                                     (78.426, 87.768, 114.895), swapRB=False)

        gender_net.setInput(blob)
        gender = GENDERS[gender_net.forward().argmax()]

        age_net.setInput(blob)
        age = AGE_BUCKETS[age_net.forward().argmax()]

        try:
            face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            embedding = embedder.embeddings([face_rgb])[0].reshape(1, -1)
            pred_name = knn.predict(embedding)[0]
            prob = knn.predict_proba(embedding)[0].max()
            name_label = pred_name if prob > 0.6 else "Unknown"
        except Exception:
            name_label = "Unknown"

        label = f"{name_label}, {gender}, {age}"
        results.append(((x, y, w, h), label))

    return results

def draw_predictions(image, results):
    for (x, y, w, h), label in results:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(image, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return image

# === Streamlit App ===
st.title("🦊 Face Recognition System")

mode = st.sidebar.radio("Choose input type:", ("Image", "Video", "Webcam"))

if mode == "Image":
    uploaded_img = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])
    if uploaded_img is not None:
        image = Image.open(uploaded_img).convert('RGB')
        image_np = np.array(image)
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        results = predict_faces(image_bgr)
        annotated = draw_predictions(image_bgr.copy(), results)

        st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption="Prediction", use_container_width=True)

elif mode == "Video":
    uploaded_vid = st.file_uploader("Upload a video", type=['mp4', 'mov', 'avi'])
    if uploaded_vid is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_vid.read())
        cap = cv2.VideoCapture(tfile.name)

        stframe = st.empty()
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            results = predict_faces(frame)
            annotated = draw_predictions(frame.copy(), results)
            stframe.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), channels="RGB")

        cap.release()

elif mode == "Webcam":
    st.warning("⚠️ Webcam works only when running locally with `streamlit run app.py`.")

    if 'run_webcam' not in st.session_state:
        st.session_state['run_webcam'] = False

    if not st.session_state['run_webcam']:
        if st.button("Start Webcam", key="start_webcam"):
            st.session_state['run_webcam'] = True

    if st.session_state['run_webcam']:
        stop = st.button("Stop Webcam", key="stop_webcam")
        stframe = st.empty()
        cap = cv2.VideoCapture(0)

        while st.session_state['run_webcam']:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to read from webcam.")
                break

            results = predict_faces(frame)
            annotated = draw_predictions(frame.copy(), results)
            stframe.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), channels="RGB")

            if stop:
                st.session_state['run_webcam'] = False
                break

        cap.release()