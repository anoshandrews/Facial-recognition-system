import cv2
import os

# Base path of the entire project
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# Correct paths
haar_path = os.path.join(BASE_DIR, 'haarcascade_frontalface_default.xml')
input_dir = os.path.join(BASE_DIR, 'my_images')
output_dir = os.path.join(BASE_DIR, 'face_recognition_project', 'facial_recognition_system', 'anosh')
# Load Haar cascade
haar_cascade = cv2.CascadeClassifier(haar_path)

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Supported image extensions
image_extensions = ['.jpg', '.jpeg', '.png']

# Counter to name faces uniquely
face_count = 0

# Loop through all files in the input directory
for filename in os.listdir(input_dir):
    if any(filename.lower().endswith(ext) for ext in image_extensions):
        img_path = os.path.join(input_dir, filename)
        img = cv2.imread(img_path)

        if img is None:
            print(f"[WARNING] Could not read image: {img_path}")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = haar_cascade.detectMultiScale(gray, scaleFactor=1.4, minNeighbors=4, minSize=(100, 100))

        for (x, y, w, h) in faces:
            cropped_face = gray[y:y+h, x:x+w]
            face_filename = os.path.join(output_dir, f"anosh_{face_count}.jpg")
            cv2.imwrite(face_filename, cropped_face)
            print(f"[INFO] Saved: {face_filename}")
            face_count += 1