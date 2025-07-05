# 🧠 Face Recognition + Age & Gender Detection

Supports input from:
- Uploaded **images**
- Uploaded **videos**
- **Live webcam feed**

---

## 📸 Features

- ✅ Detect faces in images, videos, or webcam
- ✅ Predict **gender** and **age range**
- ✅ Recognize known faces using KNN on FaceNet embeddings
- ✅ Annotate and preview results in real-time

---

## 📂 Project Structure

<pre>
```text
face_recognition_project/
├── app.py                      # Streamlit app
├── knn_model.pkl               # Trained KNN face recognition model
├── age_net.caffemodel          # Pre-trained age detection model
├── age_deploy.prototxt         # Age model config
├── gender_net.caffemodel       # Pre-trained gender detection model
├── gender_deploy.prototxt      # Gender model config
```
</pre>
---

## 🧠 Requirements

Install dependencies using:

```bash
pip install -r requirements.txt
```

## ▶️ How to run

1. Clone the repo

```bash
git clone https://github.com/anoshandrews/face_recognition_project.git
cd face_recognition_sytem
```
## Place the model files in the appropriate place

	•	age_net.caffemodel
	•	age_deploy.prototxt
	•	gender_net.caffemodel
	•	gender_deploy.prototxt
	•	knn_model.pkl

## Run the app 

```bash
streamlit run app.py
```

## 🎮 Modes of Input

Select one of the modes from the sidebar:
	•	Image: Upload JPG/PNG image
	•	Video: Upload a video file (MP4/MOV/AVI)
	•	Webcam: Use your webcam (requires local execution)

⸻

## 🧠 Tech Stack
	•	Python
	•	OpenCV
	•	Keras-FaceNet
	•	Streamlit
	•	Scikit-learn
	•	Caffe models (age/gender detection)

## 🙌 Acknowledgements
	•	Keras-FaceNet
	•	LearnOpenCV Age & Gender Models
	•	Streamlit

## 📬 Contact

Made by Anosh Andrews
Feel free to fork, star ⭐️, or raise issues.
anoshandrews@gmail.com

