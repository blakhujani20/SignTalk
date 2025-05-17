# SignTalk: Real-Time Sign Language Video Communication

SignTalk is a web-based real-time sign language communication app that bridges the gap between deaf and non-deaf individuals. It uses a trained AI model to recognize American Sign Language (ASL) hand signs from video and converts them to readable text during a video call.

## 🚀 Features

- 🔤 Real-time ASL hand sign detection
- 📹 Live video call communication (WebRTC + Flask-SocketIO)
- 🧠 Machine learning-based sign recognition (TensorFlow + MediaPipe)
- 📄 Sentence formation and prediction history
- 💻 Practice/Demo mode to try recognition without video call
- 🌙 Light/Dark mode toggle

## 🧩 Tech Stack

- **Frontend:** HTML, CSS, JavaScript
- **Backend:** Flask, Flask-SocketIO, Gunicorn
- **Model:** TensorFlow, MediaPipe
- **Video:** WebRTC

## 📁 Project Structure

```
├── app.py                  # Flask app and routes
├── models/
│   ├── asl_model.h5        # Trained sign language model
│   ├── sign_language_model.py
├── utils/
│   ├── video_feed.py       # Webcam streaming & frame generation
│   ├── postprocessing.py   # Sentence formation from predictions
├── templates/
│   ├── index.html          # Landing + demo page
│   ├── video_call.html     # Video call interface
│   ├── base.html, about.html
├── static/
│   ├── css/style.css
│   ├── js/main.js, call.js, sign_detection_for_calls.js
├── notebooks/model.ipynb   # Model training notebook
├── requirements.txt
└── README.md
```

## 🧪 Demo Usage

1. Go to [https://signtalk-pf3b.onrender.com](https://signtalk-pf3b.onrender.com) or run locally (see below)
2. Try the **Demo Mode** from homepage to test recognition without a call
3. For full experience:
   - Open the Video Call page
   - Join with one tab in Deaf Mode, one in Regular Mode
   - Start signing — the AI will translate in real-time!

## ⚙️ Run Locally

```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate #on Windows

# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py
```

Open browser at [http://localhost:5000](http://localhost:5000)


> Feel free to ⭐️ the repo if you find it useful or inspiring!
