<p align="center">
  <img src="https://img.icons8.com/ios-filled/100/ff4b4b/heart-with-pulse.png" width="80" alt="HeartVision AI"/>
</p>

<h1 align="center">❤️ HeartVision AI — rPPG-based Heart Rate & BP Monitor</h1>
<h3 align="center">Non-contact health monitoring using deep learning and remote photoplethysmography (rPPG)</h3>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/TensorFlow-2.15-orange?logo=tensorflow" />
  <img src="https://img.shields.io/badge/Streamlit-Cloud-red?logo=streamlit" />
  <img src="https://img.shields.io/badge/OpenCV-Computer_Vision-green?logo=opencv" />
  <img src="https://img.shields.io/badge/Status-Deployed-success?logo=github" />
</p>


---

## 📖 Overview
**HeartVision AI** is a deep learning–powered web application that estimates **Heart Rate (BPM)** and **Blood Pressure (mmHg)** from facial video streams using **remote photoplethysmography (rPPG)**.

Built using **Streamlit**, it allows users to upload a video, processes facial signals using **MediaPipe FaceMesh**, and predicts vital signs using a trained **TensorFlow/Keras model** — all without any physical sensors.

---

## 🚀 Live Demo
🔗 **Deployed App:**  https://rppghrmonitor.streamlit.app/

> _(Hosted on Streamlit Cloud — model securely loaded from Google Drive)_

---

## 🧠 Features
✅ Real-time **Heart Rate & Blood Pressure** estimation from video  
✅ **Face detection** and tracking using MediaPipe  
✅ **Signal preprocessing** (RGB normalization, differential signals)  
✅ **Deep Learning model (rPPG-based)** for health prediction  
✅ **Interactive Streamlit UI** with progress visualization  
✅ **Final Report Generation** (CSV & PDF)  
✅ **Secure Model Loading** from Google Drive (hidden, not stored on GitHub)

---

## 🧩 System Architecture
User Video → Face Detection (MediaPipe) → Signal Extraction (RGB)
↓
Preprocessing → rPPG Deep Learning Model (TensorFlow)
↓
Heart Rate & BP Prediction → Visualization & Report (Streamlit)

--
## ⚙️ Technologies Used
| Category | Tools / Libraries |
|-----------|------------------|
| 🧠 Deep Learning | TensorFlow, Keras |
| 🎥 Computer Vision | OpenCV, MediaPipe |
| 🌐 Web Framework | Streamlit |
| 📦 Utilities | NumPy, Matplotlib, Pandas, gdown |
| 🗂️ Deployment | Streamlit Cloud |
| ☁️ Model Storage | Google Drive (auto-downloaded securely) |

---

## 🧰 Installation & Setup

### 1️⃣ Clone the Repository

git clone https://github.com/vpsrathod/rppg_hr_monitor.git
cd rppg_hr_monitor

2️⃣ Install Dependencies
pip install -r requirements.txt

3️⃣ (Optional) Add Local Secrets File

If you’re running locally, create:

.streamlit/secrets.toml


and add:

DRIVE_FILE_ID="your_google_drive_model_id"

4️⃣ Run Locally
streamlit run app.py

☁️ Deployment

The app is deployed on Streamlit Cloud.

The trained model (best_rppg_model.h5) is stored securely on Google Drive and automatically downloaded during first run.

No sensitive files are committed to GitHub.

📄 Folder Structure
rppg_hr_monitor/
│
├── app.py                     # Streamlit main frontend
├── requirements.txt            # Dependencies
├── src/
│   ├── rppg_processor.py       # Core logic: face detection, rPPG processing, model prediction
│   ├── reporting.py            # Report generation (CSV, PDF)
│   └── utils/ (optional)
│
├── .gitignore                  # Ignores model & temp files
└── .streamlit/
    └── secrets.toml (for Streamlit Cloud)

🧪 Model Loading Logic

The model is loaded securely and automatically:

If found locally → loads instantly

If missing → downloaded from Google Drive using gdown

Works both locally and on Streamlit Cloud

@st.cache_resource
def load_rppg_model():
    if os.path.exists("best_rppg_model.h5"):
        return load_model("best_rppg_model.h5")
    else:
        gdown.download(f"https://drive.google.com/uc?id={DRIVE_FILE_ID}", "best_rppg_model.h5")
        return load_model("best_rppg_model.h5")

## 🖥️ Application Preview

### 🧩 App Interface
This is the main Streamlit interface where users upload a video and set session duration.

<p align="center">
  <img src="images/App_Interface.png" alt="App Interface" width="800"/>
</p>

---

### ❤️ Real-time Processing & Final Report
Once the video is processed, HeartVision AI tracks face landmarks, extracts color signals, and predicts heart rate and blood pressure using the rPPG model.

<p align="center">
  <img src="images/Processing Results.png" alt="Processing Results" width="800"/>
</p>

> 💡 The app generates real-time plots for Heart Rate (BPM) and Systolic BP (mmHg), followed by a downloadable final report in CSV and PDF formats.


The app provides:

📈 Real-time HR & BP waveform visualization

📊 Downloadable reports in CSV & PDF formats

--

📚 References

MediaPipe FaceMesh Documentation

rPPG Signal Processing Research Papers

TensorFlow Keras API
--

👨‍💻 Developer

Vishnu Pratap Singh Rajput

🎓 B.Tech (AI & ML) — RGPV University

💼 AI/ML Developer | Python & Django Enthusiast | Creative Technologist

🌐 GitHub-https://github.com/vpsrathod

 | LinkedIn-https://linkedin.com/in/vpsr

