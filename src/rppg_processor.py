import os # -for checking model 
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import mediapipe as mp
import matplotlib.pyplot as plt
from collections import deque
import time
import io
import streamlit as st
import gdown # new import for g-drive
import sys
from io import StringIO
import contextlib

# -----
# Secure Model Loader
# -----

@st.cache_resource
def load_rppg_model():
    """Download model from Google Drive if not Found Locally, then Load it."""
    
    MODEL_PATH = "best_rppg_model.h5"
    
    # Check if model exists first - skip download
    if os.path.exists(MODEL_PATH):
        st.info("Model already exists locally. Loading...")
        try:
            model = load_model(MODEL_PATH)
            st.success("rPPG model loaded successfully.")
            return model
        except Exception as e:
            st.error(f"Could not load model: {str(e)}")
            raise e
    
    # Only download if model doesn't exist
    try:
        DRIVE_FILE_ID = st.secrets.get("DRIVE_FILE_ID", "1amZ-gOSjoHDm-Kr0O6xYBbOqiUzsX2wK")
    except:
        DRIVE_FILE_ID = "1amZ-gOSjoHDm-Kr0O6xYBbOqiUzsX2wK"
    
    st.warning("Model not found locally. Downloading from secure cloud storage...")
    try:
        url = f"https://drive.google.com/uc?id={DRIVE_FILE_ID}"
        
        with contextlib.redirect_stdout(StringIO()), contextlib.redirect_stderr(StringIO()):
            gdown.download(url, MODEL_PATH, quiet=True, fuzzy=True)
        
        st.success("Model downloaded successfully.")
    except Exception as e:
        st.error(f"Failed to download model: {str(e)}")
        st.info(f"Please download manually and place as '{MODEL_PATH}'")
        raise e
    
    # Load the newly downloaded model
    try:
        model = load_model(MODEL_PATH)
        st.success("rPPG model loaded successfully.")
        return model
    except Exception as e:
        st.error(f"Could not load model: {str(e)}")
        raise e
        
# -----
# RPPGProcessor class
# -----

class RPPGProcessor:
    def __init__(self, session_duration):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.cap = None
        self.session_duration = session_duration
        self.accumulated_time = 0
        self.last_aligned_time = None
        self.misalignment_count = 0
        self.is_counting = False
        self.hr_values = deque(maxlen=30)
        self.bp_values = deque(maxlen=30)
        self.timestamps = deque(maxlen=30)
        self.all_hr_values = []
        self.all_bp_values = []
        self.all_timestamps = []
        self.signal_buffer = deque(maxlen=150)
        
        # Secure model loading
        self.model =load_rppg_model()
        
        #----eraise it 
           
        # Load model
        # try:
        #     self.model = load_model("best_rppg_model.h5")  # Adjusted path to root directory
        #     st.success("rPPG model loaded successfully")
        # except Exception as e:
        #     st.error(f"Error: Could not load model: {str(e)}. Exiting.")
        #     raise Exception("Model loading failed")
        # ------
        
        # Setup plotting
        self.fig, self.ax = plt.subplots(2, 1, figsize=(6, 4))
        self.hr_line, = self.ax[0].plot([], [], 'g-', linewidth=2)
        self.bp_line, = self.ax[1].plot([], [], 'r-', linewidth=2)
        self.ax[0].set_ylim(40, 180)
        self.ax[0].set_title("Heart Rate (BPM)")
        self.ax[0].set_xlabel("Time (s)")
        self.ax[0].grid(True)
        self.ax[1].set_ylim(60, 200)
        self.ax[1].set_title("Systolic BP (mmHg)")
        self.ax[1].set_xlabel("Time (s)")
        self.ax[1].grid(True)
        plt.tight_layout()


    def load_video(self, video_file):
        """Load uploaded video file"""
        # Reset session state when loading a new video
        self.accumulated_time = 0
        self.misalignment_count = 0
        self.hr_values.clear()
        self.bp_values.clear()
        self.timestamps.clear()
        self.signal_buffer.clear()
        
        # Load new video file
        tfile = video_file.read()
        with open("temp_video.mp4", "wb") as f:
            f.write(tfile)
        self.cap = cv2.VideoCapture("temp_video.mp4")
        ret, frame = self.cap.read()
        if not ret:
            st.error("Error: Could not read video frames.")
            raise Exception("Video loading failed")
        self.frame_height, self.frame_width = frame.shape[:2]
        self.guide_box = self.calculate_guide_box()
        st.write(f"Video resolution: {self.frame_width}x{self.frame_height}")  # Debug info

    def calculate_guide_box(self):
        """Calculate guide box dimensions dynamically based on frame size"""
        # Make the guide box 50% of frame width and height for better flexibility
        box_width = int(self.frame_width * 0.5)
        box_height = int(self.frame_height * 0.5)
        x = (self.frame_width - box_width) // 2
        y = (self.frame_height - box_height) // 2
        return (x, y, box_width, box_height)

    def draw_guide_box(self, frame):
        """Draw guide box with instructions"""
        x, y, w, h = self.guide_box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        l = 20
        cv2.line(frame, (x, y), (x + l, y), (0, 255, 0), 3)
        cv2.line(frame, (x, y), (x, y + l), (0, 255, 0), 3)
        cv2.line(frame, (x + w, y), (x + w - l, y), (0, 255, 0), 3)
        cv2.line(frame, (x + w, y), (x + w, y + l), (0, 255, 0), 3)
        cv2.line(frame, (x, y + h), (x + l, y + h), (0, 255, 0), 3)
        cv2.line(frame, (x, y + h), (x, y + h - l), (0, 255, 0), 3)
        cv2.line(frame, (x + w, y + h), (x + w - l, y + h), (0, 255, 0), 3)
        cv2.line(frame, (x + w, y + h), (x + w, y + h - l), (0, 255, 0), 3)

    def check_face_position(self, face_landmarks):
        """Check if face is properly positioned in guide box with relaxed tolerances"""
        if not face_landmarks:
            return False, "No face detected"
        x, y, w, h = self.guide_box
        landmarks = face_landmarks.landmark
        face_x = min(l.x for l in landmarks) * self.frame_width
        face_y = min(l.y for l in landmarks) * self.frame_height
        face_w = (max(l.x for l in landmarks) - min(l.x for l in landmarks)) * self.frame_width
        face_h = (max(l.y for l in landmarks) - min(l.y for l in landmarks)) * self.frame_height

        # Relaxed tolerances: allow some overlap beyond the box
        tolerance_x = w * 0.2  # 20% of guide box width
        tolerance_y = h * 0.2  # 20% of guide box height
        min_face_size = w * 0.25  # Minimum 25% of guide box width
        max_face_size = w * 0.9   # Maximum 90% of guide box width

        if face_x < (x - tolerance_x) or (face_x + face_w) > (x + w + tolerance_x):
            return False, f"Center your face horizontally (x: {int(face_x)}, w: {int(face_w)})"
        if face_y < (y - tolerance_y) or (face_y + face_h) > (y + h + tolerance_y):
            return False, f"Center your face vertically (y: {int(face_y)}, h: {int(face_h)})"
        if face_w < min_face_size:
            return False, "Move closer to the camera"
        if face_w > max_face_size:
            return False, "Move away from the camera"
        return True, "Face position good"

    def handle_timing(self, position_ok):
        current_time = time.time()
        if position_ok:
            if not self.is_counting:
                self.is_counting = True
                self.last_aligned_time = current_time
            else:
                self.accumulated_time += current_time - self.last_aligned_time
                self.last_aligned_time = current_time
        else:
            if self.is_counting:
                self.is_counting = False
                self.misalignment_count += 1
                if self.misalignment_count > 2:
                    self.accumulated_time = 0
                    self.misalignment_count = 0
                    self.hr_values.clear()
                    self.bp_values.clear()
                    self.timestamps.clear()
                    self.signal_buffer.clear()
                    return "Timer reset due to multiple misalignments"
            self.last_aligned_time = None
            return "Timer paused - align face in box"
        return f"Time: {self.accumulated_time:.1f}s"

    def extract_face_features(self, frame, face_landmarks):
        landmarks = face_landmarks.landmark
        h, w = frame.shape[:2]
        face_points = np.array([(int(l.x * w), int(l.y * h)) for l in landmarks])
        forehead_indices = [10, 67, 69, 108, 151]
        left_cheek_indices = [123, 147, 187, 205]
        right_cheek_indices = [356, 378, 398, 421]
        forehead_points = np.array([face_points[i] for i in forehead_indices])
        left_cheek_points = np.array([face_points[i] for i in left_cheek_indices])
        right_cheek_points = np.array([face_points[i] for i in right_cheek_indices])
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [forehead_points], 255)
        cv2.fillPoly(mask, [left_cheek_points], 255)
        cv2.fillPoly(mask, [right_cheek_points], 255)
        masked_frame = cv2.bitwise_and(frame, frame, mask=mask)
        r_channel = masked_frame[:, :, 2]
        g_channel = masked_frame[:, :, 1]
        b_channel = masked_frame[:, :, 0]
        non_zero_pixels = np.count_nonzero(mask)
        if non_zero_pixels > 0:
            r_avg = np.sum(r_channel) / non_zero_pixels
            g_avg = np.sum(g_channel) / non_zero_pixels
            b_avg = np.sum(b_channel) / non_zero_pixels
            signal = [r_avg, g_avg, b_avg]
            self.signal_buffer.append(signal)
            return True, signal
        return False, None

    def preprocess_signal(self):
        if len(self.signal_buffer) < 90:
            return None
        signals = np.array(self.signal_buffer)
        r_signal = signals[:, 0]
        g_signal = signals[:, 1]
        b_signal = signals[:, 2]
        r_norm = (r_signal - np.mean(r_signal)) / np.std(r_signal)
        g_norm = (g_signal - np.mean(g_signal)) / np.std(g_signal)
        b_norm = (b_signal - np.mean(b_signal)) / np.std(b_signal)
        r_diff = np.diff(r_norm, prepend=r_norm[0])
        g_diff = np.diff(g_norm, prepend=g_norm[0])
        b_diff = np.diff(b_norm, prepend=b_norm[0])
        combined = (r_norm + g_norm + b_norm) / 3
        processed_signal = np.column_stack((r_norm, g_norm, b_norm, r_diff, g_diff, b_diff, combined))
        current_length = processed_signal.shape[0]
        target_length = 32
        indices = np.round(np.linspace(0, current_length-1, target_length)).astype(int)
        resampled_signal = processed_signal[indices]
        model_input = resampled_signal.reshape(1, 32, 7)
        return model_input

    def predict_heart_rate(self):
        if len(self.signal_buffer) < 90:
            return None
        model_input = self.preprocess_signal()
        if model_input is None:
            return None
        try:
            prediction = self.model.predict(model_input, verbose=0)
            heart_rate = float(prediction[0][0])
            heart_rate = np.clip(heart_rate, 40, 180)
            return heart_rate
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")
            return None

    def process_frame(self, frame):
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        self.draw_guide_box(frame)
        remaining_time = max(0, self.session_duration - self.accumulated_time)

        hr, bp, position_message, timing_message = None, None, "", ""
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            face_points = [(int(l.x * self.frame_width), int(l.y * self.frame_height))
                          for l in face_landmarks.landmark]
            face_x = min(x for x, y in face_points)
            face_y = min(y for x, y in face_points)
            face_w = max(x for x, y in face_points) - face_x
            face_h = max(y for x, y in face_points) - face_y
            # Draw face bounding box for debugging (blue rectangle)
            cv2.rectangle(frame, (int(face_x), int(face_y)), (int(face_x + face_w), int(face_y + face_h)), (255, 0, 0), 2)

            position_ok, position_message = self.check_face_position(face_landmarks)
            timing_message = self.handle_timing(position_ok)

            self.mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_tesselation_style()
            )

            if position_ok and self.is_counting:
                success, _ = self.extract_face_features(frame, face_landmarks)
                if success:
                    hr = self.predict_heart_rate()
                    if hr is not None:
                        systolic, diastolic = self.estimate_blood_pressure(hr)
                        bp = (systolic, diastolic)
                        self.hr_values.append(hr)
                        self.bp_values.append(bp)
                        self.timestamps.append(self.accumulated_time)
                        self.all_hr_values.append(hr)
                        self.all_bp_values.append(bp)
                        self.all_timestamps.append(self.accumulated_time)
                        cv2.putText(frame, f"HR: {hr:.1f} BPM", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        cv2.putText(frame, f"BP: {systolic:.0f}/{diastolic:.0f} mmHg", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.putText(frame, position_message, (10, self.frame_height - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 0) if "good" in position_message.lower() else (0, 0, 255), 2)
        cv2.putText(frame, timing_message, (10, self.frame_height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Time remaining: {int(remaining_time)}s", (10, self.frame_height - 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return frame, hr, bp, f"{position_message} | {timing_message}"

    def update_plot(self):
        if len(self.timestamps) > 0:
            self.hr_line.set_xdata(self.timestamps)
            self.hr_line.set_ydata(self.hr_values)
            self.ax[0].set_xlim(min(self.timestamps), max(self.timestamps) + 2)

            systolic = [bp[0] for bp in self.bp_values]
            self.bp_line.set_xdata(self.timestamps)
            self.bp_line.set_ydata(systolic)
            self.ax[1].set_xlim(min(self.timestamps), max(self.timestamps) + 2)

        buf = io.BytesIO()
        self.fig.savefig(buf, format='png')
        buf.seek(0)
        return buf

    def estimate_blood_pressure(self, heart_rate):
        base_systolic = 120
        base_diastolic = 80
        delta_hr = heart_rate - 70
        systolic = base_systolic + delta_hr * 0.7
        diastolic = base_diastolic + delta_hr * 0.4
        systolic += np.random.normal(0, 1)
        diastolic += np.random.normal(0, 1)
        return systolic, diastolic

    def cleanup(self):
        if self.cap:
            self.cap.release()
        plt.close(self.fig)