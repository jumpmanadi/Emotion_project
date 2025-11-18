import customtkinter as ctk
import cv2
from PIL import Image
import numpy as np
import threading
import sounddevice as sd
import scipy.io.wavfile as wav
import librosa
import pickle
import os
from deepface import DeepFace
import pyttsx3

# Import our custom feature extractor
from audio_utils import extract_features

# --- CONFIGURATION ---
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")
MODEL_FILENAME = "emotion_audio_model.pkl"

class EmotionApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        # --- Window Setup ---
        self.title("Hamza's AI: Multimodal Emotion Recognition (Voice Enabled)")
        self.geometry("1000x700")
        
        # --- Layout Grid ---
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # --- SIDEBAR (Controls) ---
        self.sidebar_frame = ctk.CTkFrame(self, width=200, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, rowspan=4, sticky="nsew")
        
        self.logo_label = ctk.CTkLabel(self.sidebar_frame, text="AI MULTIMODAL", font=ctk.CTkFont(size=20, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))
        
        # Mode Switches
        self.lbl_modes = ctk.CTkLabel(self.sidebar_frame, text="Active Modes:", anchor="w", font=ctk.CTkFont(weight="bold"))
        self.lbl_modes.grid(row=1, column=0, padx=20, pady=(10, 0))

        self.use_face_var = ctk.StringVar(value="on")
        self.switch_face = ctk.CTkSwitch(self.sidebar_frame, text="Facial Emotion", variable=self.use_face_var, onvalue="on", offvalue="off")
        self.switch_face.grid(row=2, column=0, padx=20, pady=10)

        # Buttons
        self.btn_record = ctk.CTkButton(self.sidebar_frame, text="🎤 Record Audio", command=self.start_audio_recording, fg_color="#C0392B", hover_color="#E74C3C")
        self.btn_record.grid(row=4, column=0, padx=20, pady=20)

        self.btn_video = ctk.CTkButton(self.sidebar_frame, text="📷 Open Camera", command=self.toggle_camera, fg_color="#2980B9", hover_color="#3498DB")
        self.btn_video.grid(row=5, column=0, padx=20, pady=10)

        # --- MAIN AREA (Output) ---
        self.main_frame = ctk.CTkFrame(self, corner_radius=10)
        self.main_frame.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")
        self.main_frame.grid_rowconfigure(0, weight=1)
        self.main_frame.grid_columnconfigure(0, weight=1)

        # Video Feed
        self.video_label = ctk.CTkLabel(self.main_frame, text="Camera Feed Inactive", width=640, height=480, corner_radius=10, fg_color="gray20")
        self.video_label.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")

        # Output Box
        self.result_box = ctk.CTkTextbox(self.main_frame, height=150, font=("Consolas", 14))
        self.result_box.grid(row=1, column=0, padx=20, pady=(0, 20), sticky="ew")
        self.result_box.insert("0.0", "System Ready...\n")

        # --- INITIALIZATION ---
        self.audio_model = self.load_audio_model()
        self.is_recording = False
        self.camera_active = False
        self.cap = None
        self.frame_count = 0
        self.current_face_emotion = "neutral"
        
        # TTS Engine (Voice)
        self.engine = pyttsx3.init()

    def log(self, message):
        self.result_box.insert("end", f"\n{message}")
        self.result_box.see("end")

    def speak(self, text):
        """Speaking in a separate thread to not freeze GUI"""
        def _speak():
            try:
                self.engine.say(text)
                self.engine.runAndWait()
            except: pass
        threading.Thread(target=_speak, daemon=True).start()

    def load_audio_model(self):
        if not os.path.exists(MODEL_FILENAME):
            self.log(f"WARNING: '{MODEL_FILENAME}' not found. Running DEMO MODE.")
            return None
        try:
            with open(MODEL_FILENAME, 'rb') as f:
                return pickle.load(f)
        except: return None

    # --- AUDIO LOGIC ---
    def start_audio_recording(self):
        if self.is_recording: return
        threading.Thread(target=self.record_and_analyze, daemon=True).start()

    def record_and_analyze(self):
        self.is_recording = True
        self.btn_record.configure(text="Recording...", state="disabled")
        try:
            fs = 44100
            myrecording = sd.rec(int(3 * fs), samplerate=fs, channels=1, dtype='float32')
            sd.wait()
            # Save audio temporarily
            wav.write('temp_audio.wav', fs, myrecording)
            self.analyze_audio(myrecording, fs)
            # Clean up temp file
            if os.path.exists('temp_audio.wav'):
                os.remove('temp_audio.wav')
        except Exception as e: self.log(f"Audio Error: {e}")
        finally:
            self.is_recording = False
            self.btn_record.configure(text="🎤 Record Audio", state="normal")

    def analyze_audio(self, audio_data, sample_rate):
        detected_emotion = "unknown"
        if self.audio_model:
            feature = extract_features(audio_data.flatten(), sample_rate)
            if feature is not None:
                # Model expects 2D array
                detected_emotion = self.audio_model.predict(feature.reshape(1, -1))[0]
        else:
            # Fallback for demo
            import random
            detected_emotion = random.choice(['happy', 'sad', 'angry'])
        
        self.log(f"Audio: {detected_emotion}")
        self.apply_fusion_logic(detected_emotion)
        self.speak(f"You sound {detected_emotion}")

    # --- VIDEO LOGIC ---
    def toggle_camera(self):
        if self.camera_active:
            self.camera_active = False
            self.btn_video.configure(text="📷 Open Camera", fg_color="#2980B9")
            if self.cap: self.cap.release()
            self.video_label.configure(image=None, text="Inactive")
        else:
            self.camera_active = True
            self.btn_video.configure(text="■ Stop Camera", fg_color="gray")
            self.cap = cv2.VideoCapture(0)
            self.update_camera_feed()

    def update_camera_feed(self):
        if not self.camera_active: return
        ret, frame = self.cap.read()
        if not ret: return

        # Face Emotion (Every 30 frames)
        self.frame_count += 1
        if self.frame_count % 30 == 0 and self.use_face_var.get() == "on":
            threading.Thread(target=self.analyze_face, args=(frame.copy(),), daemon=True).start()

        # Display
        frame_resized = cv2.resize(frame, (640, 480))
        img = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGBA)
        # Keep reference to avoid garbage collection
        self.imgtk = ctk.CTkImage(light_image=Image.fromarray(img), size=(640, 480))
        self.video_label.configure(image=self.imgtk, text="")
        self.after(20, self.update_camera_feed)

    def analyze_face(self, frame):
        try:
            # Using enforce_detection=False to prevent crashes if face not found
            res = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False, silent=True)
            if isinstance(res, list):
                self.current_face_emotion = res[0]['dominant_emotion']
            else:
                self.current_face_emotion = res['dominant_emotion']
        except: 
            self.current_face_emotion = "neutral"

    def apply_fusion_logic(self, audio_emotion):
        face = self.current_face_emotion
        self.log(f"Fusion -> Audio: {audio_emotion} | Face: {face}")
        
        if face == 'happy':
            verdict = "POSITIVE"
            self.result_box.configure(fg_color="#27AE60")
        elif face in ['sad', 'angry', 'fear']:
            verdict = "NEGATIVE"
            self.result_box.configure(fg_color="#C0392B")
        else:
            verdict = "NEUTRAL"
            self.result_box.configure(fg_color="gray20")
        
        self.log(f"Verdict: {verdict}")
        # Speak the final verdict
        self.speak(f"Verdict is {verdict}")

    def on_closing(self):
        self.camera_active = False
        if self.cap: self.cap.release()
        self.destroy()

if __name__ == "__main__":
    app = EmotionApp()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()