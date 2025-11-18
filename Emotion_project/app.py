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

# Import our custom feature extractor
from audio_utils import extract_features

# --- CONFIGURATION ---
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

# filenames for deployment
MODEL_FILENAME = "model.pkl"
SCALER_FILENAME = "scaler.pkl"
ENCODER_FILENAME = "labelencoder.pkl"

class EmotionApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        # --- Window Setup ---
        self.title("Emotion Detection via Audio & Facial Recognition")
        self.geometry("900x600")
        
        # --- Layout Grid (Sidebar and Main Area) ---
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # --- SIDEBAR (Controls) ---
        self.sidebar_frame = ctk.CTkFrame(self, width=200, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, rowspan=4, sticky="nsew")
        
        self.logo_label = ctk.CTkLabel(self.sidebar_frame, text="AI SENTIMENT", font=ctk.CTkFont(size=20, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))
        
        self.lbl_instruction = ctk.CTkLabel(self.sidebar_frame, text="Select Input Mode:", anchor="w")
        self.lbl_instruction.grid(row=1, column=0, padx=20, pady=10)

        # Checkbox for Facial Recognition
        self.use_face_var = ctk.StringVar(value="on")
        self.switch_face = ctk.CTkSwitch(self.sidebar_frame, text="Facial Recognition",
                                         variable=self.use_face_var, onvalue="on", offvalue="off")
        self.switch_face.grid(row=2, column=0, padx=20, pady=10)

        # Buttons
        self.btn_record = ctk.CTkButton(self.sidebar_frame, text="🎤 Record Audio",
                                        command=self.start_audio_recording,
                                        fg_color="#C0392B", hover_color="#E74C3C")
        self.btn_record.grid(row=3, column=0, padx=20, pady=20)

        self.btn_video = ctk.CTkButton(self.sidebar_frame, text="📷 Open Camera",
                                       command=self.toggle_camera,
                                       fg_color="#2980B9", hover_color="#3498DB")
        self.btn_video.grid(row=4, column=0, padx=20, pady=10)

        # --- MAIN AREA (Output) ---
        self.main_frame = ctk.CTkFrame(self, corner_radius=10)
        self.main_frame.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")
        self.main_frame.grid_rowconfigure(0, weight=1)
        self.main_frame.grid_columnconfigure(0, weight=1)

        # Video Feed Placeholder
        self.video_label = ctk.CTkLabel(self.main_frame, text="Camera Feed Inactive",
                                        width=640, height=480,
                                        corner_radius=10, fg_color="gray20")
        self.video_label.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")

        # Text Output Box
        self.result_box = ctk.CTkTextbox(self.main_frame, height=150, font=("Consolas", 14))
        self.result_box.grid(row=1, column=0, padx=20, pady=(0, 20), sticky="ew")
        self.result_box.insert("0.0", "System Ready...\nWaiting for input.")

        # --- CLASS VARIABLES ---
        self.audio_model = None
        self.scaler = None
        self.label_encoder = None

        self.is_recording = False
        self.camera_active = False
        self.cap = None  # OpenCV VideoCapture object
        self.frame_count = 0
        self.current_face_emotion = "neutral"

        # Load audio-related models
        self.audio_model = self.load_audio_model()

    def log(self, message):
        """Helper function to print to the GUI text box."""
        self.result_box.insert("end", f"\n{message}")
        self.result_box.see("end")  # Auto-scroll to the bottom

    def load_audio_model(self):
        """Loads the trained audio model, scaler, and label encoder from disk."""
        missing = []
        if not os.path.exists(MODEL_FILENAME):
            missing.append(MODEL_FILENAME)
        if not os.path.exists(SCALER_FILENAME):
            missing.append(SCALER_FILENAME)
        if not os.path.exists(ENCODER_FILENAME):
            missing.append(ENCODER_FILENAME)

        if missing:
            self.log(f"WARNING: Missing files: {', '.join(missing)}")
            self.log("Please run 'python train_model.py' first.")
            self.log("Running in DEMO mode (random audio predictions).")
            return None
        
        try:
            with open(MODEL_FILENAME, 'rb') as f:
                model = pickle.load(f)
            with open(SCALER_FILENAME, 'rb') as f:
                scaler = pickle.load(f)
            with open(ENCODER_FILENAME, 'rb') as f:
                label_encoder = pickle.load(f)

            self.scaler = scaler
            self.label_encoder = label_encoder

            self.log("Audio model, scaler, and label encoder loaded successfully.")
            return model

        except Exception as e:
            self.log(f"Error loading audio model components: {e}")
            self.log("Falling back to DEMO mode for audio.")
            return None

    # --- AUDIO LOGIC ---
    def start_audio_recording(self):
        """Triggers the audio recording in a new thread."""
        if self.is_recording:
            return  # Don't start a new recording if one is active

        threading.Thread(target=self.record_and_analyze, daemon=True).start()

    def record_and_analyze(self):
        """Records 3 seconds of audio and analyzes it."""
        self.is_recording = True
        self.btn_record.configure(text="Recording... (3s)", state="disabled", fg_color="gray")
        
        fs = 44100  # Sample rate
        seconds = 3  # Duration of recording
        
        try:
            self.log("Recording audio...")
            myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=1, dtype='float32')
            sd.wait()  # Wait until recording is finished
            
            self.log("Processing Audio...")
            self.analyze_audio(myrecording, fs)
            
        except Exception as e:
            self.log(f"Audio Error: {e}")
        finally:
            self.is_recording = False
            self.btn_record.configure(text="🎤 Record Audio", state="normal", fg_color="#C0392B")

    def analyze_audio(self, audio_data, sample_rate):
        """Analyzes the recorded audio data using the loaded model."""
        detected_emotion = "unknown"
        
        if (self.audio_model is None) or (self.scaler is None) or (self.label_encoder is None):
            # DEMO MODE if model not loaded
            import random
            emotions = ['happy', 'sad', 'angry', 'neutral']
            detected_emotion = random.choice(emotions)
        else:
            # Real Feature Extraction
            # Flatten (n_samples, 1) -> (n_samples,)
            feature = extract_features(audio_data.flatten(), sample_rate)
            
            if feature is None:
                self.log("Could not extract audio features.")
                return

            # Reshape for the model's expected input (1, n_features)
            feature = feature.reshape(1, -1)

            # Scale + predict
            feature_scaled = self.scaler.transform(feature)
            y_pred_encoded = self.audio_model.predict(feature_scaled)
            detected_emotion = self.label_encoder.inverse_transform(y_pred_encoded)[0]

        self.log(f"Audio Analysis Result: User sounds '{detected_emotion.upper()}'")
        # Now, call the fusion logic
        self.apply_fusion_logic(audio_emotion=detected_emotion)

    # --- VIDEO & FUSION LOGIC ---
    def toggle_camera(self):
        """Starts or stops the camera feed."""
        if self.camera_active:
            # Stop the camera
            self.camera_active = False
            self.btn_video.configure(text="📷 Open Camera", fg_color="#2980B9")
            if self.cap:
                self.cap.release()
            self.video_label.configure(image=None, text="Camera Feed Inactive")
        else:
            # Start the camera
            self.camera_active = True
            self.btn_video.configure(text="■ Stop Camera", fg_color="gray")
            
            try:
                self.cap = cv2.VideoCapture(0)  # 0 is the default webcam
                if not self.cap.isOpened():
                    self.log("Error: Cannot open webcam.")
                    self.camera_active = False
                    return
                # Start the camera feed loop
                self.update_camera_feed()
            except Exception as e:
                self.log(f"Camera Error: {e}")
                self.camera_active = False

    def update_camera_feed(self):
        """Continuously updates the video label with frames from the webcam."""
        if not self.camera_active or not self.cap or not self.cap.isOpened():
            return

        ret, frame = self.cap.read()
        if not ret:
            self.log("Error: Can't receive frame.")
            self.after(20, self.update_camera_feed)  # Try again
            return

        # Resize for display
        frame_resized = cv2.resize(frame, (640, 480))

        # --- Facial Analysis ---
        # We run analysis less frequently (every 30 frames) to save resources
        self.frame_count += 1
        if self.frame_count % 30 == 0 and self.use_face_var.get() == "on":
            # Run analysis in a separate thread to avoid freezing the GUI
            threading.Thread(target=self.analyze_frame, args=(frame.copy(),), daemon=True).start()

        # --- Convert for Tkinter Display ---
        try:
            # Convert color from BGR (OpenCV) to RGBA (PIL)
            cv2image = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGBA)
            img = Image.fromarray(cv2image)
            # Create CTkImage
            imgtk = ctk.CTkImage(light_image=img, dark_image=img, size=(640, 480))
            # Update the label
            self.video_label.configure(image=imgtk, text="")
            # Keep a reference to avoid garbage collection
            self.video_label.image = imgtk
        except Exception as e:
            print(f"Error updating image: {e}")

        # Schedule the next update
        self.after(20, self.update_camera_feed)

    def analyze_frame(self, frame):
        """Analyzes a single video frame for emotions using DeepFace."""
        try:
            # 'enforce_detection=False' allows it to analyze even if face is small
            # 'silent=True' suppresses DeepFace's own console logs
            result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False, silent=True)
            
            # DeepFace returns a list of faces; we'll take the first one
            if result and isinstance(result, list) and 'dominant_emotion' in result[0]:
                self.current_face_emotion = result[0]['dominant_emotion']
            else:
                self.current_face_emotion = "neutral"  # Default if no face found

        except Exception as e:
            # This often happens if no face is detected
            self.current_face_emotion = "neutral"

    def apply_fusion_logic(self, audio_emotion):
        """
        Combines Audio and (last seen) Facial emotion for a final verdict.
        """
        # Get the most recently detected face emotion
        face_emotion = self.current_face_emotion
        use_face = self.use_face_var.get() == "on"
        
        self.log(f"--- FUSION ---")
        self.log(f"Audio Detected: {audio_emotion.upper()}")
        if use_face:
            self.log(f"Face Detected:  {face_emotion.upper()}")
        else:
            self.log("Face Detection is OFF. Trusting Audio.")


            

        # --- Fusion Logic ---
        if not use_face:
            final_verdict = f"AUDIO-ONLY VERDICT: {audio_emotion.upper()}"
            self.result_box.configure(fg_color="gray20")
            
        elif face_emotion == 'happy':
            if audio_emotion in ['happy', 'calm', 'surprised']:
                final_verdict = "POSITIVE (Joyful/Excited)"
                self.result_box.configure(fg_color="#27AE60")  # Green
            else:
                final_verdict = "CONFLICT (Happy Face, Negative Audio) -> Sarcasm?"
                self.result_box.configure(fg_color="#F39C12")  # Orange
        
        elif face_emotion in ['sad', 'angry', 'fear']:
            if audio_emotion in ['sad', 'angry', 'fear', 'disgust']:
                final_verdict = "NEGATIVE (Distressed/Angry)"
                self.result_box.configure(fg_color="#C0392B")  # Red
            else:
                final_verdict = "CONFLICT (Sad Face, Positive Audio) -> Masking?"
                self.result_box.configure(fg_color="#F39C12")  # Orange
        
        else:  # face_emotion is 'neutral' or 'disgust'
            final_verdict = f"NEUTRAL (Trusting Audio: {audio_emotion.upper()})"
            self.result_box.configure(fg_color="gray20")
        
        self.log(f"==> FINAL VERDICT: {final_verdict}")

    def on_closing(self):
        """Handles window close event."""
        print("Closing application...")
        self.camera_active = False
        if self.cap:
            self.cap.release()
        self.destroy()

if __name__ == "__main__":
    app = EmotionApp()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)  # Handle X button
    app.mainloop()
