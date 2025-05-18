import os
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import traceback

class SignLanguageModel:
    _cached_model = None  

    def __init__(self, model_path=None):
        try:
            if SignLanguageModel._cached_model is not None:
                self.model = SignLanguageModel._cached_model
                print("[INFO] Using cached model")
            else:
                self.model = self._load_model(model_path)
                SignLanguageModel._cached_model = self.model

            self._init_mediapipe()
            self.classes = [
                'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
                'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
                'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space'
            ]

        except Exception as e:
            print(f"[ERROR] Failed to initialize model: {e}")
            traceback.print_exc()
            raise

    def _load_model(self, model_path):
        if model_path and os.path.exists(model_path):
            print(f"[INFO] Loading model from provided path: {model_path}")
            return tf.keras.models.load_model(model_path, compile=False)

        potential_paths = [
            os.path.join(os.path.dirname(__file__), 'asl_model.h5'),
            os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'asl_model.h5'),
            'asl_model.h5',
            '/app/models/asl_model.h5',
        ]

        for path in potential_paths:
            if os.path.exists(path):
                print(f"[INFO] Found model at: {path}")
                return tf.keras.models.load_model(path, compile=False)

        raise FileNotFoundError("Model file not found in expected locations.")

    def _init_mediapipe(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils

    def preprocess_image(self, image):
        if image.shape[0] > 320 or image.shape[1] > 320:
            scale = min(320 / image.shape[0], 320 / image.shape[1])
            image = cv2.resize(image, (0, 0), fx=scale, fy=scale)

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_image)
        annotated_image = image.copy()

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    annotated_image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                )

                h, w, _ = image.shape
                x_min, y_min = w, h
                x_max, y_max = 0, 0

                for landmark in hand_landmarks.landmark:
                    x, y = int(landmark.x * w), int(landmark.y * h)
                    x_min, y_min = min(x_min, x), min(y_min, y)
                    x_max, y_max = max(x_max, x), max(y_max, y)

                padding = 20
                x_min, y_min = max(0, x_min - padding), max(0, y_min - padding)
                x_max, y_max = min(w, x_max + padding), min(h, y_max + padding)

                hand_img = image[y_min:y_max, x_min:x_max]
                if hand_img.size > 0:
                    hand_img = cv2.resize(hand_img, (64, 64))
                    hand_img = cv2.cvtColor(hand_img, cv2.COLOR_BGR2RGB)
                    hand_img = hand_img / 255.0
                    return hand_img, annotated_image, True

        fallback_img = cv2.resize(image, (64, 64))
        fallback_img = cv2.cvtColor(fallback_img, cv2.COLOR_BGR2RGB)
        fallback_img = fallback_img / 255.0
        return fallback_img, annotated_image, False

    def predict(self, image):
        processed_img, annotated_image, hand_detected = self.preprocess_image(image)

        try:
            prediction = self.model.predict(np.expand_dims(processed_img, axis=0))
            predicted_class_idx = np.argmax(prediction[0])
            confidence = float(prediction[0][predicted_class_idx])
            predicted_class = self.classes[predicted_class_idx]

            label = predicted_class if hand_detected else f"fallback: {predicted_class}"
            cv2.putText(
                annotated_image,
                f"{label} ({confidence:.2f})",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            return annotated_image, label, confidence

        except Exception as e:
            print(f"[ERROR] Prediction failed: {e}")
            traceback.print_exc()
            return annotated_image, "prediction_error", 0.0

    def fallback_predict(self, image):
        try:
            img = cv2.resize(image, (64, 64))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img / 255.0
            prediction = self.model.predict(np.expand_dims(img, axis=0))
            predicted_class_idx = np.argmax(prediction[0])
            confidence = float(prediction[0][predicted_class_idx])
            predicted_class = self.classes[predicted_class_idx]
            return predicted_class, confidence
        except Exception as e:
            print(f"[ERROR] Fallback prediction failed: {e}")
            traceback.print_exc()
            return "error", 0.0

    def get_classes(self):
        return self.classes