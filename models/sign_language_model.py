import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import os

class SignLanguageModel:
    def __init__(self, model_path=None):
        try:
            if model_path is None:
                potential_paths = [
                    os.path.join(os.path.dirname(__file__), 'asl_model.h5'),
                    os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'asl_model.h5'),
                    'asl_model.h5',  # Current directory
                    '/app/models/asl_model.h5',  # Docker container path
                ]
                
                for path in potential_paths:
                    if os.path.exists(path):
                        model_path = path
                        print(f"[INFO] Found model at: {model_path}")
                        break
                        
                if model_path is None:
                    raise FileNotFoundError("Could not find model file in any of the expected locations")
            
            print(f"[INFO] Loading model from: {model_path}")
            self.model = tf.keras.models.load_model(model_path, compile=False)
            
            print("[DEBUG] Model input shape:", self.model.input_shape)
            print("[DEBUG] Model output shape:", self.model.output_shape)
            
            self.mp_hands = mp.solutions.hands
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            self.mp_drawing = mp.solutions.drawing_utils

            self.classes = [
                'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
                'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
                'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space'
            ]
        except Exception as e:
            print(f"[ERROR] Failed to initialize model: {e}")
            raise
    
    def preprocess_image(self, image):

        if image.shape[0] < 100 or image.shape[1] < 100:
            print("[DEBUG] Frame too small for reliable hand detection.")

            fallback_img = cv2.resize(image, (64, 64))
            fallback_img = cv2.cvtColor(fallback_img, cv2.COLOR_BGR2RGB)
            fallback_img = fallback_img / 255.0
            return fallback_img, image, False 

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_image)

        annotated_image = image.copy()
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    annotated_image, 
                    hand_landmarks, 
                    self.mp_hands.HAND_CONNECTIONS
                )
                
                h, w, _ = image.shape
                x_min, y_min = w, h
                x_max, y_max = 0, 0

                for landmark in hand_landmarks.landmark:
                    x, y = int(landmark.x * w), int(landmark.y * h)
                    x_min = min(x_min, x)
                    y_min = min(y_min, y)
                    x_max = max(x_max, x)
                    y_max = max(y_max, y)

                padding = 20
                x_min = max(0, x_min - padding)
                y_min = max(0, y_min - padding)
                x_max = min(w, x_max + padding)
                y_max = min(h, y_max + padding)

                hand_img = image[y_min:y_max, x_min:x_max]
                if hand_img.shape[0] > 0 and hand_img.shape[1] > 0:
                    hand_img = cv2.resize(hand_img, (64, 64))
                    hand_img = cv2.cvtColor(hand_img, cv2.COLOR_BGR2RGB)
                    hand_img = hand_img / 255.0
                    return hand_img, annotated_image, True

        print("[DEBUG] No hand detected via Mediapipe, using fallback frame.")
        fallback_img = cv2.resize(image, (64, 64))
        fallback_img = cv2.cvtColor(fallback_img, cv2.COLOR_BGR2RGB)
        fallback_img = fallback_img / 255.0
        return fallback_img, annotated_image, False

    
    def predict(self, image):
        processed_img, annotated_image, hand_detected = self.preprocess_image(image)

        if processed_img is None:
            print("[DEBUG] No image processed.")
            return annotated_image, "no_hand", 0.0

        try:
            prediction = self.model.predict(np.expand_dims(processed_img, axis=0))
        except Exception as e:
            print(f"[ERROR] Model prediction failed: {e}")
            return annotated_image, "prediction_error", 0.0

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

    
    def get_classes(self):
        return self.classes
    
    # Add to SignLanguageModel class
    def fallback_predict(self, image):
        """Provide a simple fallback when regular prediction fails"""
        try:
            # Resize and normalize
            img = cv2.resize(image, (64, 64))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img / 255.0
            
            # Make prediction
            prediction = self.model.predict(np.expand_dims(img, axis=0))
            predicted_class_idx = np.argmax(prediction[0])
            confidence = float(prediction[0][predicted_class_idx])
            predicted_class = self.classes[predicted_class_idx]
            
            return predicted_class, confidence
        except Exception as e:
            print(f"[ERROR] Fallback prediction failed: {e}")
            return "error", 0.0
