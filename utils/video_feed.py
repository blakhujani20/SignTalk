import cv2
import numpy as np
import os
from gevent import sleep

def generate_frames(model):
    try:
        camera = cv2.VideoCapture(0)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        if not camera.isOpened():
            raise RuntimeError("Could not start camera")

        while True:
            success, frame = camera.read()
            if not success:
                break

            processed_frame, _, _ = process_frame(frame, model)
            ret, buffer = cv2.imencode('.jpg', processed_frame)
            processed_frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + processed_frame + b'\r\n')

            sleep(0.05)  # Changed from eventlet.sleep to gevent.sleep

    except Exception as e:
        print(f"[ERROR] Video capture failed: {e}")
        while True:
            frame = generate_placeholder_frame("Camera not available in production")
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            sleep(0.5)  # Changed from eventlet.sleep to gevent.sleep

def generate_placeholder_frame(message):
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(
        frame, message, (50, 240),
        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2
    )
    return frame

def process_frame(frame, model):
    try:
        h, w, _ = frame.shape

        if w > 200:
            frame = cv2.flip(frame, 1)

        if w > 200:
            roi_size = min(h, w) // 2
            roi_x = w // 2 - roi_size // 2
            roi_y = h // 2 - roi_size // 2

            cv2.rectangle(
                frame,
                (roi_x, roi_y),
                (roi_x + roi_size, roi_y + roi_size),
                (0, 255, 0),
                2
            )

            roi_frame = frame[roi_y:roi_y+roi_size, roi_x:roi_x+roi_size]
        else:
            roi_frame = frame

        resized_frame = cv2.resize(roi_frame, (64, 64))
        normalized_frame = resized_frame / 255.0
        input_tensor = np.expand_dims(normalized_frame, axis=0)

        prediction_result = model.model.predict(input_tensor)
        predicted_class_index = np.argmax(prediction_result)
        confidence = float(prediction_result[0][predicted_class_index])
        prediction = model.classes[predicted_class_index]

        if prediction == "no_hand":
            print("[DEBUG] No hand detected in frame.")

        cv2.putText(
            frame,
            f"{prediction}: {confidence:.2f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        return frame, prediction, confidence

    except Exception as e:
        print(f"[ERROR] Frame processing failed: {e}")
        cv2.putText(
            frame,
            f"Error: {str(e)}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )
        return frame, "error", 0.0