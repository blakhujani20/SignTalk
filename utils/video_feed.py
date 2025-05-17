import cv2
import numpy as np
import os
from gevent import sleep
import logging
import traceback

logger = logging.getLogger('signtalk')

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

            sleep(0.05)  

    except Exception as e:
        print(f"[ERROR] Video capture failed: {e}")
        while True:
            frame = generate_placeholder_frame("Camera not available in production")
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            sleep(0.5)  

def generate_placeholder_frame(message):
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(
        frame, message, (50, 240),
        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2
    )
    return frame

def process_frame(frame, model):
    try:
        if model is None:
            raise ValueError("Model is not loaded")
            
        h, w, _ = frame.shape
        logger.info(f"Processing frame with dimensions {w}x{h}")

        if w > 200:
            frame = cv2.flip(frame, 1)

        try:
            processed_img, annotated_frame, hand_detected = model.preprocess_image(frame)
            logger.info(f"Preprocessed image shape: {processed_img.shape}, Hand detected: {hand_detected}")
        except Exception as preproc_error:
            logger.error(f"Error in preprocess_image: {str(preproc_error)}")
            logger.error(traceback.format_exc())
            return frame, "preprocess_error", 0.0

        try:
            with tf.device('/CPU:0'):
                prediction = model.model(np.expand_dims(processed_img, axis=0), training=False).numpy()

            predicted_class_idx = np.argmax(prediction[0])
            confidence = float(prediction[0][predicted_class_idx])
            predicted_class = model.classes[predicted_class_idx]
            logger.info(f"Model prediction successful: {predicted_class} with confidence {confidence:.2f}")

        except Exception as pred_error:
            logger.error(f"Error in model prediction: {str(pred_error)}")
            logger.error(traceback.format_exc())
            return frame, "prediction_error", 0.0


        cv2.putText(
            frame,
            f"{predicted_class}: {confidence:.2f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        return frame, predicted_class, confidence

    except Exception as e:
        logger.error(f"Error in process_frame: {str(e)}")
        logger.error(traceback.format_exc())
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