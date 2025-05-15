import eventlet
eventlet.monkey_patch()

from flask import Flask, render_template, Response, jsonify, request, send_from_directory
import cv2
import numpy as np
import time
import json
from flask_socketio import SocketIO, emit, join_room, leave_room
import os
import logging
import traceback
from models.sign_language_model import SignLanguageModel
from utils.video_feed import generate_frames, process_frame
from utils.postprocessing import form_sentence

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet', ping_timeout=60, ping_interval=25)

try:
    model = SignLanguageModel()
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    model = None

prediction_history = {}  
last_prediction = {}  
prediction_threshold = 0.6  
cooldown_period = 0.5  
last_prediction_time = {} 

@app.route('/')
def index():
    """Main page with video feed and interpretation"""
    return render_template('index.html')

@app.route('/about')
def about():
    """About page with project information"""
    return render_template('about.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route for webcam feed"""
    return Response(
        generate_frames(model),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

@app.route('/static/<path:path>')
def serve_static(path):
    """Serve static files"""
    return send_from_directory('static', path)

@app.route('/predict', methods=['POST'])
def predict():
    global prediction_history, last_prediction, last_prediction_time
    user_id = request.sid if hasattr(request, 'sid') else 'default'
    if user_id not in prediction_history:
        prediction_history[user_id] = []
    if user_id not in last_prediction:
        last_prediction[user_id] = None
    if user_id not in last_prediction_time:
        last_prediction_time[user_id] = time.time()

    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    file = request.files.get('frame')
    if not file:
        logger.warning("No frame provided in request")
        return jsonify({"error": "No frame provided"}), 400

    try:
        img_bytes = file.read()
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None or frame.size == 0:
            logger.warning("Received empty or corrupted frame")
            return jsonify({"error": "Invalid frame"}), 400
            
        logger.info(f"Received frame with shape: {frame.shape}")
        
        processed_frame, prediction, confidence = process_frame(frame, model)
        
        current_time = time.time()
        logger.info(f"Prediction: {prediction}, Confidence: {confidence:.2f}")

        if confidence > prediction_threshold and (
            last_prediction[user_id] != prediction or 
            current_time - last_prediction_time[user_id] > cooldown_period
        ):
            if prediction not in ['nothing', 'no_hand', 'error']:
                prediction_history[user_id].append(prediction)
                last_prediction[user_id] = prediction
                last_prediction_time[user_id] = current_time
                
                if len(prediction_history[user_id]) > 10:
                    prediction_history[user_id].pop(0)
                    
                logger.info(f"Added prediction to history for {user_id}. Current history: {prediction_history[user_id]}")
        sentence = form_sentence(prediction_history[user_id])
        logger.info(f"Formed sentence for {user_id}: {sentence}")

        return jsonify({
            "prediction": prediction,
            "confidence": float(confidence),
            "sentence": sentence
        })
        
    except Exception as e:
        logger.error(f"Error processing frame: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"error": f"Processing error: {str(e)}"}), 500


@app.route('/clear_history', methods=['POST'])
def clear_history():
    """Clear the prediction history"""
    user_id = request.sid if hasattr(request, 'sid') else 'default'
    
    if user_id in prediction_history:
        prediction_history[user_id] = []
        logger.info(f"Prediction history cleared for {user_id}")
    
    return jsonify({"status": "success"})

@app.route('/available_signs')
def available_signs():
    """Return list of available signs that can be recognized"""
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500
        
    signs = model.get_classes()
    return jsonify({"signs": signs})

@app.route('/video_call')
def video_call():
    return render_template('video_call.html')

@socketio.on('connect')
def handle_connect():
    logger.info(f'Client connected: {request.sid}')
    emit('connected', {'status': 'connected'})

rooms = {}  

@socketio.on('join')
def handle_join(data):
    room = data['room']
    join_room(room)

    if room not in rooms:
        rooms[room] = []
    
    if request.sid not in rooms[room]:
        rooms[room].append(request.sid)

    logger.info(f"User {request.sid} joined room: {room}, users in room: {len(rooms[room])}")
    
    is_initiator = len(rooms[room]) == 1
    emit('joined_room', {'initiator': is_initiator, 'count': len(rooms[room])}, room=request.sid)
    emit('user_joined', {'count': len(rooms[room])}, room=room, include_self=False)


@socketio.on('leave')
def handle_leave(data):
    room = data.get('room', '')
    if room and room in rooms and request.sid in rooms[room]:
        leave_room(room)
        rooms[room].remove(request.sid)
        logger.info(f"User {request.sid} left room: {room}")
        emit('user_left', {'count': len(rooms[room])}, room=room)
        
        if len(rooms[room]) == 0:
            del rooms[room]
            logger.info(f"Removed empty room: {room}")


@socketio.on('disconnect')
def handle_disconnect():
    sid = request.sid
    logger.info(f"User {sid} disconnected")
    
    rooms_to_check = list(rooms.keys())
    for room_name in rooms_to_check:
        if sid in rooms[room_name]:
            rooms[room_name].remove(sid)
            logger.info(f"Removed user {sid} from room: {room_name}")
            
            emit('user_left', {'count': len(rooms[room_name])}, room=room_name)
            if len(rooms[room_name]) == 0:
                del rooms[room_name]
                logger.info(f"Removed empty room: {room_name}")
    
    if sid in prediction_history:
        del prediction_history[sid]
    if sid in last_prediction:
        del last_prediction[sid]
    if sid in last_prediction_time:
        del last_prediction_time[sid]


@socketio.on('sign_text')
def handle_sign_text(data):
    room = data.get('room', '')
    sentence = data.get('sentence', '')
    
    if not room:
        logger.warning("No room provided for sign_text")
        return
        
    logger.info(f"Sign text in room {room}: {sentence}")
    emit('receive_text', {'sentence': sentence}, room=room, include_self=False)

@socketio.on('offer')
def handle_offer(data):
    logger.info(f"Relaying offer to room: {data.get('room', 'unknown')}")
    emit('offer', {'sdp': data['sdp']}, room=data['room'], include_self=False)

@socketio.on('answer')
def handle_answer(data):
    logger.info(f"Relaying answer to room: {data.get('room', 'unknown')}")
    emit('answer', {'sdp': data['sdp']}, room=data['room'], include_self=False)

@socketio.on('ice-candidate')
def handle_ice_candidate(data):
    logger.info(f"Relaying ICE candidate to room: {data.get('room', 'unknown')}")
    emit('ice-candidate', {'candidate': data['candidate']}, room=data['room'], include_self=False)


if __name__ == '__main__':
    logger.info("Starting the application...")

    try:
        import socket
        hostname = socket.gethostname()
        local_ip = socket.gethostbyname(hostname)
        logger.info(f"Resolved local IP: {local_ip}")
    except Exception as ip_error:
        logger.warning(f"Could not resolve local IP: {ip_error}")

    try:
        import os
        port = int(os.environ.get("PORT", 10000))  # Use Render's assigned port
        logger.info(f"Listening on 0.0.0.0:{port}")
        socketio.run(app, host='0.0.0.0', port=port)

    except Exception as e:
        logger.error(f"Error starting server: {e}")
        logger.error(traceback.format_exc())
