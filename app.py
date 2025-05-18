from flask import Flask, render_template, Response, jsonify, request, send_from_directory
import cv2
import numpy as np
import time
import os
import logging
import traceback
from logging.handlers import RotatingFileHandler
from flask_socketio import SocketIO, emit, join_room, leave_room
from flask_cors import CORS
from models.sign_language_model import SignLanguageModel
from utils.video_feed import generate_frames, process_frame
from utils.postprocessing import form_sentence

if not os.path.exists('logs'):
    os.mkdir('logs')

logger = logging.getLogger('signtalk')
logger.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter(
    '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
))
logger.addHandler(console_handler)

file_handler = RotatingFileHandler('logs/signtalk.log', maxBytes=10240, backupCount=10)
file_handler.setFormatter(logging.Formatter(
    '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
))
logger.addHandler(file_handler)
logger.info('SignTalk startup')

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'default-dev-key')
CORS(app)

socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    async_mode='gevent',
    ping_timeout=120,
    ping_interval=25,
    ssl_context=None,
    max_http_buffer_size=5e6
)

model = None
def get_model():
    global model
    if model is None:
        try:
            logger.info("Lazy loading sign language model...")
            model = SignLanguageModel()
            logger.info(f"Model loaded successfully. Input shape: {model.model.input_shape}")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            logger.error(traceback.format_exc())
    return model

prediction_history = {}
last_prediction = {}
last_prediction_time = {}
prediction_threshold = 0.6
cooldown_period = 0.5

@app.errorhandler(Exception)
def handle_error(e):
    logger.error(f"Unhandled exception: {e}")
    logger.error(traceback.format_exc())
    return jsonify({"error": "Internal server error"}), 500

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/health')
def health():
    return "OK", 200

@app.route('/status')
def status():
    return jsonify({"status": "ok", "model_loaded": model is not None})

@app.route('/static/<path:path>')
def serve_static(path):
    return send_from_directory('static', path)

@app.route('/video_feed')
def video_feed():
    response = Response(
        generate_frames(get_model()),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

@app.route('/predict', methods=['POST'])
def predict():
    model = get_model()
    if model is None:
        return jsonify({"error": "Model could not be loaded"}), 500
    try:
        client_id = request.args.get('client_id') or request.remote_addr
        user_id = client_id

        prediction_history.setdefault(user_id, [])
        last_prediction.setdefault(user_id, None)
        last_prediction_time.setdefault(user_id, time.time())

        file = request.files.get('frame')
        if not file:
            return jsonify({"error": "No frame provided"}), 400

        img_bytes = file.read()
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None or frame.size == 0:
            return jsonify({"error": "Invalid frame"}), 400

        processed_frame, prediction, confidence = process_frame(frame, model)

        current_time = time.time()
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

        try:
            sentence = form_sentence(prediction_history[user_id])
        except Exception:
            logger.warning("Fallback: using raw prediction history for sentence")
            sentence = " ".join(prediction_history[user_id])

        return jsonify({
            "prediction": prediction,
            "confidence": float(confidence),
            "sentence": sentence
        })

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"error": "Server error"}), 500

@app.route('/clear_history', methods=['POST'])
def clear_history():
    user_id = request.sid if hasattr(request, 'sid') else 'default'
    prediction_history[user_id] = []
    return jsonify({"status": "success"})

@app.route('/available_signs')
def available_signs():
    model = get_model()
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500
    return jsonify({"signs": model.get_classes()})

@app.route('/video_call')
def video_call():
    return render_template('video_call.html')

# SocketIO events
rooms = {}

@socketio.on('connect')
def handle_connect():
    logger.info(f'Client connected: {request.sid}')
    emit('connected', {'status': 'connected'})

@socketio.on('join')
def handle_join(data):
    room = data['room']
    join_room(room)
    rooms.setdefault(room, [])
    if request.sid not in rooms[room]:
        rooms[room].append(request.sid)
    emit('joined_room', {
        'initiator': len(rooms[room]) == 1,
        'count': len(rooms[room])
    }, room=request.sid)
    emit('user_joined', {'count': len(rooms[room])}, room=room, include_self=False)

@socketio.on('leave')
def handle_leave(data):
    room = data.get('room', '')
    if room and request.sid in rooms.get(room, []):
        leave_room(room)
        rooms[room].remove(request.sid)
        emit('user_left', {'count': len(rooms[room])}, room=room)
        if not rooms[room]:
            del rooms[room]

@socketio.on('disconnect')
def handle_disconnect():
    sid = request.sid
    logger.info(f"User {sid} disconnected")
    for room_name in list(rooms.keys()):
        if sid in rooms[room_name]:
            rooms[room_name].remove(sid)
            emit('user_left', {'count': len(rooms[room_name])}, room=room_name)
            if not rooms[room_name]:
                del rooms[room_name]
    prediction_history.pop(sid, None)
    last_prediction.pop(sid, None)
    last_prediction_time.pop(sid, None)

@socketio.on('sign_text')
def handle_sign_text(data):
    room = data.get('room', '')
    sentence = data.get('sentence', '')
    if room:
        emit('receive_text', {'sentence': sentence}, room=room, include_self=False)

@socketio.on('offer')
def handle_offer(data):
    emit('offer', {'sdp': data['sdp']}, room=data['room'], include_self=False)

@socketio.on('answer')
def handle_answer(data):
    emit('answer', {'sdp': data['sdp']}, room=data['room'], include_self=False)

@socketio.on('ice-candidate')
def handle_ice_candidate(data):
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
        port = int(os.environ.get("PORT", 10000))
        logger.info(f"Listening on 0.0.0.0:{port}")
        socketio.run(app, host='0.0.0.0', port=port, debug=False, use_reloader=False)
    except Exception as e:
        logger.error(f"Error starting server: {e}")
        logger.error(traceback.format_exc())
