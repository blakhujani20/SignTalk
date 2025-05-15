document.addEventListener('DOMContentLoaded', function() {
    if (document.getElementById('videoElement')) {
        initializeInterpreter();
    }
});

window.socket = io.connect('http://' + document.domain + ':' + location.port);
window.socket.on("receive_text", function(data) {
    alert("Received from other user: " + data.sentence);
});

function initializeInterpreter() {
    const videoElement = document.getElementById('videoElement');
    const startButton = document.getElementById('start-btn');
    const pauseButton = document.getElementById('pause-btn');
    const clearButton = document.getElementById('clear-btn');
    const currentSignElement = document.getElementById('current-sign');
    const confidenceElement = document.getElementById('confidence');
    const interpretedSentenceElement = document.getElementById('interpreted-sentence');
    const historyListElement = document.getElementById('history-list');
    const sendTextButton = document.getElementById('send-text-btn');
    const textInput = document.getElementById('text-input');

    const state = {
        isStreaming: true,
        lastPrediction: null,
        predictionHistory: [],
        isCapturing: false
    };

    if (startButton) {
        startButton.addEventListener('click', startVideoStream);
    }

    if (pauseButton) {
        pauseButton.addEventListener('click', pauseVideoStream);
    }

    if (clearButton) {
        clearButton.addEventListener('click', clearHistory);
    }

    if (sendTextButton) {
        sendTextButton.addEventListener('click', sendTextMessage);
    }

    if (!state.isCapturing) {
        state.isCapturing = true;
        captureFrames();
    }

    function startVideoStream() {
        if (!state.isStreaming) {
            state.isStreaming = true;
            videoElement.src = "/video_feed";
            if (startButton) startButton.disabled = true;
            if (pauseButton) pauseButton.disabled = false;
            if (currentSignElement) currentSignElement.textContent = "Ready";
            if (confidenceElement) confidenceElement.textContent = "Confidence: 0.00";
        }
    }

    function pauseVideoStream() {
        if (state.isStreaming) {
            state.isStreaming = false;
            videoElement.src = "";
            if (startButton) startButton.disabled = false;
            if (pauseButton) pauseButton.disabled = true;
            if (currentSignElement) currentSignElement.textContent = "Paused";
        }
    }

    function clearHistory() {
        fetch('/clear_history', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' }
        })
        .then(response => response.json())
        .then(data => {
            state.predictionHistory = [];
            if (interpretedSentenceElement) interpretedSentenceElement.textContent = "Your signed message will appear here.";
            if (historyListElement) historyListElement.innerHTML = '';
            if (currentSignElement) currentSignElement.textContent = "Ready";
        })
        .catch(error => console.error('Error clearing history:', error));
    }

    function sendTextMessage() {
        if (!textInput) return;
        const message = textInput.value.trim();
        if (!message) return;
        alert(`Message sent: "${message}"`);
        textInput.value = '';
    }

    function captureFrames() {
        if (!state.isStreaming || !videoElement) {
            setTimeout(captureFrames, 500);
            return;
        }

        try {
            getPrediction();
            setTimeout(captureFrames, 200);
        } catch (error) {
            console.error('Error in frame capture:', error);
            setTimeout(captureFrames, 1000);
        }
    }

    function getPrediction() {
        if (Math.random() > 0.9) {
            fetch('/predict', {
                method: 'POST',
                body: new FormData()
            })
            .then(response => response.json())
            .then(data => {
                if (data.prediction && data.prediction !== 'no_hand' && data.confidence > 0.5) {
                    updatePredictionDisplay(data.prediction, data.confidence);
                    if (data.sentence) updateSentenceDisplay(data.sentence);
                }
            })
            .catch(error => console.error('Error getting prediction:', error));
        }
    }

    window.socket = io.connect(window.location.origin);
    window.socket.on("receive_text", function(data) {
        alert("Received from other user: " + data.sentence);
    });

    function updatePredictionDisplay(prediction, confidence) {
        if (currentSignElement) currentSignElement.textContent = prediction;
        if (confidenceElement) confidenceElement.textContent = `Confidence: ${confidence.toFixed(2)}`;
        updateHistory(prediction, confidence);
    }

    function updateSentenceDisplay(sentence) {
        if (interpretedSentenceElement) interpretedSentenceElement.textContent = sentence;
    }

    function updateHistory(sign, confidence) {
        if (!historyListElement) return;
        if (sign === 'nothing' || sign === 'no_hand' || confidence < 0.6) return;

        const historyItem = document.createElement('li');
        historyItem.className = 'list-group-item d-flex justify-content-between align-items-center';
        historyItem.textContent = sign;

        const confidenceBadge = document.createElement('span');
        confidenceBadge.className = getConfidenceBadgeClass(confidence);
        confidenceBadge.textContent = confidence.toFixed(2);
        historyItem.appendChild(confidenceBadge);

        historyListElement.prepend(historyItem);
        if (historyListElement.children.length > 10) {
            historyListElement.removeChild(historyListElement.lastChild);
        }
    }

    function getConfidenceBadgeClass(confidence) {
        if (confidence > 0.85) return 'badge bg-success rounded-pill';
        else if (confidence > 0.7) return 'badge bg-info rounded-pill';
        else return 'badge bg-warning rounded-pill';
    }
}

function formatTime(date) {
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
}

let demoStream = null;
let demoInterval = null;

async function toggleCamera() {
    const video = document.getElementById('video');
    if (!video) return;

    if (!demoStream) {
        try {
            demoStream = await navigator.mediaDevices.getUserMedia({ video: true });
            video.srcObject = demoStream;
            await video.play();
            startDemoPredictionLoop();
        } catch (err) {
            console.error("[ERROR] Camera access failed:", err);
            alert("Camera access is required for demo to work.");
        }
    } else {
        const tracks = demoStream.getTracks();
        tracks.forEach(track => track.stop());
        video.srcObject = null;
        demoStream = null;
        stopDemoPredictionLoop();
        document.getElementById('outputText').textContent = 'Waiting for input...';
        document.getElementById('historyBox').textContent = '';
    }
}

function startDemoPredictionLoop() {
    if (!demoInterval) {
        demoInterval = setInterval(captureAndSendDemoFrame, 500);
    }
}

function stopDemoPredictionLoop() {
    if (demoInterval) {
        clearInterval(demoInterval);
        demoInterval = null;
    }
}

async function captureAndSendDemoFrame() {
    const video = document.getElementById('video');
    if (!video || video.paused || video.ended) return;

    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth || 640;
    canvas.height = video.videoHeight || 480;

    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    const blob = await new Promise(resolve => canvas.toBlob(resolve, 'image/jpeg', 0.8));
    if (!blob) return;

    const formData = new FormData();
    formData.append('frame', blob, 'frame.jpg');

    try {
        const res = await fetch('/predict', { method: 'POST', body: formData });
        const data = await res.json();

        if (data.prediction && data.prediction !== 'no_hand') {
            document.getElementById('outputText').textContent = data.prediction;
        }
        if (data.sentence) {
            document.getElementById('historyBox').textContent = data.sentence;
        }
    } catch (error) {
        console.error("Error fetching prediction:", error);
    }
}
