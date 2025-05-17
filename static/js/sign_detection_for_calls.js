let isCapturing = false;
let captureInterval = null;
let predictionHistory = [];
let lastPrediction = null;
let lastPredictionTime = 0;
const predictionThreshold = 0.6;
const cooldownPeriod = 500; // ms

document.addEventListener('DOMContentLoaded', () => {
    const isDeafMode = window.location.search.includes("deaf=true");

    if (isDeafMode) {
        console.log("[SIGN] Deaf mode detected, will start camera + capture");

        // START CAMERA
        if (typeof startLocalVideo === 'function') {
            startLocalVideo();
        } else {
            console.error("[ERROR] startLocalVideo() is not defined or not loaded");
        }

        // INIT SIGN DETECTION WHEN VIDEO STARTS PLAYING
        const localVideo = document.getElementById('localVideo');
        if (localVideo) {
            localVideo.addEventListener('playing', () => {
                setTimeout(() => {
                    startSignDetection(localVideo);
                }, 1000);
            });
        }
    }
});


function startSignDetection(videoElement) {
    if (isCapturing) return;
    isCapturing = true;
    
    console.log("[SIGN] Starting sign language detection");

    const canvas = document.createElement('canvas');
    canvas.width = 320;
    canvas.height = 240;
    const ctx = canvas.getContext('2d');

    captureInterval = setInterval(() => {
        captureAndProcessFrame(videoElement, canvas, ctx);
    }, 200); 
}

function stopSignDetection() {
    if (!isCapturing) return;
    
    clearInterval(captureInterval);
    isCapturing = false;
    console.log("[SIGN] Stopped sign language detection");
}

const clientId = 'client_' + Math.random().toString(36).substring(2, 9);
function captureAndProcessFrame(videoElement, canvas, ctx) {
    if (!videoElement || videoElement.paused || videoElement.ended) {
        return;
    }

    try {
        canvas.width = videoElement.videoWidth || 320;
        canvas.height = videoElement.videoHeight || 240;
        
        ctx.drawImage(videoElement, 0, 0, canvas.width, canvas.height);
        
        canvas.toBlob((blob) => {
            if (!blob) {
                console.error("[ERROR] Failed to create blob from canvas");
                return;
            }
            
            const formData = new FormData();
            formData.append('frame', blob, 'frame.jpg');
            
            console.log("[DEBUG] Sending frame blob size:", blob.size);
            
            fetch('/predict?client_id=' + clientId, {
                method: 'POST',
                body: formData
            })
            .then(response => {
                if (!response.ok) {
                    throw new Error(`Server responded with ${response.status}`);
                }
                return response.json();
            })
            .then(data => {
                console.log("[DEBUG] Received prediction:", data);
                processSignPrediction(data);
            })
            .catch(error => console.error("[ERROR] Failed to get prediction:", error));
        }, 'image/jpeg', 0.8);
    } catch (e) {
        console.error("[ERROR] Frame capture error:", e);
    }
}

function processSignPrediction(data) {
    const { prediction, confidence, sentence } = data;
    const currentTime = Date.now();
    
    console.log(`[SIGN] Prediction: ${prediction}, Confidence: ${confidence}`);
    if (confidence > predictionThreshold && 
        (lastPrediction !== prediction || currentTime - lastPredictionTime > cooldownPeriod)) {

        lastPrediction = prediction;
        lastPredictionTime = currentTime;

        if (prediction !== 'nothing' && prediction !== 'no_hand') {
            predictionHistory.push(prediction);
            if (predictionHistory.length > 10) {
                predictionHistory.shift();
            }

            sendSignText(sentence);
        }
    }
}

function sendSignText(sentence) {
    if (sentence && window.socket && window.roomName) {
 
        console.log(`[SIGN] Sending text: "${sentence}"`);
        window.socket.emit('sign_text', {
            room: window.roomName,
            sentence: sentence
        });

        const sentElement = document.getElementById('sentText');
        if (sentElement) {
            sentElement.textContent = sentence;
        }
    }
}

function clearSignHistory() {
    predictionHistory = [];

    fetch('/clear_history', {
        method: 'POST'
    })
    .then(response => response.json())
    .then(data => console.log("[SIGN] History cleared"))
    .catch(error => console.error("[ERROR] Failed to clear history:", error));

    sendSignText("");
}

function addClearButton() {
    const controlsDiv = document.querySelector('.connection-status');
    if (controlsDiv) {
        const clearBtn = document.createElement('button');
        clearBtn.textContent = "Clear Signs";
        clearBtn.className = "btn small";
        clearBtn.onclick = clearSignHistory;
        controlsDiv.appendChild(clearBtn);
    }
}

window.addEventListener('DOMContentLoaded', () => {
    if (window.location.search.includes("deaf=true")) {
        addClearButton();
    }
});