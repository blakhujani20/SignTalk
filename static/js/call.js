let localStream;
let peerConnection;
let roomName = "";
let isInitiator = false;
let iceCandidatesQueue = []; 

const config = {
  iceServers: [
    { urls: "stun:stun.l.google.com:19302" },
    { urls: "stun:stun1.l.google.com:19302" },
    {
      urls: "turn:relay.metered.ca:80",
      username: "openai",
      credential: "openai",
    },
    {
      urls: "turn:relay.metered.ca:443",
      username: "openai",
      credential: "openai",
    },
  ],
};

const localVideo = document.getElementById("localVideo");
const remoteVideo = document.getElementById("remoteVideo");
const roomInput = document.getElementById("roomInput");
const joinButton = document.getElementById("joinButton");
const receivedTextElement = document.getElementById("receivedText");

const isProduction = window.location.protocol === "https:";
window.socket = io({
  transports: ["websocket", "polling"],
  path: "/socket.io",
  secure: isProduction,
  reconnectionAttempts: 5,
  reconnectionDelay: 1000
});


window.socket.on("connect", () => console.log("[STATUS] Socket connected"));
window.socket.on("disconnect", () => console.log("[STATUS] Socket disconnected"));

window.socket.on("offer", async ({ sdp }) => {
  console.log("[SIGNAL] Received offer");
  if (!peerConnection) await setupPeerConnection();
  
  try {
    await peerConnection.setRemoteDescription(new RTCSessionDescription(sdp));
    console.log("[SIGNAL] Remote description set successfully");

    processIceCandidateQueue();
    
    const answer = await peerConnection.createAnswer();
    await peerConnection.setLocalDescription(answer);
    window.socket.emit("answer", { sdp: answer, room: roomName });
    console.log("[SIGNAL] Sent answer");
  } catch (error) {
    console.error("[ERROR] Failed to process offer:", error);
  }
});

window.socket.on("answer", async ({ sdp }) => {
  console.log("[SIGNAL] Received answer");
  try {
    await peerConnection.setRemoteDescription(new RTCSessionDescription(sdp));
    console.log("[SIGNAL] Remote description set successfully from answer");

    processIceCandidateQueue();
  } catch (error) {
    console.error("[ERROR] Failed to process answer:", error);
  }
});

window.socket.on("ice-candidate", ({ candidate }) => {
  console.log("[SIGNAL] Received ICE candidate");
  
  if (peerConnection && peerConnection.remoteDescription && peerConnection.remoteDescription.type) {
    addIceCandidate(candidate);
  } else {
    console.log("[SIGNAL] Queuing ICE candidate until remote description is set");
    iceCandidatesQueue.push(candidate);
  }
});

function processIceCandidateQueue() {
  console.log(`[SIGNAL] Processing ${iceCandidatesQueue.length} queued ICE candidates`);
  
  if (iceCandidatesQueue.length > 0) {
    iceCandidatesQueue.forEach(candidate => {
      addIceCandidate(candidate);
    });
    iceCandidatesQueue = []; 
  }
}

function addIceCandidate(candidate) {
  peerConnection.addIceCandidate(new RTCIceCandidate(candidate))
    .then(() => console.log("[SIGNAL] ICE candidate added successfully"))
    .catch(err => console.warn("[ERROR] Failed to add ICE candidate:", err));
}

window.socket.on("joined_room", async ({ initiator, count }) => {
  isInitiator = initiator;
  console.log(`[STATUS] Joined room: ${roomName} as ${isInitiator ? "initiator" : "peer"} (${count} users)`);

  const started = await startCamera();
  if (started) {
    await setupPeerConnection();
    if (isInitiator && count >= 2) {
      console.log("[STATUS] Creating offer as initiator with existing users");
      createAndSendOffer();
    }
  }
});

window.socket.on("user_joined", async ({ count }) => {
  console.log(`[STATUS] A user joined the room. Total users: ${count}`);
  if (isInitiator && count >= 2 && peerConnection) {
    console.log("[STATUS] Second user joined — sending offer...");
    createAndSendOffer();
  }
});

async function createAndSendOffer() {
  try {
    const offer = await peerConnection.createOffer();
    await peerConnection.setLocalDescription(offer);
    window.socket.emit("offer", { sdp: offer, room: roomName });
    console.log("[SIGNAL] Sent offer");
  } catch (error) {
    console.error("[ERROR] Failed to create or send offer:", error);
  }
}

window.socket.on("user_left", ({ count }) => {
  console.log(`[STATUS] A user left the room. Remaining users: ${count}`);
  if (count < 2) {
    if (remoteVideo) {
      remoteVideo.srcObject = null;
    }
    
    if (receivedTextElement) {
      receivedTextElement.textContent = "Waiting for partner to join...";
    }
  }
});

window.socket.on("receive_text", ({ sentence }) => {
  if (receivedTextElement) {
    if (sentence !== undefined) {
      receivedTextElement.textContent = sentence || "Waiting for signs...";
      receivedTextElement.classList.add('new-text');
      setTimeout(() => {
        receivedTextElement.classList.remove('new-text');
      }, 500);
    }
  }
});

window.roomName = "";

function joinRoom() {
  window.roomName = roomInput.value.trim();
  if (!window.roomName) return alert("Please enter a room name to join");
  window.socket.emit("join", { room: window.roomName });
}

async function setupPeerConnection() {
  if (peerConnection) {
    peerConnection.close();
  }
  iceCandidatesQueue = [];

  peerConnection = new RTCPeerConnection(config);
  console.log("[STATUS] Peer connection created");

  peerConnection.onicecandidate = (event) => {
    if (event.candidate) {
      window.socket.emit("ice-candidate", { candidate: event.candidate, room: roomName });
      console.log("[SIGNAL] Sent ICE candidate");
    }
  };

  peerConnection.ontrack = (event) => {
    console.log("[MEDIA] Remote track received", event.streams[0]);

    if (remoteVideo) {
      remoteVideo.srcObject = event.streams[0];
      console.log("[MEDIA] Set remote video source");
      remoteVideo.muted = false;
      
      remoteVideo.onloadedmetadata = () => {
        remoteVideo.play()
          .then(() => console.log("[MEDIA] Remote video playing"))
          .catch(err => {
            console.warn("[ERROR] Remote video play failed:", err);
            if (err.name === "NotAllowedError") {
              const playBtn = document.createElement("button");
              playBtn.textContent = "Play Remote Video";
              playBtn.className = "btn";
              playBtn.onclick = () => {
                remoteVideo.play()
                  .then(() => console.log("[MEDIA] Remote video playing after user interaction"))
                  .catch(e => console.error("[ERROR] Failed to play even after interaction:", e));
                playBtn.remove();
              };
              remoteVideo.parentNode.insertBefore(playBtn, remoteVideo.nextSibling);
            }
          });
      };
    } else {
      console.warn("[ERROR] remoteVideo element not found in DOM");
    }
  };

  peerConnection.onconnectionstatechange = () => {
    console.log(`[STATUS] Connection state changed: ${peerConnection.connectionState}`);
    if (peerConnection.connectionState === "connected") {
      console.log("[STATUS] Peers successfully connected!");
    } else if (peerConnection.connectionState === "failed") {
      console.error("[ERROR] Connection failed. Attempting to restart ICE...");
      if (isInitiator) {
        createAndSendOffer();
      }
    }
  };

  peerConnection.oniceconnectionstatechange = () => {
    console.log(`[STATUS] ICE state: ${peerConnection.iceConnectionState}`);
    if (peerConnection.iceConnectionState === "failed" && isInitiator) {
      console.warn("[STATUS] ICE failed, attempting to restart...");
      createAndSendOffer();
    }
  };
  if (localStream) {
    localStream.getTracks().forEach((track) => {
      console.log(`[MEDIA] Adding local track to peer connection: ${track.kind}`);
      peerConnection.addTrack(track, localStream);
    });
  } else {
    console.warn("[WARN] No local stream available when setting up peer connection");
  }
  
  return peerConnection; 
}

async function startCamera() {
  try {
    console.log("[MEDIA] Requesting camera and microphone access...");
    localStream = await navigator.mediaDevices.getUserMedia({ 
      video: {
        width: { ideal: 640 },
        height: { ideal: 480 },
        frameRate: { ideal: 30 }
      }, 
      audio: true 
    });
    if (localVideo) {
      localVideo.srcObject = localStream;
      localVideo.muted = true;
      await localVideo.play().catch(err => console.warn("[ERROR] Local video play failed:", err));
    }
    
    console.log("[MEDIA] Local stream started successfully");
    return true;
  } catch (err) {
    console.error("[ERROR] Camera/mic access denied:", err);
    alert("Cannot access your camera/microphone. Please check your permissions and try again.");
    return false;
  }
}

window.addEventListener("beforeunload", () => {
  if (roomName) {
    window.socket.emit("leave", { room: roomName });
  }
  
  if (localStream) {
    localStream.getTracks().forEach(track => track.stop());
  }
  
  if (peerConnection) {
    peerConnection.close();
  }
});

document.addEventListener("DOMContentLoaded", () => {
  if (joinButton) joinButton.addEventListener("click", joinRoom);
  const isDeafMode = window.location.search.includes("deaf=true");
  if (isDeafMode && receivedTextElement) {
    receivedTextElement.textContent = "Sign detection active. Waiting for signs...";
  } else if (receivedTextElement) {
    receivedTextElement.textContent = "Waiting for partner to make signs...";
  }
});
window.startLocalVideo = startCamera;
