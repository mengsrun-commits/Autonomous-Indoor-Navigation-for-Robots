function handleVideoError(image) {
    // Add the error class to the parent container
    const container = document.getElementById('videoContainer');
    container.classList.add('error');
    
    console.error("Video stream failed to load. Switching to placeholder.");
    
    // Optional: Try to reconnect every 5 seconds
    setTimeout(() => {
        image.src = image.src.split('?')[0] + '?t=' + new Date().getTime();
        container.classList.remove('error');
    }, 5000);
}

let connectionTimeout;

// Run automatically when the page loads
window.onload = function() {
    attemptConnection();
};

function attemptConnection() {
    const streamImg = document.getElementById('robotStream');
    const statusText = document.getElementById('statusText');
    const loading = document.getElementById('loadingIndicator');
    const retryBtn = document.getElementById('retryBtn');
    const errorIcon = document.getElementById('errorIcon');

    // 1. Reset UI to Loading State
    statusText.innerText = "Connecting to Robot...";
    loading.style.display = "block";
    retryBtn.style.display = "none";
    errorIcon.style.display = "none";

    // 2. Start the stream request
    // We add a timestamp to prevent the browser from using a broken cached version
    streamImg.src = "/video_feed?t=" + new Date().getTime();

    // 3. Set a safety timeout (5 seconds)
    // If the image doesn't load in 5s, we trigger the error manually
    clearTimeout(connectionTimeout); 
    connectionTimeout = setTimeout(() => {
        if (streamImg.style.display !== "block") {
            handleError();
        }
    }, 5000);
}

function handleSuccess() {
    clearTimeout(connectionTimeout);
    document.getElementById('placeholder').style.display = 'none';
    document.getElementById('robotStream').style.display = 'block';
    document.getElementById('liveBadge').style.display = 'block';
}

function handleError() {
    clearTimeout(connectionTimeout);
    const streamImg = document.getElementById('robotStream');
    streamImg.src = ""; // Stop the browser from trying to load
    streamImg.style.display = "none";

    document.getElementById('statusText').innerText = "Camera Connection Failed";
    document.getElementById('loadingIndicator').style.display = "none";
    document.getElementById('errorIcon').style.display = "block";
    document.getElementById('retryBtn').style.display = "block";
    document.getElementById('liveBadge').style.display = "none";
}
