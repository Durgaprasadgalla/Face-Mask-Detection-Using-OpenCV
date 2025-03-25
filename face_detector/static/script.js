document.getElementById('startCameraBtn').addEventListener('click', function() {
    const videoFeed = document.getElementById('videoFeed');
    const videoContainer = document.getElementById('videoContainer');
    const startBtn = document.getElementById('startCameraBtn');
    const stopBtn = document.getElementById('stopCameraBtn');

    videoFeed.src = '/video_feed';  // Start the video stream
    videoContainer.style.display = 'block';
    startBtn.style.display = 'none';
    stopBtn.style.display = 'inline-block';
});

document.getElementById('stopCameraBtn').addEventListener('click', function() {
    const videoFeed = document.getElementById('videoFeed');
    const videoContainer = document.getElementById('videoContainer');
    const startBtn = document.getElementById('startCameraBtn');
    const stopBtn = document.getElementById('stopCameraBtn');

    videoFeed.src = '';  // Stop the stream
    videoContainer.style.display = 'none';
    startBtn.style.display = 'inline-block';
    stopBtn.style.display = 'none';
});