let video = document.getElementById('video');
let canvas = document.getElementById('canvas');
let context = canvas.getContext('2d');
let stream = null;

// Tab switching
function showTab(tabName) {
    // Hide all tabs
    document.querySelectorAll('.tab-content').forEach(tab => {
        tab.classList.remove('active');
    });
    document.querySelectorAll('.tab-button').forEach(btn => {
        btn.classList.remove('active');
    });

    // Show selected tab
    document.getElementById(tabName + '-tab').classList.add('active');
    event.target.classList.add('active');
}

// Camera functions
async function startCamera() {
    try {
        stream = await navigator.mediaDevices.getUserMedia({
            video: { width: 640, height: 480 },
            audio: false
        });

        video.srcObject = stream;
        document.getElementById('start-camera').disabled = true;
        document.getElementById('capture').disabled = false;
        document.getElementById('stop-camera').disabled = false;
        document.getElementById('status').textContent = 'Camera started. Click "Capture & Process" to take attendance.';
    } catch (error) {
        console.error('Error accessing camera:', error);
        document.getElementById('status').textContent = 'Error: Could not access camera. Please check permissions.';
    }
}

function stopCamera() {
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
        video.srcObject = null;
        stream = null;
    }
    document.getElementById('start-camera').disabled = false;
    document.getElementById('capture').disabled = true;
    document.getElementById('stop-camera').disabled = true;
    document.getElementById('status').textContent = 'Camera stopped.';
}

function captureAndProcess() {
    if (!stream) return;

    // Set canvas size to video size
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    // Draw current video frame to canvas
    context.drawImage(video, 0, 0, canvas.width, canvas.height);

    // Convert to base64
    const imageData = canvas.toDataURL('image/jpeg', 0.8);

    // Show processing status
    document.getElementById('status').textContent = 'Processing image...';

    // Send to backend
    fetch('http://127.0.0.1:5000/api/process-image', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ image: imageData })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            let message = data.message;
            if (data.marked && data.marked.length > 0) {
                message += ` ✅ Attendance marked for: ${data.marked.join(', ')}`;
            }
            document.getElementById('status').textContent = message;
        } else {
            document.getElementById('status').textContent = 'Error: ' + data.error;
        }
    })
    .catch(error => {
        console.error('Error:', error);
        document.getElementById('status').textContent = 'Error processing image. Please try again.';
    });
}

// Manual attendance
function markManualAttendance() {
    const name = document.getElementById('name-input').value.trim();
    if (!name) {
        document.getElementById('manual-status').textContent = 'Please enter a name.';
        return;
    }

    document.getElementById('manual-status').textContent = 'Marking attendance...';

    fetch('http://127.0.0.1:5000/api/mark-attendance', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ name: name, method: 'manual' })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            document.getElementById('manual-status').textContent = data.message;
            document.getElementById('name-input').value = '';
        } else {
            document.getElementById('manual-status').textContent = 'Error: ' + data.error;
        }
    })
    .catch(error => {
        console.error('Error:', error);
        document.getElementById('manual-status').textContent = 'Error marking attendance. Please try again.';
    });
}

// View records
function loadRecords() {
    document.getElementById('records-list').innerHTML = 'Loading records...';

    fetch('http://127.0.0.1:5000/api/attendance')
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            if (data.attendance.length === 0) {
                document.getElementById('records-list').innerHTML = '<p>No attendance records found.</p>';
                return;
            }

            let html = '<table class="records-table">';
            html += '<thead><tr><th>Name</th><th>Timestamp</th><th>Method</th></tr></thead><tbody>';

            data.attendance.forEach(record => {
                html += `<tr>
                    <td>${record.name}</td>
                    <td>${record.timestamp}</td>
                    <td>${record.method}</td>
                </tr>`;
            });

            html += '</tbody></table>';
            document.getElementById('records-list').innerHTML = html;
        } else {
            document.getElementById('records-list').innerHTML = 'Error loading records: ' + data.error;
        }
    })
    .catch(error => {
        console.error('Error:', error);
        document.getElementById('records-list').innerHTML = 'Error loading records. Please try again.';
    });
}

// Event listeners
document.getElementById('start-camera').addEventListener('click', startCamera);
document.getElementById('capture').addEventListener('click', captureAndProcess);
document.getElementById('stop-camera').addEventListener('click', stopCamera);
document.getElementById('mark-manual').addEventListener('click', markManualAttendance);
document.getElementById('refresh-records').addEventListener('click', loadRecords);

// Load records on page load
document.addEventListener('DOMContentLoaded', function() {
    loadRecords();
});
