/* =========================================
   1. CONFIGURATION & STATE
   ========================================= */
const socket = io();
let manualMode = false;
let connectionTimeout;

// DOM Elements
const instructionDiv = document.getElementById('manual-instructions');
const sidebar = document.querySelector('.controls-sidebar');
const manualBtn = document.querySelector('.manual-control');

// Configuration: Maps Keys/IDs to Commands
const CONTROLS = {
    // Keyboard Key -> { command, buttonId }
    'ArrowUp':    { cmd: 'up',    btnId: 'up-arrow' },
    'ArrowDown':  { cmd: 'down',  btnId: 'down-arrow' },
    'ArrowLeft':  { cmd: 'left',  btnId: 'left-arrow' },
    'ArrowRight': { cmd: 'right', btnId: 'right-arrow' }
};

// Reverse Map for HTML Buttons: ButtonID -> Command
// (Generated automatically from CONTROLS to avoid duplicate typing)
const BTN_COMMANDS = {};
Object.values(CONTROLS).forEach(mapping => {
    BTN_COMMANDS[mapping.btnId] = mapping.cmd;
});

/* =========================================
   2. VIDEO STREAM HANDLING
   ========================================= */
window.onload = attemptConnection;

function attemptConnection() {
    const streamImg = document.getElementById('robotStream');
    const els = {
        status: document.getElementById('statusText'),
        loading: document.getElementById('loadingIndicator'),
        retry: document.getElementById('retryBtn'),
        error: document.getElementById('errorIcon')
    };

    // Reset UI
    els.status.innerText = "Connecting to Robot...";
    els.loading.style.display = "block";
    els.retry.style.display = "none";
    els.error.style.display = "none";

    // Load Stream with Timestamp (busting cache)
    streamImg.style.display = "none"; // Hide until loaded
    streamImg.src = "/video_feed?t=" + new Date().getTime();

    // Safety Timeout (5s)
    clearTimeout(connectionTimeout);
    connectionTimeout = setTimeout(() => {
        if (streamImg.style.display === "none") handleError();
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
    streamImg.src = ""; 
    streamImg.style.display = "none";

    document.getElementById('statusText').innerText = "Camera Connection Failed";
    document.getElementById('loadingIndicator').style.display = "none";
    document.getElementById('errorIcon').style.display = "block";
    document.getElementById('retryBtn').style.display = "block";
    document.getElementById('liveBadge').style.display = "none";
}

/* =========================================
   3. MANUAL CONTROL LOGIC
   ========================================= */

// Toggle Manual Mode
/* --- Update in your JS --- */

document.querySelector('.manual-control').addEventListener('click', function() {
    manualMode = !manualMode;
    const btn = this;
    const sidebar = document.querySelector('.controls-sidebar');
    
    // Select the Mode Text Span
    const modeText = document.querySelector('.mode span');
    
    sidebar.classList.toggle('manual-active');
    
    if (manualMode) {
        // --- ON ---
        btn.textContent = "STOP MANUAL CONTROL";
        
        // Update Status Indicator
        modeText.textContent = "Manual";
        modeText.style.color = "var(--accent-red)"; // Turn text Red
        
        window.focus(); 
    } else {
        // --- OFF ---
        btn.textContent = "MANUAL CONTROL";
        
        // Update Status Indicator
        modeText.textContent = "Standby";
        modeText.style.color = "var(--accent-blue)"; // Revert to Blue
        
        // Stop Logic
        socket.emit('stop_manual_control');
        sendCommand('stop'); 
    }
});

/* =========================================
   4. DRIVE HANDLERS (Unified Logic)
   ========================================= */

// Helper to send commands and update UI
function activateDrive(command, btnId) {
    if (!manualMode) return;
    
    // 1. Send Command
    sendCommand(command);
    
    // 2. Visual Feedback
    const btn = document.getElementById(btnId);
    if (btn) btn.classList.add('active');
}

function deactivateDrive(btnId) {
    if (!manualMode) return;

    // 1. Send Stop
    sendCommand('stop');

    // 2. Remove Visual Feedback
    const btn = document.getElementById(btnId);
    if (btn) btn.classList.remove('active');
}

function sendCommand(cmd) {
    socket.emit('control', { command: cmd });
}

/* =========================================
   5. INPUT LISTENERS
   ========================================= */

// --- Keyboard ---
document.addEventListener('keydown', (e) => {
    if (!manualMode || e.repeat) return;
    
    const mapping = CONTROLS[e.key];
    if (mapping) {
        e.preventDefault();
        activateDrive(mapping.cmd, mapping.btnId);
    }
});

document.addEventListener('keyup', (e) => {
    if (!manualMode) return;
    
    const mapping = CONTROLS[e.key];
    if (mapping) {
        e.preventDefault();
        deactivateDrive(mapping.btnId);
    }
});

// --- Mouse / Touch (On-Screen Buttons) ---
// Attach listeners to all buttons defined in CONTROLS
Object.values(CONTROLS).forEach(mapping => {
    const btn = document.getElementById(mapping.btnId);
    if (!btn) return;

    // Start Action
    const startHandler = (e) => {
        e.preventDefault(); // Prevents ghost clicks on touch
        activateDrive(mapping.cmd, mapping.btnId);
    };

    // Stop Action
    const stopHandler = (e) => {
        e.preventDefault();
        deactivateDrive(mapping.btnId);
    };

    // Event Bindings
    btn.addEventListener('mousedown', startHandler);
    btn.addEventListener('touchstart', startHandler);

    btn.addEventListener('mouseup', stopHandler);
    btn.addEventListener('mouseleave', stopHandler);
    btn.addEventListener('touchend', stopHandler);
});


const icons = {
    full: `
        <svg viewBox="0 0 24 24" fill="none" stroke="#4ade80" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <rect x="1" y="6" width="18" height="12" rx="2" ry="2"></rect>
            <line x1="23" y1="13" x2="23" y2="11"></line>
            <path d="M5 10v4" fill="#4ade80" stroke="none"></path> <path d="M9 10v4" fill="#4ade80" stroke="none"></path> <path d="M13 10v4" fill="#4ade80" stroke="none"></path> </svg>`,
    
    medium: `
        <svg viewBox="0 0 24 24" fill="none" stroke="#facc15" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <rect x="1" y="6" width="18" height="12" rx="2" ry="2"></rect>
            <line x1="23" y1="13" x2="23" y2="11"></line>
            <path d="M5 10v4" fill="#facc15" stroke="none"></path> <path d="M9 10v4" fill="#facc15" stroke="none"></path> </svg>`,

    low: `
        <svg viewBox="0 0 24 24" fill="none" stroke="#ef4444" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <rect x="1" y="6" width="18" height="12" rx="2" ry="2"></rect>
            <line x1="23" y1="13" x2="23" y2="11"></line>
            <path d="M5 10v4" fill="#ef4444" stroke="none"></path> </svg>`
};

function updateBatteryOverlay(percentage) {
    const container = document.getElementById('battery-indicator');
    
    // Safety check: if camera isn't loaded, container might be null
    if (!container) return; 

    if (percentage > 70) {
        container.innerHTML = icons.full;
    } else if (percentage > 40) {
        container.innerHTML = icons.medium;
    } else {
        container.innerHTML = icons.low;
        
        // Optional: Add a blink effect for critical battery
        if (percentage < 20) {
            container.style.animation = "blink 1s infinite";
        } else {
            container.style.animation = "none";
        }
    }
}
