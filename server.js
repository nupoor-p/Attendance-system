const express = require('express');
const http = require('http');
const path = require('path');

console.log('[Server] Initializing Express app...');
const app = express();

console.log('[Server] Creating HTTP server...');
const server = http.createServer(app);

console.log('[Server] Setting up Socket.IO...');
const { Server: SocketIOServer } = require('socket.io');
const io = new SocketIOServer(server, {
  cors: {
    origin: "*",
    methods: ["GET", "POST"]
  }
});

// Middleware
app.use(express.json());
app.use(express.static(path.join(__dirname)));

// Serve index.html
app.get('/', (req, res) => {
  console.log('[Server] Serving index.html');
  res.sendFile(path.join(__dirname, 'index.html'));
});

// Socket.IO events
io.on('connection', (socket) => {
  console.log('[Server] Client connected:', socket.id);
});

// API endpoints
app.post('/api/enrollment-trigger', (req, res) => {
  console.log('[Server] Enrollment trigger received');
  io.emit('enrollment_needed', req.body);
  res.json({ status: 'ok' });
});

app.post('/api/attendance-log', (req, res) => {
  console.log('[Server] Attendance log received');
  io.emit('attendance_logged', req.body);
  res.json({ status: 'ok' });
});

app.get('/api/status', (req, res) => {
  res.json({ status: 'running', timestamp: new Date().toISOString() });
});

// Start listening
const PORT = 3000;
console.log('[Server] Starting to listen on port', PORT);

server.listen(PORT, () => {
  console.log('[Server] ✓ Server running on http://localhost:' + PORT);
}).on('error', (err) => {
  console.error('[Server] Error:', err.message);
});