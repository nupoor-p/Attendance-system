#!/usr/bin/env node
// Minimal FaceAttend Pro UI Server using only Node.js built-in modules

const http = require('http');
const fs = require('fs');
const path = require('path');
const url = require('url');

console.log('[Server] Starting minimal server...');

const PORT = 3000;

// Read HTML file
let htmlContent;
try {
  const htmlPath = path.join(__dirname, 'index.html');
  console.log('[Server] Looking for index.html at:', htmlPath);
  htmlContent = fs.readFileSync(htmlPath, 'utf8');
  console.log('[Server] ✓ HTML file loaded, size:', htmlContent.length, 'bytes');
} catch (e) {
  console.error('[Server] ✗ Error reading index.html:', e.message);
  process.exit(1);
}

// Create server
console.log('[Server] Creating HTTP server...');
const server = http.createServer((req, res) => {
  const parsedUrl = url.parse(req.url, true);
  const pathname = parsedUrl.pathname;

  console.log(`[Server] ${req.method} ${pathname}`);

  // Handle CORS
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

  if (req.method === 'OPTIONS') {
    res.writeHead(200);
    res.end();
    return;
  }

  // Handle routes
  if (pathname === '/' && req.method === 'GET') {
    res.writeHead(200, { 'Content-Type': 'text/html; charset=utf-8' });
    res.end(htmlContent);
  } 
  else if (pathname === '/api/status' && req.method === 'GET') {
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ status: 'running', timestamp: new Date().toISOString() }));
  }
  else if (pathname === '/api/enrollment-trigger' && req.method === 'POST') {
    console.log('[Server] Enrollment trigger received');
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ status: 'ok' }));
  }
  else if (pathname === '/api/attendance-log' && req.method === 'POST') {
    console.log('[Server] Attendance log received');
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ status: 'ok' }));
  }
  else {
    res.writeHead(404, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ error: 'Not found' }));
  }
});

console.log('[Server] Attempting to listen on port', PORT);

server.listen(PORT, 'localhost', () => {
  console.log('\n[Server] ✓✓✓ Server running on http://localhost:' + PORT + ' ✓✓✓');
  console.log('[Server] Open your browser and navigate to http://localhost:' + PORT);
  console.log('[Server] Press Ctrl+C to stop\n');
}).on('error', (err) => {
  console.error('[Server] ✗ Error:', err.message);
  if (err.code === 'EADDRINUSE') {
    console.error('[Server] Port ' + PORT + ' is already in use');
  }
  process.exit(1);
});

// Handle shutdown
process.on('SIGINT', () => {
  console.log('\n[Server] Shutting down...');
  server.close();
  process.exit(0);
});
