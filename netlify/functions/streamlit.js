const { spawn } = require('child_process');
const serverless = require('serverless-http');
const express = require('express');
const path = require('path');
const app = express();

app.use(express.static(path.join(__dirname, '../../')));

app.all('*', (req, res) => {
  // Start the Streamlit process
  const streamlit = spawn('streamlit', ['run', path.join(__dirname, '../../app.py'), '--server.port', '8501']);
  
  // Forward the process stdout to response
  streamlit.stdout.on('data', (data) => {
    console.log(`stdout: ${data}`);
  });
  
  // Handle errors
  streamlit.stderr.on('data', (data) => {
    console.error(`stderr: ${data}`);
  });
  
  // Handle process exit
  streamlit.on('close', (code) => {
    console.log(`child process exited with code ${code}`);
    res.end();
  });
  
  // Proxy the request to the Streamlit server
  // Note: In a real implementation, you'd need to properly proxy all requests
  res.write(`<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <meta http-equiv="refresh" content="0;url=http://localhost:8501">
  <title>Loading Forex Intelligence...</title>
</head>
<body>
  <p>Loading Forex Intelligence...</p>
</body>
</html>`);
  res.end();
});

// Export the serverless function
module.exports.handler = serverless(app);