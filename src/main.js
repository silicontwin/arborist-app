// /src/main.js
const { app, BrowserWindow } = require('electron');
const path = require('node:path');
const isDev = require('electron-is-dev');
const { execFile } = require('child_process');

const createWindow = () => {
  const win = new BrowserWindow({
    width: 1600,
    height: 900,
    // webPreferences: {
    //   preload: path.join(__dirname, 'preload.js'),
    // },
  });

  const indexPath = isDev
    ? path.join(__dirname, 'index.html') // Path in development
    : path.join(process.resourcesPath, 'app', 'src', 'index.html'); // Path in production

  win.loadFile(indexPath);
  // win.webContents.openDevTools();

  // Start the FastAPI server
  const apiPath = isDev
    ? path.join(__dirname, 'resources', 'api') // Path in development
    : path.join(process.resourcesPath, 'app', 'src', 'resources', 'api'); // Path in production

  console.log('Starting FastAPI server...');

  execFile(apiPath, (error, stdout, stderr) => {
    if (error) {
      console.error('Error starting FastAPI server:', error);
    }
    console.log(`stdout: ${stdout}`);
    console.error(`stderr: ${stderr}`);
  });

  console.log('FastAPI server should be running...');
};

app.on('ready', createWindow);
