// /src/main.js
const { app, BrowserWindow } = require('electron');
const path = require('node:path');
const isDev = require('electron-is-dev');

const createWindow = () => {
  const win = new BrowserWindow({
    width: 1600,
    height: 900,
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
    },
  });

  const indexPath = isDev
    ? path.join(__dirname, 'index.html') // Path in development
    : path.join(process.resourcesPath, 'app', 'src', 'index.html'); // Path in production

  win.loadFile(indexPath);
  win.webContents.openDevTools();
};

app.on('ready', createWindow);
