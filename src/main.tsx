// /src/main.tsx

import { app, BrowserWindow, ipcMain } from 'electron';
import axios from 'axios';
import path from 'node:path';
import isDev from 'electron-is-dev';
import { execFile, ChildProcess } from 'child_process';

let serverProcess: ChildProcess | string | null = null;
let mainWindow: BrowserWindow | null = null;
let appInitialized = false;

// Function to check if the server is ready
const isServerReady = async (
  url: string,
  retries: number = 5,
  delay: number = 1000,
): Promise<boolean> => {
  await new Promise((resolve) => setTimeout(resolve, 2000)); // Initial delay
  console.log(`Checking if server is ready at ${url}`);

  for (let i = 0; i < retries; i++) {
    try {
      await axios.get(url);
      console.log('Server is ready!');
      return true; // Server responded successfully
    } catch (error) {
      // Type guard to check if error is an instance of Error
      if (error instanceof Error) {
        console.log(
          `Attempt ${i + 1}: Server not ready, retrying in ${delay}ms...`,
          error.message,
        );
      } else {
        // Handle cases where error is not an Error instance
        console.log(
          `Attempt ${i + 1}: Server not ready, retrying in ${delay}ms...`,
        );
      }
      await new Promise((resolve) => setTimeout(resolve, delay)); // Wait before the next retry
    }
  }
  console.log('Server not ready after retries.');
  return false; // Server not ready after retries
};

const startServer = (): void => {
  if (serverProcess !== null) {
    console.log('Server already started or starting');
    return;
  }

  const apiPath = isDev
    ? path.join(__dirname, '../src/resources', 'api')
    : path.join(process.resourcesPath, 'app', 'src', 'resources', 'api');

  console.log('Starting FastAPI server...');

  serverProcess = 'starting';

  serverProcess = execFile(
    apiPath,
    (error: Error | null, stdout: string, stderr: string) => {
      if (error) {
        console.error('Error starting FastAPI server:', error);
        serverProcess = null;
        return;
      }
      console.log(`stdout: ${stdout}`);
      console.error(`stderr: ${stderr}`);
    },
  );

  console.log('FastAPI server should be running...');
};

const createWindow = (): void => {
  mainWindow = new BrowserWindow({
    width: 1600,
    height: 900,
    frame: false,
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
      preload: path.join(__dirname, 'preload.js'),
    },
  });

  const indexPath = isDev
    ? path.join(__dirname, '../index.html')
    : path.join(process.resourcesPath, 'app', 'index.html');

  mainWindow.loadFile(indexPath);
  // mainWindow.webContents.openDevTools();

  mainWindow.on('closed', () => {
    mainWindow = null;
  });
};

ipcMain.handle('fetch-data', async () => {
  console.log('IPC fetch-data called');
  const serverReady = await isServerReady('http://localhost:8000/data');
  if (!serverReady) {
    console.error('FastAPI server is not ready');
    return { error: 'FastAPI server is not ready' };
  }

  console.log('Fetching data from FastAPI server');
  try {
    const response = await axios.get('http://localhost:8000/data');
    console.log('Data fetched:', response.data);
    return response.data;
  } catch (error) {
    console.error('Error fetching data:', error);
    return { error: 'Failed to fetch data' };
  }
});

const terminateServer = (): void => {
  if (serverProcess && typeof serverProcess !== 'string') {
    console.log('Terminating FastAPI server...');
    serverProcess.kill();
    serverProcess = null;
  }
};

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

app.on('before-quit', terminateServer);

app.on('ready', () => {
  if (!appInitialized) {
    appInitialized = true;
    startServer();
    createWindow();
  }
});
