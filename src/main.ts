// /src/main.ts
import { app, BrowserWindow, ipcMain } from 'electron';
import axios from 'axios';
import path from 'node:path';
import isDev from 'electron-is-dev';
import { execFile, ChildProcess } from 'child_process';

let serverProcess: ChildProcess | null = null;

// Function to check if the server is ready
const isServerReady = async (
  url: string,
  retries: number = 5,
  delay: number = 1000,
): Promise<boolean> => {
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

const createWindow = (): void => {
  const win = new BrowserWindow({
    width: 1600,
    height: 900,
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
      preload: path.join(__dirname, 'preload.js'),
    },
  });

  const indexPath = isDev
    ? path.join(__dirname, '../src/index.html') // Path in development
    : path.join(process.resourcesPath, 'app', 'src', 'index.html'); // Path in production

  win.loadFile(indexPath);
  win.webContents.openDevTools();

  // Start the FastAPI server
  const apiPath = isDev
    ? path.join(__dirname, '../src/resources', 'api') // Path in development
    : path.join(process.resourcesPath, 'app', 'src', 'resources', 'api'); // Path in production

  console.log('Starting FastAPI server...');

  serverProcess = execFile(
    apiPath,
    (error: Error | null, stdout: string, stderr: string) => {
      if (error) {
        console.error('Error starting FastAPI server:', error);
        return;
      }
      console.log(`stdout: ${stdout}`);
      console.error(`stderr: ${stderr}`);
    },
  );

  console.log('FastAPI server should be running...');
};

const terminateServer = (): void => {
  if (serverProcess) {
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

app.on('ready', createWindow);
