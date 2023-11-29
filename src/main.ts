// /src/main.ts
import { app, BrowserWindow } from 'electron';
import path from 'node:path';
import isDev from 'electron-is-dev';
import { execFile, ChildProcess } from 'child_process';

let serverProcess: ChildProcess | null = null;

const createWindow = (): void => {
  const win = new BrowserWindow({
    width: 1600,
    height: 900,
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
