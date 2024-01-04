// /src/preload.ts
import { contextBridge, ipcRenderer } from 'electron';

type InvokeFunction = {
  (channel: 'fetch-data'): Promise<any>;
  (channel: 'get-file-storage-path'): Promise<string>;
  (channel: 'list-files', directoryPath: string): Promise<string[]>;
  (
    channel: 'upload-file',
    args: { filePath: string; destination: string },
  ): Promise<any>;
  (channel: 'select-file'): Promise<string | null>;
  (channel: 'get-desktop-path'): Promise<string>;
  (channel: 'get-data-path'): Promise<string>;
};

const electronAPI = {
  fetchData: () => ipcRenderer.invoke('fetch-data'),
  invoke: ((channel: string, ...args: any[]) =>
    ipcRenderer.invoke(channel, ...args)) as InvokeFunction,
  listFiles: (directoryPath: string) =>
    ipcRenderer.invoke('list-files', directoryPath),
  uploadFile: (filePath: string, destination: string) =>
    ipcRenderer.invoke('upload-file', { filePath, destination }),
  selectFile: () => ipcRenderer.invoke('select-file'),
  getDesktopPath: () => ipcRenderer.invoke('get-desktop-path'),
  getDataPath: () => ipcRenderer.invoke('get-data-path'),
};

contextBridge.exposeInMainWorld('electron', electronAPI);
