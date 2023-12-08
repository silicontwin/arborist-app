// /src/preload.ts
import { contextBridge, ipcRenderer } from 'electron';

// Define a type for the invoke function
type InvokeFunction = {
  (channel: 'fetch-data'): Promise<any>;
  (channel: 'get-file-storage-path'): Promise<string>;
};

const electronAPI = {
  fetchData: () => ipcRenderer.invoke('fetch-data'),
  invoke: ((channel: string, ...args: any[]) =>
    ipcRenderer.invoke(channel, ...args)) as InvokeFunction,
};

contextBridge.exposeInMainWorld('electron', electronAPI);
