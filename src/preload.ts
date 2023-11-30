// /src/preload.ts
import { contextBridge, ipcRenderer } from 'electron';

contextBridge.exposeInMainWorld('electron', {
  fetchData: () => ipcRenderer.invoke('fetch-data'),
  onFetchDataReply: (callback: (data: any) => void) => {
    ipcRenderer.on('fetch-data-reply', (_event, data) => callback(data));
  },
});
