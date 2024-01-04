// /src/types/globals.d.ts

// Extend the Window interface
interface Window {
  electron: {
    onFetchDataReply: (callback: (data: any) => void) => void;
    fetchData: () => Promise<any>;
    invoke: (channel: string, ...args: any[]) => Promise<any>;
    listFiles: (directoryPath: string) => Promise<string[]>;
    getDesktopPath: () => Promise<string>;
    getDataPath: () => Promise<string>;
  };
}
