// /src/types/globals.d.ts

interface FileDetails {
  name: string;
  size: number;
}

// Extend the Window interface
interface Window {
  electron: {
    onFetchDataReply: (callback: (data: any) => void) => void;
    fetchData: () => Promise<any>;
    invoke: (channel: string, ...args: any[]) => Promise<any>;
    listFiles: (directoryPath: string) => Promise<FileDetails[]>;
    getDesktopPath: () => Promise<string>;
    getDataPath: () => Promise<string>;
    maximizeWindow: () => Promise<void>;
    minimizeWindow: () => Promise<void>;
  };
}

// Image imports
declare module '*.png' {
  const value: any;
  export = value;
}
declare module '*.jpg' {
  const value: any;
  export = value;
}
declare module '*.jpeg' {
  const value: any;
  export = value;
}
declare module '*.gif' {
  const value: any;
  export = value;
}
declare module '*.svg' {
  const value: any;
  export = value;
}
