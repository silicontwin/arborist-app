// /src/types/globals.d.ts

// Extend the Window interface
interface Window {
  electron: {
    onFetchDataReply: (callback: (data: any) => void) => void;
    fetchData: () => Promise<any>;
  };
}
