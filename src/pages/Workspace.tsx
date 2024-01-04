// /src/pages/Workspace.tsx
import React, { useEffect, useState } from 'react';
import { FileDetails } from '../types/fileDetails';

const Workspace = () => {
  const [files, setFiles] = useState<FileDetails[]>([]);
  const [workspacePath, setWorkspacePath] = useState('');

  useEffect(() => {
    const fetchDataPathAndListFiles = async () => {
      const dataPath = await window.electron.invoke('get-data-path');
      setWorkspacePath(`"${dataPath}"`);
      const filesList: FileDetails[] = await window.electron.listFiles(
        dataPath,
      );
      setFiles(filesList);
    };

    fetchDataPathAndListFiles();
  }, []);

  const formatFileSize = (size: number) => {
    if (size < 1024) return size + ' bytes';
    else if (size < 1048576) return (size / 1024).toFixed(2) + ' KB';
    else return (size / 1048576).toFixed(2) + ' MB';
  };

  return (
    <div className="w-full flex flex-col justify-center items-center h-[calc(100vh_-_50px)]">
      <div className="w-full flex flex-col justify-start items-start h-full overflow-y-scroll">
        <div className="w-full h-[60px] px-4 flex flex-row justify-between items-center">
          <h1 className="text-[1.4em] font-light">Workspace</h1>
          <p>
            Path:{' '}
            <code className="bg-gray-100 p-1 rounded-md text-sm">
              {workspacePath}
            </code>
          </p>
        </div>
        <div className="w-full">
          {files.map((file) => (
            <div
              key={file.name}
              className="border-y py-2 w-full bg-gray-100/40 px-4 flex flex-row justify-between items-center"
            >
              <div>{file.name}</div>
              <div>{formatFileSize(file.size)}</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default Workspace;
