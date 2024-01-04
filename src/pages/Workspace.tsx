import React, { useEffect, useState } from 'react';
import { FileDetails } from '../types/fileDetails';

const Workspace = () => {
  const [files, setFiles] = useState<FileDetails[]>([]);
  const [workspacePath, setWorkspacePath] = useState('');
  const [uploadError, setUploadError] = useState('');

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

  // Drag and drop handlers
  const handleDragOver = (event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault();
  };

  const handleDrop = async (event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    const droppedFiles = event.dataTransfer.files;
    if (droppedFiles.length) {
      const file = droppedFiles[0];
      const fileName = file.name;

      // Check for file collision
      const fileExists = files.some((f) => f.name === fileName);
      if (fileExists) {
        setUploadError(`A file with the name ${fileName} already exists.`);
        return;
      }

      if (file.type === 'text/csv' || file.name.endsWith('.csv')) {
        await uploadFile(file.path);
      } else {
        console.log('Invalid file type. Only CSV files are allowed.');
      }
    }
  };

  // File upload handler
  const uploadFile = async (filePath: string) => {
    const destinationPath = await window.electron.invoke('get-data-path');
    try {
      const result = await window.electron.invoke('upload-file', {
        filePath,
        destination: destinationPath,
      });
      if (result.success) {
        const updatedFilesList = await window.electron.listFiles(
          destinationPath,
        );
        setFiles(updatedFilesList);
        setUploadError(''); // Clear any previous error
        console.log('File uploaded:', result.path);
      }
    } catch (error) {
      console.error('Error uploading file:', error);
      setUploadError('Error uploading file');
    }
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
          {files.map((file, index) => (
            <div
              key={file.name}
              className={`border-b border-b-gray-200 py-2 w-full px-4 flex flex-row justify-between items-center ${
                index % 2 === 0 ? 'bg-gray-100/40' : 'bg-white'
              } ${index === 0 ? 'border-t border-t-gray-200' : ''}`}
            >
              <div>{file.name}</div>
              <div className="flex flex-row justify-start items-center space-x-4">
                <div># observations x # features</div>
                <div>{formatFileSize(file.size)}</div>
              </div>
            </div>
          ))}
        </div>
      </div>

      <div
        id="dropzone"
        className="bg-gray-100 border w-[400px] h-[100px] mb-6 flex flex-col justify-center items-center rounded-full"
        onDragOver={handleDragOver}
        onDrop={handleDrop}
      >
        {uploadError ? (
          <p className="text-red-500">{uploadError}</p>
        ) : (
          'Drag your local dataset file here'
        )}
      </div>
    </div>
  );
};

export default Workspace;
