import React, { useEffect, useState } from 'react';
import { FileDetails } from '../types/fileDetails';
import { FaRegFolderOpen } from 'react-icons/fa';
import { MdOutlineInsertDriveFile } from 'react-icons/md';
import { TbDragDrop } from 'react-icons/tb';

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
          <div className="flex flex-row justify-start items-center space-x-1">
            <FaRegFolderOpen className="w-[26px] h-[26px]" />
            <div className="text-[1.4em]">Workspace</div>
          </div>
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
              className={`border-b border-b-gray-200 py-2 w-full px-4 flex flex-row justify-between items-center hover:bg-gray-200/60 ${
                index % 2 === 0 ? 'bg-gray-100/40' : 'bg-white'
              } ${index === 0 ? 'border-t border-t-gray-200' : ''}`}
            >
              <div className="flex flex-row justify-start items-center space-x-1">
                <MdOutlineInsertDriveFile className="w-[20px] h-[20px] text-gray-400" />
                <div className="font-medium">{file.name}</div>
              </div>
              <div className="flex flex-row justify-start items-center space-x-4">
                <div># observations x # features</div>
                <div className="w-[100px] text-right">
                  {formatFileSize(file.size)}
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      <div
        id="dropzone"
        className="bg-gray-100 border w-[380px] h-[100px] mb-10 flex flex-col justify-center items-center rounded-full"
        onDragOver={handleDragOver}
        onDrop={handleDrop}
      >
        {uploadError ? (
          <p className="text-red-500">{uploadError}</p>
        ) : (
          <div className="flex flex-row justify-start items-center space-x-2">
            <TbDragDrop className="w-[26px] h-[26px]" />
            <div className="text-[1.1em] font-semibold">
              Drag your local .csv dataset here
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default Workspace;
