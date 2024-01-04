// /src/pages/Workspace.tsx
import React, { useEffect, useState } from 'react';

const Workspace = () => {
  const [files, setFiles] = useState([]);
  const [workspacePath, setWorkspacePath] = useState('');
  const copyMarkerFileName = 'copy_marker.txt'; // Name of the marker file

  useEffect(() => {
    const fetchDataPathAndListFiles = async () => {
      const dataPath = await window.electron.invoke('get-data-path');
      console.log('Data Path:', dataPath);

      // Format the path with quotes for terminal use
      const formattedPath = `"${dataPath}"`;
      setWorkspacePath(formattedPath);

      // Call the function to list files
      let filesList = await window.electron.listFiles(dataPath);
      // Filter out the copyMarker file
      filesList = filesList.filter((file) => file !== copyMarkerFileName);
      setFiles(filesList);
    };

    fetchDataPathAndListFiles();
  }, []);

  return (
    <div className="w-full flex flex-col justify-center items-center h-[calc(100vh_-_50px)]">
      <div className="w-full flex flex-col justify-start items-start h-full overflow-y-scroll">
        <div className="w-full h-[50px] px-4 flex flex-row justify-between items-center">
          <h1 className="text-[1.4em] font-light">Workspace</h1>
          <p>
            Path:{' '}
            <code className="bg-gray-100 p-1 rounded-md text-sm">
              {workspacePath}
            </code>
          </p>
        </div>

        <ul className="w-full">
          {files.map((file) => (
            <li key={file} className="border-y py-2 w-full bg-gray-100/40 px-4">
              {file}
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
};

export default Workspace;
