// /src/pages/Homepage.tsx
import React, { useEffect, useState } from 'react';

interface FetchDataResponse {
  status: string;
}

const Homepage = () => {
  const [data, setData] = useState<FetchDataResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [isDragOver, setIsDragOver] = useState(false);
  const [fileName, setFileName] = useState<string | null>(null);
  const [fileSize, setFileSize] = useState<number | null>(null);
  const [loadingTime, setLoadingTime] = useState(0);
  const [currentStep, setCurrentStep] = useState(0);

  const steps = [
    'Loading Python . . .',
    'Starting server . . .',
    'Checking API status . . .',
  ];

  useEffect(() => {
    const timer = setInterval(() => {
      setLoadingTime((prev) => prev + 1);
    }, 1000);

    const stepTimer = setInterval(() => {
      setCurrentStep((prev) => (prev + 1) % steps.length);
    }, 4000);

    window.electron
      .fetchData()
      .then((fetchedData: FetchDataResponse) => {
        setData(fetchedData);
        setLoading(false);
        clearInterval(timer);
        clearInterval(stepTimer);
      })
      .catch((fetchError) => {
        console.error('Error in fetchData:', fetchError);
        setError('Failed to fetch data');
        setLoading(false);
        clearInterval(timer);
        clearInterval(stepTimer);
      });

    return () => {
      clearInterval(timer);
      clearInterval(stepTimer);
    };
  }, []);
  if (loading) {
    return (
      <div className="w-full h-full flex flex-col justify-center items-center">
        <div className="w-[260px] flex flex-col justify-start items-start space-y-2">
          <div className="font-bold">
            Initializing Arborist: {loadingTime} sec
          </div>

          <ul className="list-inside text-white/30">
            <ol>{steps[currentStep]}</ol>
          </ul>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="w-full h-full flex flex-col justify-center items-center">
        <div>Error: {error}</div>
      </div>
    );
  }

  // Handle drag over
  const handleDragOver = (event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setIsDragOver(true);
  };

  // Handle drag leave
  const handleDragLeave = (event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setIsDragOver(false);
  };

  // Handle file drop
  const handleDrop = (event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setIsDragOver(false);

    const files = event.dataTransfer.files;
    if (files.length) {
      const file = files[0];
      const fileExtension = file.name.split('.').pop();

      switch (fileExtension) {
        case 'spss':
        case 'sav':
          console.log(`SPSS/Sav file detected: ${file.name}`);
          setFileName(file.name);
          setFileSize(file.size);
          // Handle the SPSS/Sav data file
          break;
        case 'csv':
          console.log(`CSV file detected: ${file.name}`);
          setFileName(file.name);
          setFileSize(file.size);
          // Handle the CSV data file
          break;
        default:
          console.log(`Unsupported file type: ${file.name}`);
          setFileName(`Unsupported file type: ${file.name}`);
          setFileSize(null);
      }
    }
  };

  return (
    <div className="w-full h-full flex flex-col justify-between items-start">
      <div
        id="dropZone"
        className={`w-full h-full p-4 flex flex-col justify-center items-center ${
          isDragOver ? 'bg-[#242424]' : ''
        }`}
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
      >
        {fileName ? (
          fileSize ? (
            <div className="flex flex-col justify-start items-center space-y-4">
              <div>
                <p>File Name: {fileName}</p>
                <p>File Size: {fileSize} bytes</p>
              </div>

              <button className="bg-red-600 text-white px-3 py-1.5 rounded-md">
                Analyze
              </button>
            </div>
          ) : (
            <div>{fileName}</div> // Display error message for unsupported file type
          )
        ) : (
          <div className="flex flex-col justify-center items-center w-full space-y-4">
            <div className="text-xl">Drag your local dataset file here</div>
            <div className="text-white/30 flex flex-row justify-start items-center space-x-2">
              <div>Supported file types:</div>
              <ul className="flex flex-row justify-start items-center space-x-2">
                <li className="border-[2px] border-white/10 rounded-full px-[10px] py-[4px]">
                  .spss
                </li>
                <li className="border-[2px] border-white/10 rounded-full px-[10px] py-[4px]">
                  .sav
                </li>
                <li className="border-[2px] border-white/10 rounded-full px-[10px] py-[4px]">
                  .csv
                </li>
              </ul>
            </div>
          </div>
        )}
      </div>

      {/* <pre className="w-full overflow-x-auto whitespace-pre-wrap text-left text-xs border bg-[#242424] rounded-md p-4">
        <code>{JSON.stringify(data, null, 2)}</code>
      </pre> */}

      <div className="h-[50px] w-full text-white/30 bg-[#242424] flex flex-row justify-start items-center px-4 space-x-1">
        <div>API:</div>
        {(data?.status && <div className="text-[#bf5700]">Online</div>) || (
          <div className="text-gray-300">Offline</div>
        )}
      </div>
    </div>
  );
};

export default Homepage;
