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
  const [fileContent, setFileContent] = useState<File | null>(null); // Store the file
  const [uploadedData, setUploadedData] = useState(null); // State to store uploaded data

  // Calculate the data metrics
  let observations = 0;
  let features = 0;
  let fileSizeKB = fileSize ? fileSize / 1024 : 0; // Convert bytes to kilobytes

  if (uploadedData?.uploadedData) {
    // Split the data into rows and filter out empty rows
    const rows = uploadedData.uploadedData
      .split('\n')
      .filter((row) => row.trim() !== '');

    // Calculate observations (excluding the header)
    observations = rows.length > 0 ? rows.length - 1 : 0; // Assuming the first row is a header

    // Calculate features based on the header row
    features = rows[0]?.split(',').length || 0;
  }

  const steps = [
    'Loading Python . . .',
    'Starting server process . . .',
    'Waiting for application startup . . .',
    'Checking API status . . .',
    'Application startup complete . . .',
  ];

  useEffect(() => {
    const timer = setInterval(() => {
      setLoadingTime((prev) => prev + 1);
    }, 1000);

    const stepTimer = setInterval(() => {
      setCurrentStep((prev) => (prev + 1) % steps.length);
    }, 3000);

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

  // Upload the file
  const uploadFile = async (file: File) => {
    const formData = new FormData();
    formData.append('file', file);

    // console.log(`File Name: ${file.name}, File Type: ${file.type}`);

    try {
      const response = await fetch('http://localhost:8000/uploadfile/', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('File upload failed');
      }

      const result = await response.json();
      console.log('Uploaded data:', result);
      setUploadedData(result); // Update state with uploaded data
    } catch (error) {
      console.error('Error in uploading file:', error);
      alert(error.message);
    }
  };

  // Handle file drop
  const handleDrop = (event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setIsDragOver(false);

    const files = event.dataTransfer.files;
    if (files.length) {
      const file = files[0];
      const fileExtension = file.name.split('.').pop()?.toLowerCase();

      switch (fileExtension) {
        case 'spss':
        case 'sav':
          console.log(`SPSS/Sav file detected: ${file.name}`);
          setFileName(file.name);
          setFileSize(file.size);
          setFileContent(file); // Store the SPSS/Sav data file
          break;
        case 'csv':
          console.log(`CSV file detected: ${file.name}`);
          setFileName(file.name);
          setFileSize(file.size);
          setFileContent(file); // Store the CSV data file
          break;
        case 'txt':
          console.log(`CSV file detected: ${file.name}`);
          setFileName(file.name);
          setFileSize(file.size);
          setFileContent(file); // Store the TXT data file
          break;
        default:
          console.log(`Unsupported file type: ${file.name}`);
          setFileName(`Unsupported file type: ${file.name}`);
          setFileSize(null);
          setFileContent(null); // Clear the file content for unsupported types
      }
    }

    uploadFile(files[0]);
  };

  return (
    <div className="w-full h-full flex flex-col justify-between items-start overflow-x-auto">
      {!fileName && (
        <div
          id="dropZone"
          className={`w-full h-full p-4 flex flex-col justify-center items-center ${
            isDragOver ? 'bg-[#242424]' : ''
          }`}
          onDrop={handleDrop}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
        >
          {!fileName && (
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

          {!fileName && !fileSize && (
            <div>{fileName}</div> // Display error message for unsupported file typ
          )}

          {/* <pre className="w-full overflow-x-auto whitespace-pre-wrap text-left text-xs border bg-[#242424] rounded-md p-4">
        <code>{JSON.stringify(data, null, 2)}</code>
      </pre> */}
        </div>
      )}

      {fileName && fileSize && uploadedData && (
        <div className="w-full h-full p-4 flex flex-col justify-start items-start space-y-4">
          {fileName && fileSize && (
            <div className="flex flex-row justify-between items-center w-full">
              <div className="flex flex-row justify-start items-center space-x-2">
                <div>
                  {fileName}{' '}
                  <span className="text-white/50">
                    ({fileSizeKB.toFixed(2)} KB)
                  </span>
                </div>

                <div className="flex flex-row justify-start items-start bg-white/10 rounded-full px-[8px] py-[2px] text-sm">
                  {observations.toLocaleString()} observations x {features}{' '}
                  features
                </div>
              </div>

              <div className="flex flex-row justify-start items-center space-x-4">
                <div className="flex flex-row justify-start items-center space-x-1">
                  <div className="text-white/40">Model:</div>
                  <select className="bg-black text-white px-2.5 py-[7px] rounded-md">
                    <option value="bart">BART</option>
                    <option value="xbart">XBART</option>
                    <option value="bcf">bcf</option>
                  </select>
                </div>

                <button
                  // onClick={uploadFile}
                  className="bg-[#bf5700] text-white px-2.5 py-1 rounded-md"
                >
                  Analyze
                </button>
              </div>
            </div>
          )}

          <div className="w-full text-sm">
            {uploadedData?.uploadedData &&
              (() => {
                // Split the data into observations
                const observations = uploadedData.uploadedData.split('\n');

                // Split each observation by its corresponding features
                const features = observations.map((observation: string) =>
                  observation.split(','),
                );

                return (
                  <table className="w-full border-collapse">
                    <tbody>
                      {features.map(
                        (feature: string[], featureIndex: number) => (
                          <tr
                            key={featureIndex}
                            className={
                              featureIndex % 2 === 0 ? 'bg-[#24242460]' : ''
                            }
                          >
                            {feature.map((cell: string, cellIndex: number) => (
                              <td
                                key={cellIndex}
                                className={`border-b border-b-[#FFFFFF10] p-1.5 ${
                                  featureIndex === 0
                                    ? 'text-blue-500'
                                    : 'text-white'
                                }`}
                              >
                                {cell}
                              </td>
                            ))}
                          </tr>
                        ),
                      )}
                    </tbody>
                  </table>
                );
              })()}
          </div>
        </div>
      )}

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
