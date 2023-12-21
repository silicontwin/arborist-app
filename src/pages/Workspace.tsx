// /src/pages/Workspace.tsx
import React, { useEffect, useState } from 'react';

interface FetchDataResponse {
  api: string;
}

const Workspace = () => {
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
  const [predictions, setPredictions] = useState(null); // State for storing predictions
  const [userDataPath, setUserDataPath] = useState('');
  const [isUploading, setIsUploading] = useState(false); // State for tracking upload status
  const [isAnalyzing, setIsAnalyzing] = useState(false); // State for tracking analysis status

  // Calculate the data metrics
  let observations = 0;
  let features = 0;
  let fileSizeKB = fileSize ? fileSize / 1024 : 0; // Convert bytes to kilobytes

  if (uploadedData?.uploadedData) {
    // Split the data into rows and filter out empty rows
    const rows = uploadedData.uploadedData
      .split('\n')
      .filter((row: string) => row.trim() !== '');

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

  // Function to fetch the userData path
  useEffect(() => {
    const fetchUserDataPath = async () => {
      try {
        const path = await window.electron.invoke('get-file-storage-path');
        setUserDataPath(path);
      } catch (error) {
        console.error('Error getting user data path:', error);
      }
    };

    fetchUserDataPath();
  }, []);

  // if (loading) {
  //   return (
  //     <div className="w-full min-h-[calc(100vh_-_50px)] flex flex-col justify-center items-center">
  //       <div className="w-[260px] flex flex-col justify-start items-start space-y-2">
  //         <div className="font-bold">
  //           Initializing Arborist: {loadingTime} sec
  //         </div>

  //         <ul className="list-inside text-white/30">
  //           <ol>{steps[currentStep]}</ol>
  //         </ul>
  //       </div>
  //     </div>
  //   );
  // }

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

  // Get the file storage path
  const getFileStoragePath = async () => {
    try {
      const path = await window.electron.invoke('get-file-storage-path');
      return path;
    } catch (error) {
      console.error('Error getting file storage path:', error);
      return null;
    }
  };

  // Upload the file
  const uploadFile = async (file: File) => {
    setIsUploading(true);

    const fileStoragePath = await getFileStoragePath();
    if (!fileStoragePath) {
      console.error('Failed to get file storage path');
      setIsUploading(false);
      return;
    }

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
    } finally {
      setIsUploading(false);
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

  // ---------------------------------------------------------------------------

  // Prepend predictions to data
  const prependPredictionsToData = (dataString: string, predictions: any) => {
    const predictionValues = predictions.predictions.map((p: number[]) => p[0]);

    // Trim the dataString to remove any trailing newlines or whitespace
    const trimmedDataString = dataString.trim();

    const observations = trimmedDataString.split('\n');
    const headers = 'predictions,' + observations[0]; // Prepending the 'predictions' header

    const observationRows = observations.slice(1).map((observation, index) => {
      const prediction =
        predictionValues[index] !== undefined
          ? predictionValues[index].toString()
          : 'N/A';
      return `${prediction},${observation}`; // Prepending the prediction to each observation
    });

    return [headers, ...observationRows].join('\n');
  };

  const analyzeData = async () => {
    if (!uploadedData?.uploadedData) {
      alert('No data to analyze');
      return;
    }

    setIsAnalyzing(true);

    const observations = uploadedData.uploadedData.split('\n').slice(1);
    const formattedData = observations.map((observation: string) =>
      observation.split(',').map((val: string) => parseFloat(val)),
    );

    // Determine the correct length of observations
    const correctObvervationLength = formattedData[0].length;

    // Filter out observations with incorrect lengths
    const consistentFormattedData = formattedData.filter(
      (observation: number[]) =>
        observation.length === correctObvervationLength,
    );

    if (consistentFormattedData.length !== formattedData.length) {
      console.warn(
        'Some observations were skipped due to inconsistent lengths',
      );
    }

    const X = consistentFormattedData.map((observation: number[]) =>
      observation.slice(0, -1),
    ); // all features except last

    const y = consistentFormattedData.map(
      (observation: number[]) => observation[observation.length - 1],
    ); // last feature

    const requestBody = { X, y };

    console.log('Sending data to /predict:', requestBody); // Debugging

    try {
      const response = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestBody),
      });

      if (!response.ok) {
        throw new Error('Prediction failed');
      }

      const predictionData = await response.json();
      console.log('Received predictions:', predictionData);

      // Ensure predictionData has the 'predictions' key and is an array
      if (
        !predictionData.predictions ||
        !Array.isArray(predictionData.predictions)
      ) {
        console.error(
          'Prediction data is missing the predictions key or is not an array:',
          predictionData,
        );
        return;
      }

      setPredictions(predictionData.predictions); // Store the predictions in state

      const updatedData = prependPredictionsToData(
        uploadedData.uploadedData,
        predictionData,
      );
      setUploadedData({ ...uploadedData, uploadedData: updatedData });
      setIsAnalyzing(false);
    } catch (error) {
      console.error('Error in making prediction:', error);
      alert(error.message);
      setIsAnalyzing(false);
    }
  };

  const isPredictionColumn = (
    cellIndex: number,
    predictionsAvailable: boolean,
  ) => {
    return predictionsAvailable && cellIndex === 0;
  };

  // ---------------------------------------------------------------------------

  const downloadCSV = () => {
    const csvString = uploadedData?.uploadedData;
    const blob = new Blob([csvString], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);

    // Create a temporary link to trigger the download
    const link = document.createElement('a');
    link.href = url;
    link.setAttribute('download', 'predictions.csv'); // Need to add the date to the beginning of the file name
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <div className="w-full h-full flex flex-col justify-start items-start">
      <div
        id="statusBar"
        className="h-[30px] text-sm w-full text-white/30 bg-[#24242460] flex flex-row justify-start items-center px-4 space-x-4"
      >
        <div className="flex space-x-1">
          <div>API:</div>
          {(data?.api === 'online' && (
            <div className="text-yellow-300">Online</div>
          )) || <div className="text-gray-300">Offline</div>}
        </div>

        <div className="flex space-x-1">
          Workspace path:{' '}
          <span className="text-white/60">`{userDataPath}`</span>
        </div>
      </div>

      <div
        id="loadingPanel"
        className={`w-full min-h-[calc(100vh_-_80px)] bg-[#393A4C] flex flex-col justify-center items-center ${
          isUploading || isAnalyzing ? '' : 'hidden'
        }`}
      >
        {isUploading && 'Loading data . . .'}
        {isAnalyzing && 'Analyzing data . . .'}
      </div>

      {!fileName && (
        <div
          id="dropZone"
          className={`w-full h-full flex flex-col justify-center items-center ${
            isDragOver ? 'bg-[#242424]' : ''
          }`}
          onDrop={handleDrop}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
        >
          {!fileName && (
            <div className="h-[calc(100vh_-_80px)] flex flex-col justify-center items-center w-full space-y-4">
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
        <div className="w-full h-full flex flex-col justify-start items-start">
          {fileName && fileSize && (
            <div className="h-[50px] flex flex-row justify-between items-center w-full absolute top-[80px] left-0 px-4">
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

              <div className="flex flex-row justify-start items-center space-x-2">
                <div className="flex flex-row justify-start items-center space-x-1">
                  <div className="text-white/40">Model:</div>
                  <select className="bg-black text-white px-2 py-[5px] rounded-md">
                    <option value="bart">BART</option>
                    <option value="xbart">XBART</option>
                    <option value="bcf">bcf</option>
                  </select>
                </div>

                <button
                  onClick={analyzeData}
                  className="bg-red-600 text-white px-2 py-0.5 rounded-md"
                >
                  Analyze
                </button>

                {predictions && (
                  <div>
                    <button
                      onClick={downloadCSV}
                      className="bg-yellow-400 text-black px-2 py-0.5 rounded-md"
                    >
                      Download
                    </button>
                  </div>
                )}

                {/* {predictions && (
                  <div>
                    <h3>Predictions:</h3>
                    <pre>
                      <code>{JSON.stringify(predictions, null, 2)}</code>
                    </pre>
                  </div>
                )} */}
              </div>
            </div>
          )}

          <div
            id="data"
            className="w-full overflow-auto min-h-[calc(100vh_-_130px)] max-h-[calc(100vh_-_130px)] mt-[50px] px-4 text-sm"
          >
            {uploadedData?.uploadedData &&
              (() => {
                // Split the data into observations
                const observations = uploadedData.uploadedData.split('\n');

                // Split each observation by its corresponding features
                const features = observations.map((observation: string) =>
                  observation.split(','),
                );
                const predictionsAvailable = predictions !== null;

                return (
                  <table className="w-full border-collapse">
                    <tbody>
                      {features.map(
                        (feature: string[], featureIndex: number) => (
                          <tr
                            key={featureIndex}
                            className={
                              featureIndex % 2 === 0 ? 'bg-[#24242480]' : ''
                            }
                          >
                            {feature.map((cell: string, cellIndex: number) => (
                              <td
                                key={cellIndex}
                                className={`border-b border-b-[#FFFFFF10] p-1.5 ${
                                  isPredictionColumn(
                                    cellIndex,
                                    predictionsAvailable,
                                  )
                                    ? 'bg-[#393A4C]'
                                    : ''
                                } ${
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
    </div>
  );
};

export default Workspace;
