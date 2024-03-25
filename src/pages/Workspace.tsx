import React, { useEffect, useState, useMemo } from 'react';
import { FileDetails } from '../types/fileDetails';
import { FaRegFolderOpen } from 'react-icons/fa';
import { MdOutlineInsertDriveFile } from 'react-icons/md';
import { TbDragDrop } from 'react-icons/tb';
import { VscSettings } from 'react-icons/vsc';
import { BiSolidError } from 'react-icons/bi';

interface DataItem {
  [key: string]: any;
}

interface ColumnStatus {
  [key: string]: { isNumeric: boolean; isChecked: boolean };
}

interface StateTypes {
  selectedTreatment: string;
}

const Workspace = () => {
  const [files, setFiles] = useState<FileDetails[]>([]);
  const [workspacePath, setWorkspacePath] = useState('');
  const [uploadError, setUploadError] = useState('');
  const [selectedFileData, setSelectedFileData] = useState('');
  const [selectedFileName, setSelectedFileName] = useState('');
  const [isDataModalOpen, setIsDataModalOpen] = useState(false);
  const [action, setAction] = useState('summarize');
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [predictions, setPredictions] = useState(null);
  const [analysisStartTime, setAnalysisStartTime] = useState<number | null>(
    null,
  );
  const [analysisEndTime, setAnalysisEndTime] = useState<number | null>(null);
  const [elapsedTime, setElapsedTime] = useState<number | null>(null);
  const [intervalId, setIntervalId] = useState<NodeJS.Timeout | null>(null);
  const [totalElapsedTime, setTotalElapsedTime] = useState<string | null>(null);
  const [apiStatus, setApiStatus] = useState('Starting server');
  const [observationSelection, setObservationSelection] = useState('all');
  const [selectedModel, setSelectedModel] = useState('bart');
  const [columnNumericStatus, setColumnNumericStatus] = useState<ColumnStatus>(
    {},
  );
  const [selectedFeature, setSelectedFeature] = useState<string>('');
  const [selectedOutcome, setSelectedOutcome] =
    useState<string>('Please select');
  const [selectedTreatment, setSelectedTreatment] =
    useState<StateTypes['selectedTreatment']>('');
  const [availableFeatures, setAvailableFeatures] = useState<string[]>([]);
  const [headTailRows, setHeadTailRows] = useState<number>(20);
  const [isModelParamsVisible, setIsModelParamsVisible] =
    useState<boolean>(false);
  const [observationsRemoved, setObservationsRemoved] = useState<number>(0);
  const [isAnalysisComplete, setIsAnalysisComplete] = useState(false);

  const selectedColumns = useMemo(() => {
    return Object.entries(columnNumericStatus)
      .filter(([_, value]) => value.isChecked && value.isNumeric)
      .map(([key]) => key);
  }, [columnNumericStatus]);

  useEffect(() => {
    console.log('Selected features:', selectedColumns);
  }, [selectedColumns]);

  useEffect(() => {
    // Fetch API status when component mounts
    const fetchApiStatus = async () => {
      try {
        const status = await window.electron.invoke('fetch-data');
        setApiStatus(status.api === 'online' ? 'Online' : 'Offline');
      } catch (error) {
        console.error('Error fetching API status:', error);
        setApiStatus('Error');
      }
    };

    fetchApiStatus();
  }, []);

  useEffect(() => {
    const fetchDataPathAndListFiles = async () => {
      const dataPath = await window.electron.invoke('get-data-path');
      setWorkspacePath(dataPath);
      const filesList: FileDetails[] = await window.electron.listFiles(
        dataPath,
      );
      setFiles(filesList);
    };

    fetchDataPathAndListFiles();

    // Reset the predictions state to null on component mount
    setPredictions(null);
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

  const handleFileDoubleClick = async (fileName: string) => {
    try {
      // Reset analysis time and related states before starting a new analysis
      setIsAnalyzing(false);
      setElapsedTime(null);
      setAnalysisStartTime(null);
      setAnalysisEndTime(null);
      setTotalElapsedTime(null);
      if (intervalId) {
        clearInterval(intervalId);
        setIntervalId(null);
      }

      // Reset action to 'summarize' every time a new file is opened
      setAction('summarize');

      // // Reset selected outcome
      // setSelectedOutcome('Please select');

      const response = await fetch('http://localhost:8000/summarize', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          fileName: fileName,
          workspacePath: workspacePath,
          headTailRows: headTailRows,
          outcomeVariable: '',
          action: 'summarize',
        }),
      });

      if (!response.ok) {
        throw new Error(`Failed to process file: ${response.statusText}`);
      }

      const data = await response.json();
      setSelectedFileData(JSON.stringify(data.data, null, 2));
      setObservationsRemoved(data.wrangle.observationsRemoved);

      // Ensure data.is_numeric exists and is an object before trying to use Object.keys
      if (data.is_numeric && typeof data.is_numeric === 'object') {
        const numericStatusUpdates: ColumnStatus = {};
        Object.keys(data.is_numeric).forEach((column) => {
          numericStatusUpdates[column] = {
            isNumeric: data.is_numeric[column],
            // Set isChecked to true for all numeric columns
            isChecked: data.is_numeric[column],
          };
        });
        setColumnNumericStatus(numericStatusUpdates);
      } else {
        console.error('Invalid or missing is_numeric data:', data);
      }

      setIsDataModalOpen(true);
      setSelectedFileName(fileName);

      // Reset the observationSelection state to its default value
      setObservationSelection('all');

      // Update available features to include all numeric columns
      const numericColumns = Object.keys(data.is_numeric).filter(
        (column) => data.is_numeric[column],
      );
      setAvailableFeatures(numericColumns);
    } catch (error) {
      console.error('Error processing file:', error);
      alert(
        `Error: ${
          error instanceof Error ? error.message : 'An unknown error occurred'
        }`,
      );
    }
  };

  const renderTableFromJsonData = () => {
    if (!selectedFileData.trim()) {
      return <div>Select a file to view its data.</div>;
    }

    let jsonData: DataItem[];
    try {
      jsonData = JSON.parse(selectedFileData);
    } catch (error) {
      console.error('Error parsing JSON data:', error);
      return <div>Invalid JSON format</div>;
    }

    if (!jsonData || !Array.isArray(jsonData) || jsonData.length === 0) {
      return <div>No data available</div>;
    }

    const columns = Object.keys(jsonData[0]);

    const handleCheckboxChange = (columnName: string) => {
      if (
        columnNumericStatus[columnName]?.isNumeric &&
        columnName !== selectedOutcome
      ) {
        setColumnNumericStatus((prevState) => ({
          ...prevState,
          [columnName]: {
            ...prevState[columnName],
            isChecked: !prevState[columnName].isChecked,
          },
        }));
      }
    };

    return (
      <table className="min-w-full text-sm border-collapse">
        <thead>
          <tr>
            {columns.map((column, index) => (
              <th
                key={index}
                className={`sticky top-0 z-10 border py-2 px-4 font-bold text-left uppercase text-[.925em] whitespace-nowrap ${
                  column === selectedOutcome
                    ? 'bg-blue-500 text-white'
                    : 'bg-white text-gray-700'
                }`}
              >
                {columnNumericStatus[column]?.isNumeric ? (
                  <label className="flex flex-row justify-start items-center space-x-1 cursor-pointer">
                    {!isAnalyzing && !totalElapsedTime && (
                      <input
                        type="checkbox"
                        checked={
                          column === selectedOutcome ||
                          columnNumericStatus[column]?.isChecked
                        }
                        onChange={() => handleCheckboxChange(column)}
                        className="form-checkbox"
                        disabled={column === selectedOutcome}
                      />
                    )}
                    <span
                      className={
                        column === selectedOutcome ? 'text-white' : 'text-black'
                      }
                    >
                      {column}
                    </span>
                  </label>
                ) : (
                  <span className="text-gray-400">{column}</span>
                )}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {jsonData.map((item, rowIndex) => (
            <tr
              key={rowIndex}
              className={`${
                Object.values(item).some((value) => value === '...')
                  ? 'bg-yellow-100'
                  : rowIndex % 2 === 0
                  ? 'bg-gray-100'
                  : 'bg-white'
              }`}
            >
              {columns.map((column, colIndex) => (
                <td
                  key={colIndex}
                  className={`border py-2 px-4 text-left whitespace-nowrap ${
                    column === selectedOutcome
                      ? 'bg-blue-500 text-white'
                      : !columnNumericStatus[column]?.isChecked ||
                        !columnNumericStatus[column]?.isNumeric
                      ? 'text-gray-400'
                      : 'text-black'
                  }`}
                >
                  {item[column]}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    );
  };

  const switchAction = (newAction: 'summarize' | 'analyze') => {
    setAction(newAction);
  };

  // ---

  const analyzeData = async (
    fileName: string,
    selectedColumns: any,
    outcomeVariable: any,
  ) => {
    // const selectedColumns = Object.entries(columnNumericStatus)
    //   .filter(([_, value]) => value.isChecked && value.isNumeric)
    //   .map(([key]) => key);

    // Ensure there are selected features before proceeding
    if (selectedColumns.length === 0) {
      alert('Please select at least one numeric column to analyze.');
      return;
    }

    console.log(
      'Starting analysis for:',
      fileName,
      'with columns:',
      selectedColumns,
    );

    const startTime = Date.now(); // Start tracking time
    setIsAnalyzing(true);

    // Start an interval to update elapsed time every 10 milliseconds
    const id = setInterval(() => {
      setElapsedTime(Date.now() - startTime);
    }, 10);
    setIntervalId(id);

    try {
      const response = await fetch('http://localhost:8000/summarize', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          fileName,
          workspacePath,
          headTailRows,
          selectedColumns,
          outcomeVariable,
          action: 'analyze',
        }),
      });

      if (!response.ok) {
        const responseBody = await response.text();
        throw new Error(
          `Failed to analyze file: ${response.statusText} - ${responseBody}`,
        );
      }

      console.log('Selected Outcome before fetch:', selectedOutcome);

      const data = await response.json();
      console.log('Analysis data received:', data);

      setSelectedFileData(JSON.stringify(data.data, null, 2));

      // Update the columnNumericStatus state to include the specified columns as numeric
      setColumnNumericStatus((prev) => ({
        ...prev,
        'Posterior Average (y hat)': { isNumeric: true, isChecked: true },
        '2.5th percentile': {
          isNumeric: true,
          isChecked: true,
        },
        '97.5th percentile': {
          isNumeric: true,
          isChecked: true,
        },
      }));

      if (data['Posterior Average (y hat)']) {
        setPredictions(data['Posterior Average (y hat)']);
      }

      // Complete the timing process
      clearInterval(intervalId!); // Clear the interval once the analysis is done
      const endTime = Date.now();
      setAnalysisEndTime(endTime);
      const finalElapsedTime = endTime - startTime;
      setElapsedTime(finalElapsedTime); // Update elapsed time one last time
      setTotalElapsedTime((finalElapsedTime / 1000).toFixed(2) + ' seconds'); // Storing the total elapsed time

      // Reset the observationSelection state to its default value
      setObservationSelection('all');

      // Reset selected outcome
      // setSelectedOutcome('Please select');
      setIsAnalysisComplete(true);
    } catch (error) {
      console.error('Error analyzing file:', error);
      alert(
        `Error: ${
          error instanceof Error ? error.message : 'An unknown error occurred'
        }`,
      );
    } finally {
      setIsAnalyzing(false);
    }
  };

  // Clear interval when component is unmounted or when analysis is not in progress
  useEffect(() => {
    return () => {
      if (intervalId) {
        clearInterval(intervalId);
      }
    };
  }, [intervalId]);

  // Function to display elapsed time in seconds to the nearest hundredth
  const displayElapsedTime = () => {
    return elapsedTime != null
      ? (elapsedTime / 1000).toFixed(2) + ' seconds' // Updated to show hundredths of a second
      : 'Calculating...';
  };

  const prependPredictionsToData = (
    dataString: string,
    predictionValues: number[],
  ) => {
    const rows = dataString.split('\n');
    const headers = 'Posterior Average (y^),' + rows[0];

    const updatedRows = rows.slice(1).map((row: string, index: number) => {
      const prediction =
        predictionValues[index] !== undefined
          ? predictionValues[index].toString()
          : 'N/A';
      return `${prediction},${row}`;
    });

    return [headers, ...updatedRows].join('\n');
  };

  // ---------------------------------------------------------------------------

  const downloadCSV = () => {
    const csvString = selectedFileData;
    if (!csvString) {
      alert('No data available to download');
      return;
    }

    // Remove the '.csv' extension from the file name, if it exists
    const baseFileName = selectedFileName.replace(/\.csv$/i, '');
    const newFileName = `${baseFileName}_predictions.csv`;

    const blob = new Blob([csvString], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);

    const link = document.createElement('a');
    link.href = url;
    link.setAttribute('download', newFileName);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  // ---

  // Determine if the cell is in the predictions column
  const isPredictionColumn = (cellIndex: number) => {
    return predictions != null && cellIndex === 0;
  };

  // ---

  const handleChange = (event: React.ChangeEvent<HTMLSelectElement>) => {
    setSelectedModel(event.target.value);
  };

  // useEffect to update available features when columnNumericStatus changes or a file is selected
  useEffect(() => {
    const features = Object.entries(columnNumericStatus)
      .filter(([_, value]) => value.isNumeric && value.isChecked)
      .map(([key]) => key);

    setAvailableFeatures(features);
  }, [columnNumericStatus, selectedFileName]);

  // // Reset selected features and outcome when a new file is selected
  // useEffect(() => {
  //   setSelectedOutcome('Please select');
  //   setSelectedFeature('');
  // }, [selectedFileName]);

  useEffect(() => {
    if (isAnalysisComplete) {
      setSelectedOutcome('Please select');
      setSelectedFeature('');
      setIsAnalysisComplete(false); // Reset the flag for the next analysis
    }
  }, [selectedFileName, isAnalysisComplete]);

  // const handleOutcomeChange = (event: React.ChangeEvent<HTMLSelectElement>) => {
  //   const newOutcome = event.target.value;
  //   setSelectedOutcome(newOutcome);

  //   setColumnNumericStatus((prev) => {
  //     const newState = { ...prev };
  //     if (selectedOutcome && newState[selectedOutcome]) {
  //       newState[selectedOutcome].isChecked = false;
  //     }
  //     return newState;
  //   });
  // };

  const getDropdownOptions = (excludeOptions: string[]) => {
    return Object.entries(columnNumericStatus)
      .filter(
        ([key, value]) => value.isNumeric && !excludeOptions.includes(key),
      )
      .map(([key]) => (
        <option key={key} value={key}>
          {key}
        </option>
      ));
  };

  const handleOutcomeChange = (event: React.ChangeEvent<HTMLSelectElement>) => {
    const newOutcome = event.target.value;
    setSelectedOutcome(newOutcome);
  };

  const handleTreatmentChange = (
    event: React.ChangeEvent<HTMLSelectElement>,
  ) => {
    const newTreatment = event.target.value;
    setSelectedTreatment(newTreatment);
  };

  // Function to get options for the features dropdown, excluding the selected outcome
  const getFeatureOptions = () =>
    availableFeatures
      .filter(
        (feature) =>
          feature !== selectedOutcome && feature !== selectedTreatment,
      )
      .map((feature) => (
        <option key={feature} value={feature}>
          {feature}
        </option>
      ));

  const toggleModelParamsVisibility = () => {
    setIsModelParamsVisible((prev) => !prev);
  };

  return (
    <>
      <div className="w-full flex flex-col justify-center items-center h-[calc(100vh_-_50px)]">
        <div className="w-full flex flex-col justify-start items-start h-full overflow-y-scroll">
          <div className="w-full h-[60px] px-4 flex flex-row justify-between items-center">
            <div className="flex flex-row justify-start items-center space-x-1">
              <FaRegFolderOpen className="w-[26px] h-[26px]" />

              {!isDataModalOpen ? (
                <div className="flex flex-row justify-start items-center space-x-4">
                  <div className="h-full flex flex-row justify-start items-center space-x-1 text-[1.4em]">
                    <div>Workspace</div>
                    <div className="text-gray-400">/</div>
                    <div>Datasets</div>
                  </div>
                  <div className="flex justify-start items-center space-x-1">
                    <div>Head/tail rows:</div>
                    <select
                      className="rounded-md px-1.5 py-1 text-sm font-bold border"
                      value={headTailRows}
                      onChange={(e) =>
                        setHeadTailRows(parseInt(e.target.value, 10))
                      }
                    >
                      <option value={20}>20</option>
                      <option value={100}>100</option>
                      <option value={200}>200</option>
                      <option value={500}>500</option>
                      <option value={1000}>1000</option>
                    </select>
                  </div>
                </div>
              ) : (
                <div className="flex flex-row justify-between items-center">
                  <div className="flex flex-row justify-start items-center space-x-4">
                    <div className="h-full flex flex-row justify-start items-center space-x-1 text-[1.4em]">
                      <div>Workspace</div>
                      <div className="text-gray-400">/</div>
                      <div>Datasets</div>
                      <div className="text-gray-400">/</div>
                      <div>{selectedFileName}</div>
                    </div>

                    <button
                      onClick={() => setIsDataModalOpen(false)}
                      className="bg-gray-100 border rounded-md px-1.5 py-0.5 text-sm font-bold"
                    >
                      Close
                    </button>

                    {isDataModalOpen && observationsRemoved > 0 && (
                      <div className="bg-orange-400 flex justify-center items-center space-x-1 rounded-md px-1.5 py-0.5 text-sm text-white font-bold">
                        <BiSolidError />
                        <div>{observationsRemoved}</div>
                      </div>
                    )}

                    <div id="analyzeTime">
                      {isAnalyzing ? (
                        <span>
                          Training: {displayElapsedTime()} have elapsed
                        </span>
                      ) : (
                        totalElapsedTime && (
                          <span>Total training time: {totalElapsedTime}</span>
                        )
                      )}
                    </div>
                  </div>
                </div>
              )}
            </div>

            {!isDataModalOpen ? (
              <div className="w-auto flex flex-row space-x-1">
                <span>Local path:</span>
                <code className="bg-gray-100 p-1 rounded-md text-sm">
                  {`"${workspacePath}"`}
                </code>
              </div>
            ) : (
              <div className="flex flex-row justify-start items-center space-x-2">
                {apiStatus !== 'Online' && (
                  <div id="apiStatus">
                    API Status:{' '}
                    <span className="bg-gray-100 rounded-full px-2 py-1 text-sm font-semibold">
                      {apiStatus}
                    </span>
                  </div>
                )}

                {apiStatus === 'Online' &&
                  !isAnalyzing &&
                  !totalElapsedTime && (
                    <div className="flex justify-start items-center space-x-4">
                      <div className="flex justify-start items-center space-x-1">
                        <div>Outcome (y):</div>
                        <select
                          className="rounded-md px-1.5 py-1 text-sm font-bold border"
                          value={selectedOutcome}
                          onChange={handleOutcomeChange}
                        >
                          <option value="Please select">Please select</option>
                          {getDropdownOptions([
                            selectedTreatment,
                            selectedFeature,
                          ])}
                        </select>
                      </div>

                      <div className="flex justify-start items-center space-x-1">
                        <div>Features (X):</div>
                        <div>
                          <select
                            className="rounded-md px-1.5 py-1 text-sm font-bold border"
                            value={selectedFeature}
                            onChange={(e) => setSelectedFeature(e.target.value)}
                          >
                            {getFeatureOptions()}
                          </select>
                        </div>
                      </div>

                      {(selectedModel === 'bcf' ||
                        selectedModel === 'xbcf') && (
                        <div className="flex justify-start items-center space-x-1">
                          <div>Treatment (Z):</div>
                          <select
                            className="rounded-md px-1.5 py-1 text-sm font-bold border"
                            value={selectedTreatment}
                            onChange={handleTreatmentChange}
                          >
                            <option value="Please select">Please select</option>
                            {getDropdownOptions([
                              selectedOutcome,
                              selectedFeature,
                            ])}
                          </select>
                        </div>
                      )}

                      <div className="flex justify-start items-center">
                        <div className="flex flex-row justify-start items-center space-x-2">
                          <div className="">Model:</div>
                          <select
                            className="rounded-md px-1.5 py-1 text-sm font-bold border"
                            value={selectedModel}
                            onChange={handleChange}
                          >
                            <option value="bart">BART</option>
                            <option value="xbart">XBART</option>
                            <option value="bart-warm-start">
                              BART (Warm-start)
                            </option>
                            <option value="bcf">BCF</option>
                            <option value="xbcf">XBCF</option>
                          </select>
                        </div>

                        <VscSettings
                          className="w-[24px] h-[24px] mx-2 text-blue-600 cursor-pointer"
                          onClick={toggleModelParamsVisibility}
                        />

                        <button
                          onClick={() => {
                            analyzeData(
                              selectedFileName,
                              selectedColumns,
                              selectedOutcome,
                            );
                            switchAction('analyze');
                          }}
                          className="rounded-md px-1.5 py-1 text-sm font-bold bg-red-600 text-white"
                        >
                          Train Model
                        </button>
                      </div>
                    </div>
                  )}

                {!isAnalyzing && totalElapsedTime && (
                  <div className="flex space-x-4">
                    <button
                      onClick={downloadCSV}
                      className="rounded-md px-1.5 py-1 text-sm font-bold border bg-blue-600 text-white"
                    >
                      Save Trained Model
                    </button>

                    <button
                      onClick={downloadCSV}
                      className="rounded-md px-1.5 py-1 text-sm font-bold border bg-blue-600 text-white"
                    >
                      Download Posterior Matrices
                    </button>
                  </div>
                )}
              </div>
            )}
          </div>

          <div className="w-full">
            {files.map((file, index) => (
              <div
                key={file.name}
                onDoubleClick={
                  apiStatus === 'Online'
                    ? () => handleFileDoubleClick(file.name)
                    : undefined
                }
                className={`border-b border-b-gray-200 py-2 w-full px-4 flex flex-row justify-between items-center hover:bg-gray-200/60 ${
                  index % 2 === 0 ? 'bg-gray-100/40' : 'bg-white'
                } ${index === 0 ? 'border-t border-t-gray-200' : ''} ${
                  apiStatus === 'Online' && 'cursor-pointer'
                }`}
              >
                <div className="flex flex-row justify-start items-center space-x-1">
                  <MdOutlineInsertDriveFile className="w-[20px] h-[20px] text-gray-400" />
                  <div className="font-medium">{file.name}</div>
                </div>
                <div className="flex flex-row justify-start items-center space-x-4">
                  <div className="w-[100px] text-right text-sm font-medium text-gray-600">
                    {formatFileSize(file.size)}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      <div
        id="dropzone"
        className="bg-red-600 text-white fixed bottom-10 left-1/2 transform -translate-x-1/2 flex flex-col justify-center items-center border w-[380px] h-[90px] rounded-full z-10 shadow-lg"
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

      <div
        id="dataModal"
        className={`fixed top-[110px] left-0 w-full h-[calc(100vh_-_110px)] bg-white z-20 ${
          isDataModalOpen ? '' : 'hidden'
        }`}
      >
        <div className="w-full h-[calc(100vh_-_110px)] overflow-auto">
          {selectedFileData.trim() ? (
            renderTableFromJsonData()
          ) : (
            <div>Select a file to start.</div>
          )}
        </div>
      </div>

      <div className="absolute bottom-0 left-0 text-[0.725em] text-gray-400 z-10 text-center px-2 py-1 rounded-r-md">
        {apiStatus}
      </div>

      <button
        className="absolute bottom-0 right-0 text-[0.725em] z-10 text-center px-2 py-1 rounded-r-md"
        onClick={() =>
          window.electron.invoke('show-notification', {
            title: 'Model Training Complete',
            body: 'Save your model to generate predictions!',
          })
        }
      >
        Show Notification
      </button>

      {isModelParamsVisible && (
        <div
          id="modelParams"
          className="w-[400px] bg-blue-600 text-white p-2 rounded-b-md absolute top-[110px] right-[30px] z-20 shadow-md"
        >
          <div>
            <div className="uppercase font-semibold text-[0.925em] border-b border-b-white/30 py-1 text-center">
              Model Parameters
            </div>

            <div className="w-full flex flex-col justify-start items-center">
              <div className="w-full flex justify-between items-center border-b border-b-white/30 py-1">
                <div>Number of Trees (M)</div>
                <input
                  type="number"
                  className="w-[100px] text-black p-1"
                  defaultValue="100"
                />
              </div>
              <div className="w-full flex justify-between items-center border-b border-b-white/30 py-1">
                <div>Burn-in Iterations</div>
                <input
                  type="number"
                  className="w-[100px] text-black p-1"
                  defaultValue="1000"
                />
              </div>
              <div className="w-full flex justify-between items-center border-b border-b-white/30 py-1">
                <div>Number of Draws</div>
                <input
                  type="number"
                  className="w-[100px] text-black p-1"
                  defaultValue="5000"
                />
              </div>
              <div className="w-full flex justify-between items-center border-b border-b-white/30 py-1">
                <div>Thinning</div>
                <input
                  type="number"
                  className="w-[100px] text-black p-1"
                  defaultValue="1"
                />
              </div>
              <div className="w-full flex justify-between items-center border-b border-b-white/30 py-1">
                <div>Prior Mean of Leaf Parameters</div>
                <input
                  type="number"
                  className="w-[100px] text-black p-1"
                  defaultValue="0"
                />
              </div>
              <div className="w-full flex justify-between items-center border-b border-b-white/30 py-1">
                <div>Prior Variance of Leaf Parameters</div>
                <input
                  type="number"
                  className="w-[100px] text-black p-1"
                  defaultValue="1"
                />
              </div>
              <div className="w-full flex justify-between items-center border-b border-b-white/30 py-1">
                <div>Alpha (α) for Tree Structure Prior</div>
                <input
                  type="number"
                  className="w-[100px] text-black p-1"
                  defaultValue="0.95"
                />
              </div>
              <div className="w-full flex justify-between items-center border-b border-b-white/30 py-1">
                <div>Beta (β) for Tree Structure Prior</div>
                <input
                  type="number"
                  className="w-[100px] text-black p-1"
                  defaultValue="2"
                />
              </div>
              <div className="w-full flex justify-between items-center border-b border-b-white/30 py-1">
                <div>Tree Depth (D)</div>
                <input
                  type="number"
                  className="w-[100px] text-black p-1"
                  defaultValue="3"
                />
              </div>
              <div className="w-full flex justify-between items-center py-1">
                <div>Minimum Node Size</div>
                <input
                  type="number"
                  className="w-[100px] text-black p-1"
                  defaultValue="5"
                />
              </div>
            </div>
          </div>
        </div>
      )}
    </>
  );
};

export default Workspace;
