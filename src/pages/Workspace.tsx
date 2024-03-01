import React, { useEffect, useState, useMemo } from 'react';
import { FileDetails } from '../types/fileDetails';
import { FaRegFolderOpen } from 'react-icons/fa';
import { MdOutlineInsertDriveFile } from 'react-icons/md';
import { TbDragDrop } from 'react-icons/tb';
import { VscSettings } from 'react-icons/vsc';

interface DataItem {
  [key: string]: any;
}

interface ColumnStatus {
  [key: string]: { isNumeric: boolean; isChecked: boolean };
}

const Workspace = () => {
  const [files, setFiles] = useState<FileDetails[]>([]);
  const [workspacePath, setWorkspacePath] = useState('');
  const [uploadError, setUploadError] = useState('');
  const [selectedFileData, setSelectedFileData] = useState('');
  const [selectedFileName, setSelectedFileName] = useState('');
  const [isDataModalOpen, setIsDataModalOpen] = useState(false);
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
  const [selectedModel, setSelectedModel] = useState('xbart');
  const [columnNumericStatus, setColumnNumericStatus] = useState<ColumnStatus>(
    {},
  );
  const [selectedFeature, setSelectedFeature] = useState<string>('');
  const [headTailRows, setHeadTailRows] = useState<number>(100);
  const [isModelParamsVisible, setIsModelParamsVisible] =
    useState<boolean>(false);

  // Memoize the selected features computation
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

      const response = await fetch('http://localhost:8000/summarize', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          fileName: fileName,
          workspacePath: workspacePath,
          headTailRows: headTailRows,
          action: 'summarize',
        }),
      });

      if (!response.ok) {
        throw new Error(`Failed to process file: ${response.statusText}`);
      }

      const data = await response.json();
      setSelectedFileData(JSON.stringify(data.data, null, 2));

      // Ensure data.is_numeric exists and is an object before trying to use Object.keys
      if (data.is_numeric && typeof data.is_numeric === 'object') {
        const numericStatusUpdates: ColumnStatus = {};
        Object.keys(data.is_numeric).forEach((column) => {
          numericStatusUpdates[column] = {
            isNumeric: data.is_numeric[column],
            isChecked: true,
          }; // Default numeric columns to checked
        });
        setColumnNumericStatus(numericStatusUpdates);
      } else {
        console.error('Invalid or missing is_numeric data:', data);
      }

      setIsDataModalOpen(true);
      setSelectedFileName(fileName);

      // Reset the observationSelection state to its default value
      setObservationSelection('all');
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
      setColumnNumericStatus((prevState) => {
        const updatedStatus = {
          ...prevState,
          [columnName]: {
            ...prevState[columnName],
            isChecked: !prevState[columnName].isChecked,
          },
        };

        // Find the first checked and numeric column to set as the selected feature
        const firstCheckedColumn = Object.keys(updatedStatus).find(
          (key) => updatedStatus[key].isChecked && updatedStatus[key].isNumeric,
        );

        // Update the selected feature state
        setSelectedFeature(firstCheckedColumn || '');

        return updatedStatus;
      });
    };

    return (
      <table className="min-w-full text-sm border-collapse">
        <thead>
          <tr>
            {columns.map((column, index) => (
              <th
                key={index}
                className={`border py-2 px-4 bg-white font-bold text-left uppercase text-[.925em] whitespace-nowrap ${
                  !columnNumericStatus[column]?.isChecked ||
                  !columnNumericStatus[column]?.isNumeric
                    ? 'text-gray-400'
                    : 'text-black'
                }`}
              >
                {columnNumericStatus[column]?.isNumeric ? (
                  <label className="flex flex-row justify-start items-center space-x-1 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={columnNumericStatus[column]?.isChecked}
                      onChange={() => handleCheckboxChange(column)}
                      className="form-checkbox"
                    />
                    <span>{column}</span>
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
                    !columnNumericStatus[column]?.isChecked ||
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

  // ---

  const analyzeData = async (fileName: string) => {
    const selectedColumns = Object.entries(columnNumericStatus)
      .filter(([_, value]) => value.isChecked && value.isNumeric)
      .map(([key]) => key);

    // Ensure there are selected features before proceeding
    if (selectedColumns.length === 0) {
      alert('Please select at least one numeric column to analyze.');
      return;
    }

    console.log('Starting analysis for:', fileName);

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
          fileName: fileName,
          workspacePath: workspacePath,
          headTailRows: headTailRows,
          selectedColumns: selectedColumns,
          action: 'analyze',
        }),
      });

      if (!response.ok) {
        const responseBody = await response.text();
        throw new Error(
          `Failed to analyze file: ${response.statusText} - ${responseBody}`,
        );
      }

      const data = await response.json();
      console.log('Analysis data received:', data);
      setSelectedFileData(JSON.stringify(data.data, null, 2));

      if (data.predictions) {
        setPredictions(data.predictions);
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

  const getFeatureOptions = () => {
    // Filter columns that are numeric and checked
    const checkedColumns = Object.entries(columnNumericStatus)
      .filter(([columnName, status]) => status.isNumeric && status.isChecked)
      .map(([columnName]) => columnName);

    // Generate <option> elements for these columns
    return checkedColumns.map((columnName) => (
      <option key={columnName} value={columnName}>
        {columnName}
      </option>
    ));
  };

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
                <div className="flex justify-start items-center space-x-4">
                  <div className="text-[1.4em]">Workspace</div>
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
                  <div className="flex flex-row justify-start items-center space-x-3">
                    <div className="h-full flex flex-row justify-start items-center space-x-1 text-[1.4em]">
                      <div>Workspace</div>
                      <div className="text-gray-400">/</div>
                      <div>{selectedFileName}</div>
                    </div>

                    <button
                      onClick={() => setIsDataModalOpen(false)}
                      className="bg-gray-100 border rounded-md px-1.5 py-0.5 text-sm font-bold"
                    >
                      Close
                    </button>

                    <div id="analyzeTime">
                      {isAnalyzing ? (
                        <span>
                          Analyzing: {displayElapsedTime()} have elapsed
                        </span>
                      ) : (
                        totalElapsedTime && (
                          <span>Total analysis time: {totalElapsedTime}</span>
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

                {apiStatus === 'Online' && !predictions && (
                  <div className="flex justify-start items-center space-x-8">
                    <div className="flex justify-start items-center space-x-1">
                      <div>Features:</div>
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
                    {/* <div className="flex justify-start items-center space-x-1">
                      <div>Observations:</div>
                      <div>
                        <select
                          className="rounded-md px-1.5 py-1 text-sm font-bold border"
                          value={observationSelection}
                          onChange={(e) =>
                            setObservationSelection(e.target.value)
                          }
                        >
                          <option value="all">All</option>
                          <option value="some">Some</option>
                        </select>
                      </div>
                      {observationSelection === 'some' && (
                        <>
                          <div>
                            From:{' '}
                            <input
                              type="number"
                              className="w-24 rounded-md px-1.5 py-1 text-sm font-bold border"
                              placeholder="Row #"
                            />
                          </div>
                          <div>
                            To:{' '}
                            <input
                              type="number"
                              className="w-24 rounded-md px-1.5 py-1 text-sm font-bold border"
                              placeholder="Row #"
                            />
                          </div>
                        </>
                      )}
                    </div> */}

                    <div className="flex justify-start items-center">
                      <div className="flex flex-row justify-start items-center space-x-2">
                        <div className="">Model:</div>
                        <select
                          className="rounded-md px-1.5 py-1 text-sm font-bold border"
                          value={selectedModel}
                          onChange={handleChange}
                        >
                          <option value="bcf">BCF</option>
                          <option value="bart">BART</option>
                          <option value="xbart">XBART</option>
                        </select>
                      </div>

                      <VscSettings
                        className="w-[24px] h-[24px] mx-2 text-blue-600 cursor-pointer"
                        onClick={toggleModelParamsVisibility}
                      />

                      <button
                        onClick={() => analyzeData(selectedFileName)}
                        className="rounded-md px-1.5 py-1 text-sm font-bold bg-red-600 text-white"
                      >
                        Analyze
                      </button>
                    </div>
                  </div>
                )}

                {predictions && (
                  <div className="flex space-x-4">
                    <button
                      onClick={downloadCSV}
                      className="rounded-md px-1.5 py-1 text-sm font-bold border bg-blue-600 text-white"
                    >
                      Download Analysis
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
        className="bg-gray-100 fixed bottom-10 left-1/2 transform -translate-x-1/2 flex flex-col justify-center items-center border w-[380px] h-[90px] rounded-full z-10"
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
