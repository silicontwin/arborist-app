// /src/pages/Homepage.tsx
import React, { useEffect, useState } from 'react';

interface FetchDataResponse {
  status: string;
}

const Homepage = () => {
  const [data, setData] = useState<FetchDataResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    window.electron
      .fetchData()
      .then((fetchedData: FetchDataResponse) => {
        setData(fetchedData);
        setLoading(false);
      })
      .catch((fetchError) => {
        console.error('Error in fetchData:', fetchError);
        setError('Failed to fetch data');
        setLoading(false);
      });
  }, []);

  if (loading) {
    return (
      <div className="w-full h-full flex flex-col justify-center items-center font-bold">
        <div>Initializing Analytics Engine . . .</div>
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

  return (
    <div className="w-full h-full flex flex-col justify-between items-start">
      <div className="w-full h-full p-4 flex flex-col justify-center items-center">
        Drag SPSS file here
      </div>

      <pre className="w-full overflow-x-auto whitespace-pre-wrap text-left text-xs border bg-white rounded-md p-4">
        <code>{JSON.stringify(data, null, 2)}</code>
      </pre>

      <div className="h-[50px] w-full bg-gray-200 flex flex-row justify-start items-center px-4">
        API Status: {data?.status}
      </div>
    </div>
  );
};

export default Homepage;
