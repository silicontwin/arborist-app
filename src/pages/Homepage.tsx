// /src/pages/Homepage.tsx
import React, { useEffect, useState } from 'react';

const Homepage = () => {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    window.electron
      .fetchData()
      .then((fetchedData) => {
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
    <div className="w-full h-full flex flex-col justify-center items-start p-10 space-y-4">
      <h1 className="font-bold">Data from FastAPI:</h1>

      <pre className="w-full overflow-x-auto whitespace-pre-wrap text-left text-xs border bg-white rounded-md p-4">
        <code>{JSON.stringify(data, null, 2)}</code>
      </pre>
    </div>
  );
};

export default Homepage;
