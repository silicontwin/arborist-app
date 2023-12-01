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
    return <div>Loading...</div>;
  }

  if (error) {
    return <div>Error: {error}</div>;
  }

  return (
    <div>
      <h1>Data from IPC:</h1>
      <pre>{JSON.stringify(data, null, 2)}</pre>
    </div>
  );
};

export default Homepage;
