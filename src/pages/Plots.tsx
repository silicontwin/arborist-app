// src/pages/Plots.tsx
import React, { useEffect, useState } from 'react';

const Plots = () => {
  const [plotSrc, setPlotSrc] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(true);

  // useEffect(() => {
  //   const fetchPlot = async () => {
  //     setIsLoading(true);
  //     try {
  //       const plotData = await window.electron.invoke('fetch-plot');
  //       const formattedSrc = `data:image/png;base64,${plotData}`;
  //       setPlotSrc(formattedSrc);
  //     } catch (error) {
  //       console.error('Failed to fetch plot:', error);
  //     } finally {
  //       setIsLoading(false);
  //     }
  //   };

  //   fetchPlot();
  // }, []);

  return (
    <div className="px-5 py-20 w-full flex flex-col justify-start items-center h-[calc(100vh_-_50px)]">
      <div className="w-[760px] flex flex-col justify-start items-start space-y-8">
        <div className="text-2xl">Plots</div>
        {/* {isLoading ? (
          <div>Loading plot...</div>
        ) : plotSrc ? (
          <img src={plotSrc} alt="Plot" />
        ) : (
          <div>Plot not available</div>
        )} */}
      </div>
    </div>
  );
};

export default Plots;
