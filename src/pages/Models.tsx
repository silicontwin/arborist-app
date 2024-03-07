import React from 'react';

const Models = () => {
  return (
    <div className="px-5 py-20 w-full flex flex-col justify-center items-center h-[calc(100vh_-_50px)]">
      <div className="w-[760px] flex flex-col justify-start items-start space-y-8">
        <div className="text-2xl">Model Cards</div>

        <div className="flex flex-col space-y-4">
          A collection of your saved models.
        </div>
      </div>
    </div>
  );
};

export default Models;
