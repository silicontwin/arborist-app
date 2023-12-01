// /src/components/Header.tsx

import React from 'react';

const Header = () => {
  const dragStyle: any = { '-webkit-app-region': 'drag' };

  return (
    <div
      style={dragStyle}
      className="h-[50px] text-white flex flex-row justify-between uppercase font-semibold bg-[#bf5700] w-full px-4"
    >
      <div className="h-full flex flex-col justify-center items-center">
        TxBSPI App 2
      </div>

      <div className="h-full flex flex-col justify-center items-center">
        <div>Home</div>

        <div>About</div>
      </div>
    </div>
  );
};

export default Header;
