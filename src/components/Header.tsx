// /src/components/Header.tsx

import React from 'react';
import { Link } from 'react-router-dom';

const Header = () => {
  // const dragStyle: any = { '-webkit-app-region': 'drag' };

  return (
    <div
      // style={dragStyle}
      className="h-[50px] text-white flex flex-row justify-between uppercase font-semibold bg-[#bf5700] w-full px-4"
    >
      <div className="h-full flex flex-col justify-center items-center">
        TxBSPI App 2
      </div>

      <div className="h-full flex flex-row justify-start items-center space-x-4">
        <Link to="/">Home</Link>
        <Link to="/about">About</Link>
      </div>
    </div>
  );
};

export default Header;
