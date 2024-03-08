// /src/components/Header.tsx
import React from 'react';
import { Link, Router } from 'react-router-dom';
import { useLocation } from 'react-router-dom';
import { FaRegFolderOpen } from 'react-icons/fa';
import { AiOutlineBoxPlot } from 'react-icons/ai';
import { HiMiniRocketLaunch } from 'react-icons/hi2';
import { HiOutlineCollection } from 'react-icons/hi';
import { TbArrowsMaximize, TbArrowsDiagonalMinimize2 } from 'react-icons/tb';

const Header = () => {
  // Define the styles for the draggable area
  const dragStyle: React.CSSProperties = { WebkitAppRegion: 'drag' };

  // Define the styles for the non-draggable elements (like buttons and links)
  const noDragStyle: React.CSSProperties = { WebkitAppRegion: 'no-drag' };

  // Get the current location object
  const location = useLocation();

  // Log the current path
  // console.log('Current path:', location.pathname);

  return (
    <div
      style={dragStyle}
      className="bg-gray-100 border-b h-[50px] flex flex-row justify-between w-full px-4"
    >
      <div className="h-full flex flex-row justify-center items-center space-x-3">
        <div className="font-bold text-[1.2em] text-blue-700">Arborist</div>
        <div className="font-light text-[0.825em] opacity-40 pt-[2px] flex flex-row space-x-2">
          <div>v.0.2</div>
          <div>prototype</div>
          <div>not for use with real data</div>
        </div>
      </div>

      <div className="h-full flex flex-row justify-start items-center space-x-5 font-semibold">
        {/* Add noDragStyle to interactive elements */}
        <Link
          to="/"
          style={noDragStyle}
          className={`${location.pathname === '/' ? 'text-red-600' : ''}`}
        >
          <div className="flex flex-row justify-start items-center space-x-1">
            <FaRegFolderOpen className="w-[20px] h-[20px]" />
            <div>Datasets</div>
          </div>
        </Link>

        <Link
          to="/models"
          style={noDragStyle}
          className={`${location.pathname === '/models' ? 'text-red-600' : ''}`}
        >
          <div className="flex flex-row justify-start items-center space-x-1">
            <HiOutlineCollection className="w-[20px] h-[20px]" />
            <div>Models</div>
          </div>
        </Link>

        <Link
          to="/plots"
          style={noDragStyle}
          className={`${location.pathname === '/plots' ? 'text-red-600' : ''}`}
        >
          <div className="flex flex-row justify-start items-center space-x-1">
            <AiOutlineBoxPlot className="w-[24px] h-[24px]" />
            <div>Plots</div>
          </div>
        </Link>

        <TbArrowsMaximize className="w-[20px] h-[20px]" />
        <TbArrowsDiagonalMinimize2 className="w-[20px] h-[20px]" />

        {/* <Link
          to="/upload"
          style={noDragStyle}
          className={`${location.pathname === '/upload' ? 'text-red-600' : ''}`}
        >
          Upload
        </Link> */}

        {/* <Link
          to="/about"
          style={noDragStyle}
          className={`${location.pathname === '/about' ? 'text-red-600' : ''}`}
        >
          <div className="flex flex-row justify-start items-center space-x-1">
            <HiMiniRocketLaunch className="w-[18px] h-[18px]" />
            <div>About</div>
          </div>
        </Link> */}
      </div>
    </div>
  );
};

export default Header;
