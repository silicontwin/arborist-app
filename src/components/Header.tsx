// /src/components/Header.tsx
import React from 'react';
import { Link } from 'react-router-dom';
import { useLocation } from 'react-router-dom';
import { FaRegFolderOpen } from 'react-icons/fa';
import { AiOutlineBoxPlot } from 'react-icons/ai';
import { HiOutlineCollection } from 'react-icons/hi';
import { TbArrowsMaximize, TbArrowsDiagonalMinimize2 } from 'react-icons/tb';
import logo from '../images/assets/logo.png';

const Header = () => {
  const dragStyle: React.CSSProperties = { WebkitAppRegion: 'drag' };
  const noDragStyle: React.CSSProperties = { WebkitAppRegion: 'no-drag' };
  const location = useLocation();

  const handleMaximizeClick = () => {
    window.electron.maximizeWindow();
  };

  const handleMinimizeClick = () => {
    window.electron.minimizeWindow();
  };

  return (
    <div
      style={dragStyle}
      className="bg-gray-100 border-b h-[50px] flex justify-between items-center w-full px-4"
    >
      <div className="flex items-center justify-start flex-1">
        <img
          src={logo}
          alt="Arborist"
          className="w-[316.67px] h-[50px] mb-[1px]"
        />
        <div className="w-[580px] font-light text-[0.825em] text-blue-700 opacity-40 pt-[2px] flex space-x-2">
          <div>V 0.2</div>
          <div>prototype</div>
          <div>not for use with real data</div>
        </div>
      </div>

      <div
        className="h-full flex-1 flex justify-center items-center font-semibold"
        style={noDragStyle}
      >
        <Link
          to="/"
          className={`h-full flex items-center space-x-1 px-4 border-x ${
            location.pathname === '/' ? 'bg-white text-red-600' : ''
          }`}
        >
          <FaRegFolderOpen className="w-[20px] h-[20px]" />
          <div>Datasets</div>
        </Link>
        <Link
          to="/models"
          className={`h-full flex items-center space-x-1 px-4 ${
            location.pathname === '/models' ? 'bg-white text-red-600' : ''
          }`}
        >
          <HiOutlineCollection className="w-[20px] h-[20px]" />
          <div>Models</div>
        </Link>
        <Link
          to="/plots"
          className={`h-full flex items-center space-x-1 px-4 border-x ${
            location.pathname === '/plots' ? 'bg-white text-red-600' : ''
          }`}
        >
          <AiOutlineBoxPlot className="w-[24px] h-[24px]" />
          <div>Plots</div>
        </Link>
      </div>

      <div
        style={dragStyle}
        className="flex items-center space-x-2 flex-1 justify-end"
      >
        <div
          onClick={handleMaximizeClick}
          className="cursor-pointer"
          style={noDragStyle}
        >
          <TbArrowsMaximize className="w-[20px] h-[20px]" />
        </div>
        <div
          onClick={handleMinimizeClick}
          className="cursor-pointer"
          style={noDragStyle}
        >
          <TbArrowsDiagonalMinimize2 className="w-[20px] h-[20px]" />
        </div>
      </div>
    </div>
  );
};

export default Header;
