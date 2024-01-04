// /src/components/Header.tsx
import React from 'react';
import { Link, Router } from 'react-router-dom';
import { useLocation } from 'react-router-dom';

const Header = () => {
  // Define the styles for the draggable area
  const dragStyle: React.CSSProperties = { WebkitAppRegion: 'drag' };

  // Define the styles for the non-draggable elements (like buttons and links)
  const noDragStyle: React.CSSProperties = { WebkitAppRegion: 'no-drag' };

  // Get the current location object
  const location = useLocation();

  // Log the current path
  console.log('Current path:', location.pathname);

  return (
    <div
      style={dragStyle}
      className="bg-gray-100 border-b h-[50px] flex flex-row justify-between w-full px-5"
    >
      <div className="h-full flex flex-row justify-center items-center space-x-3">
        <div className="font-bold text-[1.2em] text-blue-700">Arborist</div>
        <div className="font-light text-[0.825em] opacity-40">
          v.0.1 | alpha prototype: not for use with real data
        </div>
      </div>

      <div className="h-full flex flex-row justify-start items-center space-x-5 font-semibold">
        {/* Add noDragStyle to interactive elements */}
        <Link
          to="/"
          style={noDragStyle}
          className={`${location.pathname === '/' ? 'text-red-600' : ''}`}
        >
          Workspace
        </Link>

        <Link
          to="/upload"
          style={noDragStyle}
          className={`${location.pathname === '/upload' ? 'text-red-600' : ''}`}
        >
          Upload
        </Link>

        <Link
          to="/about"
          style={noDragStyle}
          className={`${location.pathname === '/about' ? 'text-red-600' : ''}`}
        >
          About
        </Link>
      </div>
    </div>
  );
};

export default Header;
