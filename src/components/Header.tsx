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
      className="h-[50px] text-sm flex flex-row justify-between bg-[#242424] w-full px-5"
    >
      <div className="h-full flex flex-row justify-center items-center space-x-2">
        <div className="uppercase font-semibold">Arborist</div>
        <div className="font-light text-white/70">prototype v.0.1</div>
      </div>

      <div className="h-full flex flex-row justify-start items-center space-x-4 uppercase font-semibold">
        {/* Add noDragStyle to interactive elements */}
        <Link
          to="/"
          style={noDragStyle}
          className={`${location.pathname === '/' ? 'text-yellow-300' : ''}`}
        >
          Workspace
        </Link>

        <Link
          to="/about"
          style={noDragStyle}
          className={`${
            location.pathname === '/about' ? 'text-yellow-300' : ''
          }`}
        >
          About
        </Link>
      </div>
    </div>
  );
};

export default Header;
