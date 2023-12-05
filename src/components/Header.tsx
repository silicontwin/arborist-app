// /src/components/Header.tsx
import React from 'react';
import { Link } from 'react-router-dom';

const Header = () => {
  // Define the styles for the draggable area
  const dragStyle: React.CSSProperties = { WebkitAppRegion: 'drag' };

  // Define the styles for the non-draggable elements (like buttons and links)
  const noDragStyle: React.CSSProperties = { WebkitAppRegion: 'no-drag' };

  // bg-[#bf5700]

  return (
    <div
      style={dragStyle}
      className="h-[50px] text-white/70 text-sm flex flex-row justify-between uppercase font-semibold bg-[#242424] w-full px-5"
    >
      <div className="h-full flex flex-col justify-center items-center">
        Arborist
      </div>

      <div className="h-full flex flex-row justify-start items-center space-x-4">
        {/* Add noDragStyle to interactive elements */}
        <Link to="/" style={noDragStyle}>
          Home
        </Link>
        <Link to="/about" style={noDragStyle}>
          About
        </Link>
      </div>
    </div>
  );
};

export default Header;
