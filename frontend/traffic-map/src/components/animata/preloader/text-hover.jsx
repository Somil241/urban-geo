import React, { useRef, useEffect, useState } from "react";
import { motion } from "framer-motion";
 
export const TextHoverEffect = ({
  text,
  duration,
}) => {
  const svgRef = useRef(null);
  const [cursor, setCursor] = useState({ x: 0, y: 0 });
  const [hovered, setHovered] = useState(false);
  const [maskPosition, setMaskPosition] = useState({ cx: "50%", cy: "50%" });
 
  useEffect(() => {
    if (svgRef.current && cursor.x !== null && cursor.y !== null) {
      const svgRect = svgRef.current.getBoundingClientRect();
      const cxPercentage = ((cursor.x - svgRect.left) / svgRect.width) * 100;
      const cyPercentage = ((cursor.y - svgRect.top) / svgRect.height) * 100;
      setMaskPosition({
        cx: `${cxPercentage}%`,
        cy: `${cyPercentage}%`,
      });
    }
  }, [cursor]);

  return (
    <div style={{ width: '100%', height: '100%', position: 'relative' }}>
      <svg
        ref={svgRef}
        width="100%"
        height="100%"
        viewBox="0 0 400 120"
        xmlns="http://www.w3.org/2000/svg"
        onMouseEnter={() => {
          setHovered(true);
        }}
        onMouseLeave={() => {
          setHovered(false);
        }}
        onMouseMove={(e) => {
          if (svgRef.current) {
            setCursor({ 
              x: e.clientX, 
              y: e.clientY 
            });
          }
        }}
        className="select-none"
        style={{
          width: '100%',
          height: '100%',
          display: 'block',
          cursor: 'pointer',
          pointerEvents: 'auto',
          position: 'relative',
          zIndex: 10
        }}
      >
      <defs>
        <linearGradient
          id="textGradient"
          gradientUnits="userSpaceOnUse"
          cx="50%"
          cy="50%"
          r="25%"
        >
          {hovered && (
            <>
              <stop offset="0%" stopColor="#eab308" />
              <stop offset="25%" stopColor="#ef4444" />
              <stop offset="50%" stopColor="#3b82f6" />
              <stop offset="75%" stopColor="#06b6d4" />
              <stop offset="100%" stopColor="#8b5cf6" />
            </>
          )}
        </linearGradient>
 
        <motion.radialGradient
          id="revealMask"
          gradientUnits="userSpaceOnUse"
          r="30%"
          initial={{ cx: "50%", cy: "50%" }}
          animate={hovered ? maskPosition : { cx: "50%", cy: "50%" }}
          transition={{ 
            duration: duration ?? 0.3, 
            ease: "easeOut",
            type: "spring",
            stiffness: 300,
            damping: 30
          }}
        >
          <stop offset="0%" stopColor="white" />
          <stop offset="100%" stopColor="black" />
        </motion.radialGradient>
        <mask id="textMask">
          <rect
            x="0"
            y="0"
            width="100%"
            height="100%"
            fill="url(#revealMask)"
          />
        </mask>
      </defs>
      <text
        x="200"
        y="60"
        textAnchor="middle"
        dominantBaseline="middle"
        fill="transparent"
        stroke="#e5e7eb"
        strokeWidth="1.5"
        fontSize="56"
        fontFamily="helvetica, Arial, sans-serif"
        fontWeight="bold"
        style={{ 
          opacity: hovered ? 0.5 : 0.3,
          pointerEvents: 'none'
        }}
      >
        {text}
      </text>
      <motion.text
        x="200"
        y="60"
        textAnchor="middle"
        dominantBaseline="middle"
        fill="transparent"
        stroke="#e5e7eb"
        strokeWidth="1.5"
        fontSize="56"
        fontFamily="helvetica, Arial, sans-serif"
        fontWeight="bold"
        style={{ pointerEvents: 'none' }}
        initial={{ strokeDashoffset: 1000, strokeDasharray: 1000 }}
        animate={{
          strokeDashoffset: 0,
          strokeDasharray: 1000,
        }}
        transition={{
          duration: 4,
          ease: "easeInOut",
        }}
      >
        {text}
      </motion.text>
      <text
        x="200"
        y="60"
        textAnchor="middle"
        dominantBaseline="middle"
        fill="transparent"
        stroke={hovered ? "url(#textGradient)" : "#e5e7eb"}
        strokeWidth={hovered ? "2" : "1.5"}
        fontSize="56"
        fontFamily="helvetica, Arial, sans-serif"
        fontWeight="bold"
        mask={hovered ? "url(#textMask)" : undefined}
        style={{ 
          pointerEvents: 'none',
          opacity: hovered ? 1 : 0.8
        }}
      >
        {text}
      </text>
      </svg>
    </div>
  );
};

