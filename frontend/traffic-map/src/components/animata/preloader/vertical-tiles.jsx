import React, { useCallback, useEffect, useRef, useState } from "react";
import { motion, useInView } from "framer-motion";
import { cn } from "../../../lib/utils";

export default function VerticalTiles({
  tileClassName,
  minTileWidth = 32,
  animationDuration = 0.5,
  animationDelay = 0.2,
  stagger = 0.05,
  children,
}) {
  const [tiles, setTiles] = useState([]);
  const [shouldAnimate, setShouldAnimate] = useState(false);
  const containerRef = useRef(null);
  const isInView = useInView(containerRef, { once: true, amount: 0.1 });

  // Trigger animation immediately on mount
  useEffect(() => {
    const timer = setTimeout(() => setShouldAnimate(true), 100);
    return () => clearTimeout(timer);
  }, []);

  const calculateTiles = useCallback(() => {
    if (containerRef.current) {
      const { offsetWidth: width } = containerRef.current;
      const tileCount = Math.max(3, Math.floor(width / minTileWidth));
      const tileWidth = width / tileCount + 1;

      const newTiles = Array.from({ length: tileCount }, (_, index) => ({
        id: index,
        width: tileWidth,
        order: Math.abs(index - Math.floor((tileCount - 1) / 2)),
      }));

      setTiles(newTiles);
    }
  }, [minTileWidth]);

  useEffect(() => {
    calculateTiles();
    const resizeObserver = new ResizeObserver(calculateTiles);
    if (containerRef.current) {
      resizeObserver.observe(containerRef.current);
    }
    return () => resizeObserver.disconnect();
  }, [calculateTiles]);

  return (
    <div 
      ref={containerRef} 
      className="relative overflow-hidden"
      style={{
        position: 'relative',
        overflow: 'hidden',
        width: '100%',
        height: '100%'
      }}
    >
      <div style={{ 
        position: 'relative', 
        zIndex: 1, 
        width: '100%', 
        height: '100%',
        pointerEvents: 'none'
      }}>
        <div style={{ pointerEvents: 'auto', width: '100%', height: '100%' }}>
          {children}
        </div>
      </div>

      <div 
        className="absolute inset-0 flex"
        style={{
          position: 'absolute',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          display: 'flex',
          zIndex: 20,
          pointerEvents: 'none'
        }}
      >
        {tiles.map((tile) => (
          <motion.div
            key={tile.id}
            className={cn("bg-gray-800", tileClassName)}
            style={{
              width: tile.width,
              position: "absolute",
              left: `${(tile.id * 100) / tiles.length}%`,
              top: 0,
              height: "100%",
              backgroundColor: tileClassName === "bg-black" ? "#1a1a1a" : 
                              tileClassName === "bg-gray-900" ? "#374151" : "#4b5563",
              zIndex: 20,
              borderRight: '1px solid rgba(255, 255, 255, 0.05)',
              pointerEvents: 'none'
            }}
            initial={{ y: 0 }}
            animate={(isInView || shouldAnimate) ? { y: "100%" } : { y: 0 }}
            transition={{
              duration: animationDuration,
              delay: animationDelay + tile.order * stagger,
              ease: [0.45, 0, 0.55, 1],
            }}
          />
        ))}
      </div>
    </div>
  );
}

