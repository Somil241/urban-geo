import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import VerticalTiles from './animata/preloader/vertical-tiles';
import { TextHoverEffect } from './animata/preloader/text-hover';

const SplashScreen = ({ onStart }) => {
  const [showContent, setShowContent] = useState(false);
  const [showButton, setShowButton] = useState(false);
  const [isExiting, setIsExiting] = useState(false);

  React.useEffect(() => {
    // Show content after tiles start dropping
    const timer1 = setTimeout(() => setShowContent(true), 500);
    // Show button after text animation
    const timer2 = setTimeout(() => setShowButton(true), 3000);
    
    return () => {
      clearTimeout(timer1);
      clearTimeout(timer2);
    };
  }, []);

  const handleStart = () => {
    setIsExiting(true);
    setTimeout(() => {
      onStart();
    }, 600);
  };

  return (
    <motion.div 
      className="fixed inset-0 z-50 bg-black overflow-hidden"
      style={{ 
        position: 'fixed',
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
        zIndex: 50,
        backgroundColor: '#000000',
        overflow: 'hidden'
      }}
      initial={{ opacity: 1 }}
      animate={{ opacity: isExiting ? 0 : 1 }}
      transition={{ duration: 0.6, ease: "easeInOut" }}
    >
      <VerticalTiles
        tileClassName="bg-gray-900"
        minTileWidth={40}
        animationDuration={1.2}
        animationDelay={0.2}
        stagger={0.04}
      >
        <div 
          className="flex flex-col items-center justify-center h-full w-full relative"
          style={{
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
            height: '100%',
            width: '100%',
            position: 'relative'
          }}
        >
          <AnimatePresence>
            {showContent && !isExiting && (
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20, scale: 0.95 }}
                transition={{ duration: 0.8, ease: "easeOut" }}
                className="flex flex-col items-center justify-center w-full h-full"
                style={{
                  display: 'flex',
                  flexDirection: 'column',
                  alignItems: 'center',
                  justifyContent: 'center',
                  width: '100%',
                  height: '100%',
                  position: 'relative',
                  zIndex: 100
                }}
              >
                <div 
                  className="w-full max-w-4xl h-64 mb-12"
                  style={{
                    width: '100%',
                    maxWidth: '56rem',
                    height: '16rem',
                    marginBottom: '3rem',
                    position: 'relative',
                    zIndex: 100,
                    pointerEvents: 'auto'
                  }}
                >
                  <TextHoverEffect text="Urban Geo" duration={0.3} />
                </div>

                <AnimatePresence>
                  {showButton && (
                    <motion.button
                      type="button"
                      initial={{ opacity: 0, scale: 0.8 }}
                      animate={{ opacity: 1, scale: 1 }}
                      exit={{ opacity: 0, scale: 0.8 }}
                      transition={{ 
                        duration: 0.5, 
                        delay: 0.2,
                        type: "spring",
                        stiffness: 200,
                        damping: 15
                      }}
                      onClick={handleStart}
                      className="px-8 py-4 bg-gradient-to-r from-emerald-500 to-emerald-600 text-white font-semibold text-lg rounded-xl shadow-lg shadow-emerald-500/50 hover:from-emerald-400 hover:to-emerald-500 hover:shadow-xl hover:shadow-emerald-500/60 active:scale-95 transition-all duration-300 backdrop-blur-sm border border-emerald-400/30"
                      style={{
                        padding: '1rem 2rem',
                        background: 'linear-gradient(to right, #10b981, #059669)',
                        color: '#ffffff',
                        fontWeight: 600,
                        fontSize: '1.125rem',
                        borderRadius: '0.75rem',
                        boxShadow: '0 10px 15px -3px rgba(16, 185, 129, 0.5)',
                        border: '1px solid rgba(16, 185, 129, 0.3)',
                        cursor: 'pointer',
                        transition: 'all 0.3s',
                        position: 'relative',
                        zIndex: 1000,
                        pointerEvents: 'auto',
                        WebkitTapHighlightColor: 'transparent'
                      }}
                    >
                      Start
                    </motion.button>
                  )}
                </AnimatePresence>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </VerticalTiles>
    </motion.div>
  );
};

export default SplashScreen;

