/**
 * =============================================================================
 * TrustLens Frontend - Application Entry Point
 * =============================================================================
 * 
 * This is the main entry point for the React application.
 * It mounts the root App component to the DOM and enables React Strict Mode
 * for highlighting potential problems in the application during development.
 * 
 * File: main.jsx
 * =============================================================================
 */

import { StrictMode } from 'react'           // React Strict Mode for development warnings
import { createRoot } from 'react-dom/client' // React 18 createRoot API for concurrent rendering
import './index.css'                          // Global CSS styles (Tailwind CSS base)
import App from './App.jsx'                   // Main application component

/**
 * Mount the React application to the DOM.
 * 
 * StrictMode is a development tool that:
 * - Identifies unsafe lifecycles
 * - Warns about legacy API usage
 * - Detects unexpected side effects
 * - Ensures reusable state
 * 
 * Note: StrictMode renders components twice in development to detect side effects.
 */
createRoot(document.getElementById('root')).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
