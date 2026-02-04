/**
 * =============================================================================
 * TrustLens Frontend - Main Application Component
 * =============================================================================
 * 
 * This is the root component of the TrustLens fraud detection dashboard.
 * It provides a real-time monitoring interface for analyzing credit card
 * transactions using deep learning models (MLP + Autoencoder) with
 * explainable AI (SHAP) visualizations.
 * 
 * Features:
 * - Real-time transaction monitoring dashboard
 * - Fraud/Approved status display with confidence scores
 * - XAI (Explainable AI) feature contribution charts
 * - Audit log history of analyzed transactions
 * - Responsive sidebar navigation
 * 
 * Use Cases Implemented:
 * - UC-12: Explainable AI Visualization
 * - UC-16: Audit Log/Session History
 * 
 * File: App.jsx
 * =============================================================================
 */

import React, { useState } from 'react';
// Lucide React icons for UI elements
import { 
  ShieldAlert,        // Security/fraud icon for branding
  LayoutDashboard,    // Dashboard navigation icon
  Database,           // Audit logs icon
  UserCheck,          // Admin panel icon
  Zap,                // Quick action/trigger icon
  Clock,              // Time-related icon
  DollarSign,         // Money/amount icon
  MessageSquareText,  // Explanation/text icon
  Activity,           // Activity/monitoring icon
  ArrowRight,         // Navigation arrow
  Search,             // Search functionality
  Filter              // Filter functionality
} from 'lucide-react';
import { useTransactionStore } from './store';    // Zustand state management
import XAIChart from './components/XAIChart';     // SHAP visualization component
import { getFraudPrediction } from './api';       // Backend API service

/**
 * Main Application Component
 * 
 * Renders the TrustLens fraud detection dashboard with:
 * - Sidebar navigation (Monitor, Audit Logs, Admin Panel)
 * - Dashboard view with real-time fraud analysis
 * - Audit logs view with transaction history table
 * 
 * @component
 * @returns {JSX.Element} The complete application UI
 */
export default function App() {
  // ===========================================================================
  // State Management
  // ===========================================================================
  
  // Access transaction history and addTransaction action from Zustand store
  const { history, addTransaction } = useTransactionStore();
  
  // Loading state for API calls (shows "Analyzing..." during prediction)
  const [loading, setLoading] = useState(false);
  
  // Current active tab: 'dashboard' (monitor) or 'logs' (audit history)
  const [activeTab, setActiveTab] = useState('dashboard');
  
  // Get the most recent transaction for display on dashboard
  const latestTx = history[0];

  // ===========================================================================
  // Transaction Analysis Handler
  // ===========================================================================
  
  /**
   * Trigger a fraud analysis scan.
   * 
   * This function:
   * 1. Generates simulated transaction data (30% fraud-like, 70% normal)
   * 2. Sends data to the backend /predict endpoint
   * 3. Stores the result in the transaction history
   * 4. Switches to dashboard view to show results
   * 
   * Note: In production, this would receive real transaction data
   * from a payment processing system or data stream.
   */
  const handleAnalysis = async () => {
    setLoading(true);
    
    // ==== SIMULATION: Generate test transaction data ====
    // Randomly decide if this should simulate a fraudulent transaction (30% chance)
    // In production, real transaction data would be used instead
    const simulateFraud = Math.random() < 0.3;
    
    let inputData;
    if (simulateFraud) {
      // =======================================================================
      // FRAUD-LIKE TRANSACTION SIMULATION
      // =======================================================================
      // Generate transaction with extreme/anomalous PCA feature values.
      // Features V4, V10, V12, V14 (indices 3, 9, 11, 13) are known strong
      // fraud indicators in the credit card fraud dataset.
      // Higher amounts (500-2500) are also associated with fraud patterns.
      // =======================================================================
      inputData = {
        Time: Math.floor(Math.random() * 500),  // Random timestamp
        V_features: Array(28).fill(0).map((_, i) => {
          // V14, V12, V10, V4 are typically strong fraud indicators
          if ([3, 9, 11, 13].includes(i)) {
            // Extreme values: -8 to -3 or 3 to 8 (high variance)
            return (Math.random() > 0.5 ? 1 : -1) * (Math.random() * 5 + 3);
          }
          // Other features: moderate variance (-3 to 3)
          return (Math.random() - 0.5) * 6;
        }),
        Amount: parseFloat((Math.random() * 2000 + 500).toFixed(2)) // $500-$2500
      };
    } else {
      // =======================================================================
      // NORMAL/LEGITIMATE TRANSACTION SIMULATION
      // =======================================================================
      // Generate transaction with typical/expected PCA feature values.
      // Small variance in features and lower transaction amounts
      // are characteristic of normal transactions.
      // =======================================================================
      inputData = { 
        Time: Math.floor(Math.random() * 500),  // Random timestamp
        V_features: Array(28).fill(0).map(() => (Math.random() - 0.5) * 2), // Small variance (-1 to 1)
        Amount: parseFloat((Math.random() * 150).toFixed(2)) // Normal amounts ($0-$150)
      };
    }
    
    // =========================================================================
    // API Call: Send transaction to backend for analysis
    // =========================================================================
    try {
      // Call the FastAPI /predict endpoint
      const response = await getFraudPrediction(inputData);
      
      // Extract SHAP explanation data for the XAI chart
      const chartData = response.xai_data || [];

      // Add the complete transaction record to history store
      // Combines API response with input data and metadata
      addTransaction({ 
        ...response,          // API response (is_fraud, status, risk_score, explanation, etc.)
        ...inputData,         // Original input data (Time, V_features, Amount)
        id: Date.now(),       // Unique ID using timestamp
        time_captured: new Date().toLocaleTimeString(), // Human-readable timestamp
        chartData             // SHAP data for visualization
      });
      
      // Switch to dashboard view to show the new result
      setActiveTab('dashboard');
    } catch (err) {
      // Log errors but don't crash the UI
      console.error("API Error:", err);
    } finally {
      // Always reset loading state
      setLoading(false);
    }
  };

  // ===========================================================================
  // Component Render
  // ===========================================================================
  return (
    <div className="min-h-screen bg-slate-50 flex font-sans">
      
      {/* =====================================================================
          SIDEBAR NAVIGATION
          =====================================================================
          Fixed left sidebar with navigation links.
          - Monitor (Dashboard): Real-time fraud detection view
          - Audit Logs: Transaction history table
          - Admin Panel: Link to admin interface (placeholder)
          ===================================================================== */}
      <aside className="w-64 bg-slate-900 text-white fixed h-full z-20 shadow-xl flex flex-col">
        {/* Brand Header */}
        <div className="p-5 flex items-center gap-2 border-b border-slate-800">
          <ShieldAlert className="text-blue-400" size={24} />
          <span className="text-lg font-black uppercase tracking-tight">TrustLens</span>
        </div>
        
        {/* Navigation Menu */}
        <nav className="p-3 flex-1 space-y-1 mt-2">
          {/* Monitor (Dashboard) Button */}
          <button 
            onClick={() => setActiveTab('dashboard')} 
            className={`w-full flex items-center gap-3 px-4 py-2.5 rounded-xl transition ${activeTab === 'dashboard' ? 'bg-blue-600 text-white shadow-lg' : 'text-slate-400 hover:bg-slate-800'}`}
          >
            <LayoutDashboard size={18} /> <span className="text-sm font-bold">Monitor</span>
          </button>
          
          {/* Audit Logs Button */}
          <button 
            onClick={() => setActiveTab('logs')} 
            className={`w-full flex items-center gap-3 px-4 py-2.5 rounded-xl transition ${activeTab === 'logs' ? 'bg-blue-600 text-white shadow-lg' : 'text-slate-400 hover:bg-slate-800'}`}
          >
            <Database size={18} /> <span className="text-sm font-bold">Audit Logs</span>
          </button>

          {/* Admin Panel Link (External) */}
          <a href="/admin" className="w-full flex items-center gap-3 px-4 py-2.5 rounded-xl text-slate-400 hover:bg-slate-800 transition">
            <UserCheck size={18} /> <span className="text-sm font-bold">Admin Panel</span>
          </a>
        </nav>

        {/* Version Footer */}
        <div className="p-5 border-t border-slate-800">
          <p className="text-[9px] text-slate-600 font-bold uppercase italic text-center">v2.3 | Full-Width Edition</p>
        </div>
      </aside>

      {/* =====================================================================
          MAIN CONTENT AREA
          =====================================================================
          Renders either the Dashboard or Audit Logs based on activeTab state.
          Uses ml-64 to offset for the fixed sidebar width.
          ===================================================================== */}
      <main className="flex-1 ml-64 p-8 w-full overflow-x-hidden">
        
        {/* =================================================================
            PAGE 1: MONITORING DASHBOARD
            =================================================================
            Real-time fraud detection interface showing:
            - Header with scan trigger button
            - Fraud/Approved status card with risk score
            - XAI explanation section with text and SHAP chart
            ================================================================= */}
        {activeTab === 'dashboard' && (
          <div className="w-full space-y-6 animate-in fade-in duration-500">
            {/* Dashboard Header */}
            <header className="flex justify-between items-center mb-8 border-b border-slate-200 pb-5">
              <div>
                <h1 className="text-3xl font-black text-slate-800 tracking-tight">Real-time Monitor</h1>
                <p className="text-slate-500 text-sm font-medium">Deep Learning Fraud Detection Intelligence</p>
              </div>
              {/* Trigger Live Scan Button - Initiates fraud analysis */}
              <button 
                onClick={handleAnalysis} 
                disabled={loading} 
                className="bg-blue-600 text-white px-8 py-3 rounded-xl font-bold shadow-lg hover:bg-blue-700 transition flex items-center gap-2 active:scale-95"
              >
                <Zap size={18} fill="currentColor" />
                {loading ? "Analyzing..." : "Trigger Live Scan"}
              </button>
            </header>

            {/* Empty State: No transactions analyzed yet */}
            {!latestTx ? (
              <div className="w-full bg-white border-2 border-dashed border-slate-200 rounded-[2rem] p-24 flex flex-col items-center justify-center text-slate-300">
                <Activity size={64} className="mb-4 animate-pulse" />
                <h2 className="text-xl font-black uppercase tracking-widest">System Ready</h2>
              </div>
            ) : (
              /* Transaction Analysis Results */
              <div className="w-full space-y-6">
                
                {/* =============================================================
                    STATUS CARD
                    =============================================================
                    Displays BLOCKED (red) or APPROVED (green) status with
                    the model confidence score and transaction amount.
                    ============================================================= */}
                <section className={`w-full p-8 rounded-[2rem] shadow-lg flex items-center justify-between text-white border-b-4 ${latestTx.is_fraud ? 'bg-red-600 border-red-800' : 'bg-green-600 border-green-800'}`}>
                  <div className="flex items-center gap-6">
                    <ShieldAlert size={56} className="bg-white/20 p-3 rounded-full shadow-inner" />
                    <div>
                      {/* Status: BLOCKED or APPROVED */}
                      <h2 className="text-5xl font-black italic uppercase leading-none tracking-tighter">{latestTx.status}</h2>
                      {/* Confidence: Combined risk score from MLP + Autoencoder */}
                      <p className="text-xs font-black opacity-80 uppercase mt-2 tracking-widest">Model Confidence: {latestTx.risk_score}%</p>
                    </div>
                  </div>
                  {/* Transaction Amount Display */}
                  <div className="text-right bg-black/10 px-6 py-3 rounded-2xl border border-white/10">
                    <p className="text-white/60 text-[10px] font-black uppercase tracking-widest">Captured Amount</p>
                    <p className="text-4xl font-black font-mono leading-none">${latestTx.Amount}</p>
                  </div>
                </section>

                {/* =============================================================
                    XAI (EXPLAINABLE AI) SECTION
                    =============================================================
                    Implements Use Case UC-12: Explainable AI Visualization
                    
                    Contains two parts:
                    1. Textual Explanation - Human-readable reason for the decision
                    2. SHAP Feature Chart - Visual representation of feature impacts
                    
                    This section addresses the "black box" problem in ML by showing
                    WHY the model made its fraud/not-fraud decision.
                    ============================================================= */}
                <section className="bg-white p-10 rounded-[2rem] shadow-sm border border-slate-200 w-full">
                  {/* Section Header with Labels */}
                  <div className="flex justify-between items-center mb-10">
                    <h3 className="text-[10px] font-black text-slate-400 uppercase tracking-widest flex items-center gap-2">
                      <MessageSquareText size={16} className="text-blue-500" /> Explainable AI (XAI) Focus Area
                    </h3>
                    <div className="flex gap-2">
                       <span className="text-[9px] font-bold bg-blue-50 text-blue-600 px-3 py-1 rounded-lg uppercase">Requirement UC-12</span>
                       <span className="text-[9px] font-bold bg-slate-50 text-slate-400 px-3 py-1 rounded-lg uppercase">SHAP Explainer</span>
                    </div>
                  </div>
                  
                  <div className="space-y-12">
                    {/* Part 1: Textual Explanation from Backend */}
                    <div className="space-y-6">
                      {/* Main explanation text from the /predict API response */}
                      <p className="text-3xl font-extrabold text-slate-800 border-l-[12px] border-blue-500 pl-8 leading-[1.15]">
                        {latestTx.explanation}
                      </p>
                      {/* Additional context for analysts */}
                      <div className="bg-slate-50 p-6 rounded-2xl border border-slate-100 text-base text-slate-600 font-medium leading-relaxed">
                        The neural network flagged this transaction due to extreme variance in 
                        latent PCA components. These specific features correlate with high-risk merchant 
                        categories and abnormal geolocation shifts. This interpretability allows human 
                        analysts to verify the "Black Box" decision.
                      </div>
                    </div>

                    {/* Part 2: SHAP Feature Contribution Chart */}
                    <div className="pt-10 border-t border-slate-100">
                       <h4 className="text-[10px] font-black text-slate-400 uppercase tracking-widest mb-6">SHAP Feature Contribution Analysis</h4>
                       <div className="bg-slate-50/50 p-8 rounded-[2rem] border border-slate-100">
                          {/* XAIChart component renders the horizontal bar chart */}
                          <XAIChart explanationData={latestTx.chartData} />
                       </div>
                       <p className="text-[11px] text-slate-400 mt-6 italic text-center italic font-medium">
                         * Visualizing the mathematical weight of each feature on the model's final fraud prediction.
                       </p>
                    </div>
                  </div>
                </section>
              </div>
            )}
          </div>
        )}

        {/* =================================================================
            PAGE 2: AUDIT LOGS (Transaction History)
            =================================================================
            Implements Use Case UC-16: Audit Log/Session History
            
            Displays a table of all transactions analyzed during the current
            session, showing timestamp, amount, status, and confidence score.
            Maximum 10 transactions are kept in memory (configured in store.js).
            ================================================================= */}
        {activeTab === 'logs' && (
          <div className="w-full space-y-6 animate-in slide-in-from-right-4 duration-500">
            {/* Page Header */}
            <header className="mb-8 border-b border-slate-200 pb-5">
              <h1 className="text-3xl font-black text-slate-800 tracking-tight">Audit Archive</h1>
              <p className="text-slate-500 text-sm font-medium tracking-tight">Session History & Transaction Forensic Log (UC-16)</p>
            </header>

            {/* Transaction History Table */}
            <div className="bg-white rounded-[2rem] shadow-sm border border-slate-200 overflow-hidden w-full">
              <table className="w-full text-left">
                {/* Table Header */}
                <thead className="bg-slate-50 text-slate-400 uppercase text-[10px] font-black border-b">
                  <tr>
                    <th className="px-10 py-5">Timestamp</th>
                    <th className="px-10 py-5">Transaction Amount</th>
                    <th className="px-10 py-5 text-center">System Status</th>
                    <th className="px-10 py-5 text-right">Confidence Score</th>
                  </tr>
                </thead>
                {/* Table Body */}
                <tbody className="divide-y divide-slate-100 text-sm font-medium">
                  {/* Empty State: No logs in current session */}
                  {history.length === 0 ? (
                    <tr><td colSpan="4" className="px-10 py-24 text-center text-slate-300 italic font-bold">No logs found for current monitoring session.</td></tr>
                  ) : (
                    /* Map through transaction history and render rows */
                    history.map((tx) => (
                      <tr key={tx.id} className="hover:bg-blue-50/40 transition duration-300 group">
                        {/* Timestamp Column */}
                        <td className="px-10 py-5 font-mono text-xs text-slate-400 group-hover:text-blue-600 transition-colors font-bold">{tx.time_captured}</td>
                        {/* Amount Column */}
                        <td className="px-10 py-5 text-slate-800 font-black text-base">${tx.Amount}</td>
                        {/* Status Badge (BLOCKED = red, APPROVED = green) */}
                        <td className="px-10 py-5 text-center">
                          <span className={`px-4 py-1 rounded-full text-[10px] font-black uppercase shadow-sm border ${tx.is_fraud ? 'bg-red-50 text-red-600 border-red-100' : 'bg-green-50 text-green-600 border-green-100'}`}>
                            {tx.status}
                          </span>
                        </td>
                        {/* Confidence Score Column */}
                        <td className="px-10 py-5 text-right font-mono text-slate-400 group-hover:text-slate-900 transition-all font-black">
                          {tx.risk_score}% <ArrowRight size={14} className="inline ml-1 opacity-0 group-hover:opacity-100 transition-opacity" />
                        </td>
                      </tr>
                    ))
                  )}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}