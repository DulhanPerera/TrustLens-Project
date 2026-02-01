import React, { useState } from 'react';
import { 
  ShieldAlert, LayoutDashboard, Database, UserCheck, 
  Zap, Clock, DollarSign, MessageSquareText, Activity, ArrowRight, Search, Filter
} from 'lucide-react';
import { useTransactionStore } from './store';
import XAIChart from './components/XAIChart';
import { getFraudPrediction } from './api';

export default function App() {
  const { history, addTransaction } = useTransactionStore();
  const [loading, setLoading] = useState(false);
  const [activeTab, setActiveTab] = useState('dashboard');
  
  const latestTx = history[0];

  const handleAnalysis = async () => {
    setLoading(true);
    
    // Randomly decide if this should simulate a fraudulent transaction (30% chance)
    const simulateFraud = Math.random() < 0.3;
    
    let inputData;
    if (simulateFraud) {
      // Generate fraud-like transaction with extreme/anomalous values
      inputData = {
        Time: Math.floor(Math.random() * 500),
        V_features: Array(28).fill(0).map((_, i) => {
          // V14, V12, V10, V4 are typically strong fraud indicators
          if ([3, 9, 11, 13].includes(i)) {
            return (Math.random() > 0.5 ? 1 : -1) * (Math.random() * 5 + 3); // Extreme values (-8 to -3 or 3 to 8)
          }
          return (Math.random() - 0.5) * 6; // More variance for other features
        }),
        Amount: parseFloat((Math.random() * 2000 + 500).toFixed(2)) // Higher amounts (500-2500)
      };
    } else {
      // Generate normal/legitimate transaction
      inputData = { 
        Time: Math.floor(Math.random() * 500), 
        V_features: Array(28).fill(0).map(() => (Math.random() - 0.5) * 2), // Smaller variance
        Amount: parseFloat((Math.random() * 150).toFixed(2)) // Normal amounts
      };
    }
    
    try {
      const response = await getFraudPrediction(inputData);
      const chartData = response.xai_data || [];

      addTransaction({ 
        ...response, 
        ...inputData,
        id: Date.now(), 
        time_captured: new Date().toLocaleTimeString(), 
        chartData 
      });
      setActiveTab('dashboard');
    } catch (err) {
      console.error("API Error:", err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-slate-50 flex font-sans">
      
      {/* 1. SIDEBAR NAVIGATION */}
      <aside className="w-64 bg-slate-900 text-white fixed h-full z-20 shadow-xl flex flex-col">
        <div className="p-5 flex items-center gap-2 border-b border-slate-800">
          <ShieldAlert className="text-blue-400" size={24} />
          <span className="text-lg font-black uppercase tracking-tight">TrustLens</span>
        </div>
        
        <nav className="p-3 flex-1 space-y-1 mt-2">
          <button 
            onClick={() => setActiveTab('dashboard')} 
            className={`w-full flex items-center gap-3 px-4 py-2.5 rounded-xl transition ${activeTab === 'dashboard' ? 'bg-blue-600 text-white shadow-lg' : 'text-slate-400 hover:bg-slate-800'}`}
          >
            <LayoutDashboard size={18} /> <span className="text-sm font-bold">Monitor</span>
          </button>
          
          <button 
            onClick={() => setActiveTab('logs')} 
            className={`w-full flex items-center gap-3 px-4 py-2.5 rounded-xl transition ${activeTab === 'logs' ? 'bg-blue-600 text-white shadow-lg' : 'text-slate-400 hover:bg-slate-800'}`}
          >
            <Database size={18} /> <span className="text-sm font-bold">Audit Logs</span>
          </button>

          <a href="/admin" className="w-full flex items-center gap-3 px-4 py-2.5 rounded-xl text-slate-400 hover:bg-slate-800 transition">
            <UserCheck size={18} /> <span className="text-sm font-bold">Admin Panel</span>
          </a>
        </nav>

        <div className="p-5 border-t border-slate-800">
          <p className="text-[9px] text-slate-600 font-bold uppercase italic text-center">v2.3 | Full-Width Edition</p>
        </div>
      </aside>

      {/* 2. MAIN CONTENT AREA - Now set to full width */}
      <main className="flex-1 ml-64 p-8 w-full overflow-x-hidden">
        
        {/* PAGE 1: MONITORING DASHBOARD */}
        {activeTab === 'dashboard' && (
          <div className="w-full space-y-6 animate-in fade-in duration-500">
            <header className="flex justify-between items-center mb-8 border-b border-slate-200 pb-5">
              <div>
                <h1 className="text-3xl font-black text-slate-800 tracking-tight">Real-time Monitor</h1>
                <p className="text-slate-500 text-sm font-medium">Deep Learning Fraud Detection Intelligence</p>
              </div>
              <button 
                onClick={handleAnalysis} 
                disabled={loading} 
                className="bg-blue-600 text-white px-8 py-3 rounded-xl font-bold shadow-lg hover:bg-blue-700 transition flex items-center gap-2 active:scale-95"
              >
                <Zap size={18} fill="currentColor" />
                {loading ? "Analyzing..." : "Trigger Live Scan"}
              </button>
            </header>

            {!latestTx ? (
              <div className="w-full bg-white border-2 border-dashed border-slate-200 rounded-[2rem] p-24 flex flex-col items-center justify-center text-slate-300">
                <Activity size={64} className="mb-4 animate-pulse" />
                <h2 className="text-xl font-black uppercase tracking-widest">System Ready</h2>
              </div>
            ) : (
              <div className="w-full space-y-6">
                
                {/* Status Card (Stretches Full Width) */}
                <section className={`w-full p-8 rounded-[2rem] shadow-lg flex items-center justify-between text-white border-b-4 ${latestTx.is_fraud ? 'bg-red-600 border-red-800' : 'bg-green-600 border-green-800'}`}>
                  <div className="flex items-center gap-6">
                    <ShieldAlert size={56} className="bg-white/20 p-3 rounded-full shadow-inner" />
                    <div>
                      <h2 className="text-5xl font-black italic uppercase leading-none tracking-tighter">{latestTx.status}</h2>
                      <p className="text-xs font-black opacity-80 uppercase mt-2 tracking-widest">Model Confidence: {latestTx.risk_score}%</p>
                    </div>
                  </div>
                  <div className="text-right bg-black/10 px-6 py-3 rounded-2xl border border-white/10">
                    <p className="text-white/60 text-[10px] font-black uppercase tracking-widest">Captured Amount</p>
                    <p className="text-4xl font-black font-mono leading-none">${latestTx.Amount}</p>
                  </div>
                </section>

                {/* Vertical XAI Interpretation Area (Full Width) */}
                <section className="bg-white p-10 rounded-[2rem] shadow-sm border border-slate-200 w-full">
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
                    {/* Part 1: Textual Explanation */}
                    <div className="space-y-6">
                      <p className="text-3xl font-extrabold text-slate-800 border-l-[12px] border-blue-500 pl-8 leading-[1.15]">
                        {latestTx.explanation}
                      </p>
                      <div className="bg-slate-50 p-6 rounded-2xl border border-slate-100 text-base text-slate-600 font-medium leading-relaxed">
                        The neural network flagged this transaction due to extreme variance in 
                        latent PCA components. These specific features correlate with high-risk merchant 
                        categories and abnormal geolocation shifts. This interpretability allows human 
                        analysts to verify the "Black Box" decision.
                      </div>
                    </div>

                    {/* Part 2: Feature Graph (Now stretches horizontally) */}
                    <div className="pt-10 border-t border-slate-100">
                       <h4 className="text-[10px] font-black text-slate-400 uppercase tracking-widest mb-6">SHAP Feature Contribution Analysis</h4>
                       <div className="bg-slate-50/50 p-8 rounded-[2rem] border border-slate-100">
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

        {/* PAGE 2: AUDIT LOGS (FULL WIDTH) */}
        {activeTab === 'logs' && (
          <div className="w-full space-y-6 animate-in slide-in-from-right-4 duration-500">
            <header className="mb-8 border-b border-slate-200 pb-5">
              <h1 className="text-3xl font-black text-slate-800 tracking-tight">Audit Archive</h1>
              <p className="text-slate-500 text-sm font-medium tracking-tight">Session History & Transaction Forensic Log (UC-16)</p>
            </header>

            <div className="bg-white rounded-[2rem] shadow-sm border border-slate-200 overflow-hidden w-full">
              <table className="w-full text-left">
                <thead className="bg-slate-50 text-slate-400 uppercase text-[10px] font-black border-b">
                  <tr>
                    <th className="px-10 py-5">Timestamp</th>
                    <th className="px-10 py-5">Transaction Amount</th>
                    <th className="px-10 py-5 text-center">System Status</th>
                    <th className="px-10 py-5 text-right">Confidence Score</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100 text-sm font-medium">
                  {history.length === 0 ? (
                    <tr><td colSpan="4" className="px-10 py-24 text-center text-slate-300 italic font-bold">No logs found for current monitoring session.</td></tr>
                  ) : (
                    history.map((tx) => (
                      <tr key={tx.id} className="hover:bg-blue-50/40 transition duration-300 group">
                        <td className="px-10 py-5 font-mono text-xs text-slate-400 group-hover:text-blue-600 transition-colors font-bold">{tx.time_captured}</td>
                        <td className="px-10 py-5 text-slate-800 font-black text-base">${tx.Amount}</td>
                        <td className="px-10 py-5 text-center">
                          <span className={`px-4 py-1 rounded-full text-[10px] font-black uppercase shadow-sm border ${tx.is_fraud ? 'bg-red-50 text-red-600 border-red-100' : 'bg-green-50 text-green-600 border-green-100'}`}>
                            {tx.status}
                          </span>
                        </td>
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