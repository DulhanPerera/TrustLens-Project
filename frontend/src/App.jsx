import React, { useState, useMemo } from 'react';
import { ShieldAlert, Activity, Database, Zap, ArrowRight, Settings, Code } from 'lucide-react';
import { useTransactionStore } from './store';
import XAIChart from './components/XAIChart';
import { getFraudPrediction } from './api';

function App() {
  const { history, addTransaction } = useTransactionStore();
  const [loading, setLoading] = useState(false);
  const [selectedTxId, setSelectedTxId] = useState(null);
  const [threshold, setThreshold] = useState(0.5); // FR05: Configurable Threshold
  const [apiDebug, setApiDebug] = useState({ request: null, response: null }); // Track API calls

  // Get the chart data for the currently selected transaction
  const activeChartData = useMemo(() => {
    const selected = history.find(tx => tx.id === selectedTxId);
    if (selected) return selected.chartData;
    // Default/Fallback view
    return history[0]?.chartData || [
      { name: 'V14', impact: 0 }, { name: 'V17', impact: 0 }, { name: 'V12', impact: 0 }
    ];
  }, [selectedTxId, history]);

  const handleAnalysis = async () => {
    setLoading(true);
    const testData = { 
      Time: Math.floor(Math.random() * 1000), 
      V_features: Array(28).fill(0).map(() => (Math.random() - 0.5) * 5), 
      Amount: (Math.random() * 200).toFixed(2) 
    };
    
    // Store the request for debug display
    setApiDebug({ request: testData, response: null });
    
    try {
      const result = await getFraudPrediction(testData);
      
      // Store the response for debug display
      setApiDebug(prev => ({ ...prev, response: result }));
      
      // Dynamic Chart Data Mapping
      // In production, your backend should ideally return SHAP values directly
      const chartData = [
        { name: 'V14', impact: result.is_fraud ? (Math.random() * 0.5 + 0.4) : (Math.random() * 0.2) },
        { name: 'V17', impact: result.is_fraud ? (Math.random() * 0.5 + 0.3) : (Math.random() * 0.2) },
        { name: 'V12', impact: (Math.random() - 0.5) * 0.4 },
        { name: 'Amt', impact: testData.Amount > 150 ? 0.3 : -0.1 }
      ];

      const newTx = { 
        ...result, 
        id: Date.now(), 
        time: new Date().toLocaleTimeString(),
        amount: testData.Amount,
        chartData: chartData 
      };

      addTransaction(newTx);
      setSelectedTxId(newTx.id); // Automatically focus the newest scan
    } catch (err) {
      console.error("Integration Error:", err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex bg-slate-50 font-sans" style={{ minHeight: '100vh', display: 'flex', backgroundColor: '#f8fafc' }}>
      {/* Sidebar */}
      <aside className="w-64 bg-slate-900 text-white p-6 hidden md:flex flex-col border-r border-slate-800">
        <div className="flex items-center gap-2 mb-10 text-blue-400">
          <ShieldAlert size={32} />
          <span className="text-xl font-bold text-white tracking-tight">TrustLens</span>
        </div>
        <nav className="space-y-2 flex-1">
          <div className="flex items-center gap-3 p-3 bg-blue-600/20 text-blue-400 rounded-xl cursor-pointer">
            <Activity size={20} /> Dashboard
          </div>
          <div className="flex items-center gap-3 p-3 text-slate-400 hover:text-white transition rounded-xl cursor-pointer">
            <Database size={20} /> Audit Logs
          </div>
        </nav>
        <div className="bg-slate-800/50 p-4 rounded-xl mb-4">
            <div className="flex items-center gap-2 text-xs font-semibold text-slate-400 mb-2 uppercase">
                <Settings size={14} /> Threshold (FR05)
            </div>
            <input 
                type="range" min="0" max="1" step="0.1" value={threshold} 
                onChange={(e) => setThreshold(e.target.value)}
                className="w-full accent-blue-500"
            />
            <div className="text-right text-xs text-blue-400 font-bold mt-1">{threshold * 100}% Risk</div>
        </div>
      </aside>

      {/* Main Panel */}
      <main className="flex-1 p-8 overflow-y-auto">
        <div className="max-w-6xl mx-auto">
          <header className="flex justify-between items-center mb-8">
            <div>
              <h1 className="text-3xl font-extrabold text-slate-800">Fraud Analysis</h1>
              <p className="text-slate-500">Real-time Deep Learning Monitoring</p>
            </div>
            <button 
              onClick={handleAnalysis}
              disabled={loading}
              className="bg-blue-600 hover:bg-blue-700 text-white px-6 py-3 rounded-xl font-semibold shadow-lg shadow-blue-200 transition-all flex items-center gap-2 active:scale-95"
            >
              <Zap size={18} fill="currentColor" />
              {loading ? "Analyzing..." : "Trigger Scan"}
            </button>
          </header>

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
            {/* Chart Area */}
            <div className="lg:col-span-2 space-y-8">
              <section className="bg-white p-6 rounded-2xl shadow-sm border border-slate-200">
                <h2 className="text-lg font-bold mb-4 flex items-center gap-2 text-slate-700">
                  <Activity size={20} className="text-blue-500" /> XAI Interpretation (UC-12)
                </h2>
                <XAIChart explanationData={activeChartData} />
                <p className="mt-4 text-xs text-slate-400 italic text-center">
                    Visualizing SHAP feature importance for Transaction ID: {selectedTxId || 'None'}
                </p>
              </section>

              {/* API Debug Panel */}
              {apiDebug.request && (
                <section className="bg-slate-900 p-6 rounded-2xl shadow-sm border border-slate-700">
                  <h2 className="text-lg font-bold mb-4 flex items-center gap-2 text-white">
                    <Code size={20} className="text-green-400" /> API Request/Response
                  </h2>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div>
                      <h3 className="text-xs font-bold text-green-400 mb-2 uppercase">→ Request (POST /predict)</h3>
                      <pre className="bg-slate-800 p-3 rounded-lg text-xs text-slate-300 overflow-auto max-h-48">
{JSON.stringify(apiDebug.request, null, 2)}
                      </pre>
                    </div>
                    <div>
                      <h3 className="text-xs font-bold text-blue-400 mb-2 uppercase">← Response</h3>
                      <pre className="bg-slate-800 p-3 rounded-lg text-xs text-slate-300 overflow-auto max-h-48">
{apiDebug.response ? JSON.stringify(apiDebug.response, null, 2) : 'Loading...'}
                      </pre>
                    </div>
                  </div>
                </section>
              )}

              {/* Audit Table (UC-16) */}
              <section className="bg-white p-6 rounded-2xl shadow-sm border border-slate-200 overflow-hidden">
                <h2 className="text-lg font-bold mb-4 flex items-center gap-2 text-slate-700">
                  <Database size={20} className="text-slate-400" /> System Audit Logs
                </h2>
                <div className="overflow-x-auto">
                    <table className="w-full text-left text-sm">
                        <thead className="bg-slate-50 text-slate-500 uppercase text-[10px] font-bold">
                            <tr>
                                <th className="px-4 py-3">Timestamp</th>
                                <th className="px-4 py-3">Amount</th>
                                <th className="px-4 py-3">Result</th>
                                <th className="px-4 py-3">Risk Score</th>
                            </tr>
                        </thead>
                        <tbody className="divide-y divide-slate-100">
                            {history.map((tx) => (
                                <tr key={tx.id} className="hover:bg-slate-50/50 transition">
                                    <td className="px-4 py-3 font-mono text-xs">{tx.time}</td>
                                    <td className="px-4 py-3 text-slate-600">${tx.amount}</td>
                                    <td className={`px-4 py-3 font-bold ${tx.is_fraud ? 'text-red-500' : 'text-green-500'}`}>
                                        {tx.status}
                                    </td>
                                    <td className="px-4 py-3 text-slate-400">{tx.risk_score}%</td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
              </section>
            </div>

            {/* Live Feed (UC-11) */}
            <section className="bg-white p-6 rounded-2xl shadow-sm border border-slate-200 h-[700px] flex flex-col">
              <h2 className="text-lg font-bold mb-4">Live Alert Feed</h2>
              <div className="space-y-4 overflow-y-auto flex-1 pr-2">
                {history.length === 0 && (
                    <div className="flex flex-col items-center justify-center h-full text-slate-300 gap-2">
                        <Activity size={48} strokeWidth={1} />
                        <p className="text-sm italic">Listening for transactions...</p>
                    </div>
                )}
                {history.map((tx) => (
                  <div 
                    key={tx.id} 
                    onClick={() => setSelectedTxId(tx.id)}
                    className={`p-4 rounded-xl border-2 cursor-pointer transition-all ${
                        selectedTxId === tx.id 
                        ? 'border-blue-500 bg-blue-50/30' 
                        : 'border-transparent bg-slate-50 hover:border-slate-200'
                    }`}
                  >
                    <div className="flex justify-between items-start mb-2">
                      <span className={`text-[10px] font-black px-2 py-0.5 rounded uppercase ${tx.is_fraud ? 'bg-red-500 text-white' : 'bg-green-500 text-white'}`}>
                        {tx.status}
                      </span>
                      <span className="text-[10px] text-slate-400 font-mono">{tx.time}</span>
                    </div>
                    <div className="flex justify-between items-end">
                        <div>
                            <p className="text-sm font-bold text-slate-700">Risk: {tx.risk_score}%</p>
                            <p className="text-[10px] text-slate-400">Amt: ${tx.amount}</p>
                        </div>
                        <ArrowRight size={14} className={selectedTxId === tx.id ? 'text-blue-500' : 'text-slate-300'} />
                    </div>
                  </div>
                ))}
              </div>
            </section>
          </div>
        </div>
      </main>
    </div>
  );
}

export default App;