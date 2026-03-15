import React, { useEffect, useMemo, useState } from 'react';
import {
  ShieldAlert,
  LayoutDashboard,
  Database,
  UserCheck,
  Zap,
  MessageSquareText,
  Activity,
  ArrowRight,
  X,
  LogOut,
} from 'lucide-react';
import XAIChart from './components/XAIChart';
import GoogleLoginButton from './components/GoogleLoginButton';
import { getFraudPrediction } from './api';

const API_BASE = 'http://127.0.0.1:8000';

function mapBackendTransactionToUI(doc) {
  const createdAt = doc.created_at ? new Date(doc.created_at) : new Date();

  return {
    id: doc._id || doc.mongo_id || Date.now(),
    mongo_id: doc._id || null,
    transaction_id: doc.transaction_id || null,

    Time: doc.input?.Time ?? 0,
    V_features: doc.input?.V_features ?? [],
    Amount: doc.input?.Amount ?? 0,

    is_fraud: Boolean(doc.is_fraud),
    status: doc.status || 'UNKNOWN',
    risk_score: doc.risk_score ?? 0,
    mlp_prob_raw: doc.mlp_prob_raw ?? 0,
    fraud_prob: doc.fraud_prob ?? 0,
    recon_error: doc.recon_error ?? 0,
    anomaly_score: doc.anomaly_score ?? 0,
    combined_score: doc.combined_score ?? 0,

    thresholds: doc.thresholds || {},
    explanation: doc.explanation || 'No explanation available.',
    chartData: doc.xai_data || [],
    ae_xai_data: doc.ae_xai_data || [],

    raw_doc: doc,
    created_at: doc.created_at || null,
    time_captured: createdAt.toLocaleTimeString(),
    date_captured: createdAt.toLocaleDateString(),
  };
}

function DetailRow({ label, value }) {
  return (
    <div className="grid grid-cols-2 gap-4 py-3 border-b border-slate-100">
      <div className="text-xs font-bold uppercase tracking-wider text-slate-400">{label}</div>
      <div className="text-sm font-medium text-slate-800 break-words">{value}</div>
    </div>
  );
}

function HealthStatusItem({ label, ok, loading }) {
  let dotClass =
    'w-2.5 h-2.5 rounded-full shadow-[0_0_12px_rgba(148,163,184,0.5)] bg-slate-400';
  let text = 'Checking';

  if (!loading) {
    if (ok) {
      dotClass =
        'w-2.5 h-2.5 rounded-full bg-green-400 shadow-[0_0_12px_rgba(74,222,128,0.8)]';
      text = 'Online';
    } else {
      dotClass =
        'w-2.5 h-2.5 rounded-full bg-red-400 shadow-[0_0_12px_rgba(248,113,113,0.8)]';
      text = 'Offline';
    }
  }

  return (
    <div className="flex items-center justify-between gap-4">
      <div className="flex items-center gap-2">
        <span className={dotClass}></span>
        <span>{label}</span>
      </div>
      <span className="text-[11px] uppercase tracking-wider text-blue-200/80 font-bold">
        {text}
      </span>
    </div>
  );
}

export default function App() {
  const [user, setUser] = useState(() => {
    const saved = localStorage.getItem('trustlens_user');
    return saved ? JSON.parse(saved) : null;
  });

  const [loading, setLoading] = useState(false);
  const [activeTab, setActiveTab] = useState('dashboard');
  const [history, setHistory] = useState([]);
  const [loadingLogs, setLoadingLogs] = useState(false);
  const [logsError, setLogsError] = useState('');

  const [selectedTx, setSelectedTx] = useState(null);
  const [detailLoading, setDetailLoading] = useState(false);
  const [detailError, setDetailError] = useState('');

  const [health, setHealth] = useState(null);
  const [healthLoading, setHealthLoading] = useState(true);
  const [healthError, setHealthError] = useState('');

  const latestTx = useMemo(() => {
    return history.length > 0 ? history[0] : null;
  }, [history]);

  const loadHealth = async () => {
    setHealthError('');

    try {
      const res = await fetch(`${API_BASE}/health`);
      if (!res.ok) {
        throw new Error(`Failed to fetch health: ${res.status}`);
      }

      const data = await res.json();
      setHealth(data);
    } catch (error) {
      console.error('Failed to load system health:', error);
      setHealth(null);
      setHealthError('System health data unavailable');
    } finally {
      setHealthLoading(false);
    }
  };

  const loadTransactions = async () => {
    setLoadingLogs(true);
    setLogsError('');

    try {
      const res = await fetch(`${API_BASE}/transactions?limit=50`);
      if (!res.ok) {
        throw new Error(`Failed to fetch transactions: ${res.status}`);
      }

      const data = await res.json();
      const items = Array.isArray(data.items) ? data.items : [];
      const mapped = items.map(mapBackendTransactionToUI);

      mapped.sort((a, b) => {
        const aTime = a.created_at ? new Date(a.created_at).getTime() : 0;
        const bTime = b.created_at ? new Date(b.created_at).getTime() : 0;
        return bTime - aTime;
      });

      setHistory(mapped);
    } catch (error) {
      console.error('Failed to load audit logs:', error);
      setLogsError('Failed to load saved audit logs from backend.');
    } finally {
      setLoadingLogs(false);
    }
  };

  useEffect(() => {
    if (user) {
      loadTransactions();
    }
  }, [user]);

  useEffect(() => {
    if (!user) {
      setHealthLoading(true);
      loadHealth();

      const interval = setInterval(() => {
        loadHealth();
      }, 15000);

      return () => clearInterval(interval);
    }
  }, [user]);

  const handleAnalysis = async () => {
    setLoading(true);

    const simulateFraud = Math.random() < 0.3;

    let inputData;
    if (simulateFraud) {
      inputData = {
        transaction_id: `TXN-${Date.now()}`,
        Time: Math.floor(Math.random() * 500),
        V_features: Array(28)
          .fill(0)
          .map((_, i) => {
            if ([3, 9, 11, 13].includes(i)) {
              return (Math.random() > 0.5 ? 1 : -1) * (Math.random() * 5 + 3);
            }
            return (Math.random() - 0.5) * 6;
          }),
        Amount: parseFloat((Math.random() * 2000 + 500).toFixed(2)),
      };
    } else {
      inputData = {
        transaction_id: `TXN-${Date.now()}`,
        Time: Math.floor(Math.random() * 500),
        V_features: Array(28)
          .fill(0)
          .map(() => (Math.random() - 0.5) * 2),
        Amount: parseFloat((Math.random() * 150).toFixed(2)),
      };
    }

    try {
      const response = await getFraudPrediction(inputData);

      const newTx = {
        id: response.mongo_id || Date.now(),
        mongo_id: response.mongo_id || null,
        transaction_id: response.transaction_id || inputData.transaction_id,
        ...inputData,
        ...response,
        chartData: response.xai_data || [],
        ae_xai_data: response.ae_xai_data || [],
        thresholds: response.thresholds || {},
        created_at: new Date().toISOString(),
        time_captured: new Date().toLocaleTimeString(),
        date_captured: new Date().toLocaleDateString(),
      };

      setHistory((prev) => [newTx, ...prev]);
      setActiveTab('dashboard');
    } catch (err) {
      console.error('API Error:', err);
    } finally {
      setLoading(false);
    }
  };

  const openTransactionDetails = async (tx) => {
    if (!tx?.mongo_id) {
      setSelectedTx(tx);
      return;
    }

    setDetailLoading(true);
    setDetailError('');
    setSelectedTx(null);

    try {
      const res = await fetch(`${API_BASE}/transactions/${tx.mongo_id}`);
      if (!res.ok) {
        throw new Error(`Failed to fetch transaction details: ${res.status}`);
      }

      const doc = await res.json();
      const mapped = mapBackendTransactionToUI(doc);
      setSelectedTx(mapped);
    } catch (error) {
      console.error('Failed to fetch transaction details:', error);
      setDetailError('Failed to load transaction details.');
    } finally {
      setDetailLoading(false);
    }
  };

  const closeModal = () => {
    setSelectedTx(null);
    setDetailError('');
    setDetailLoading(false);
  };

  const handleLogout = () => {
    localStorage.removeItem('trustlens_user');
    setUser(null);
    setHistory([]);
    setSelectedTx(null);
    setActiveTab('dashboard');

    if (window.google?.accounts?.id) {
      window.google.accounts.id.disableAutoSelect();
    }
  };

  if (!user) {
    return (
      <div className="min-h-screen grid grid-cols-1 lg:grid-cols-2 bg-slate-900">
        <div className="relative hidden lg:flex flex-col justify-center px-20 text-white bg-gradient-to-br from-[#0B172A] via-[#0A1428] to-[#081225] overflow-hidden">
          <div className="absolute w-[600px] h-[600px] bg-blue-600 opacity-20 blur-[140px] rounded-full -top-40 -left-40 animate-pulse"></div>
          <div className="absolute w-[500px] h-[500px] bg-blue-500 opacity-20 blur-[140px] rounded-full bottom-0 right-0 animate-pulse"></div>

          <div className="relative z-10">
            <div className="mb-8 flex items-center gap-4">
              <div className="bg-white/10 p-4 rounded-2xl border border-white/10">
                <ShieldAlert size={40} className="text-blue-300" />
              </div>

              <h1 className="text-5xl xl:text-6xl font-black">
                TrustLens
              </h1>
            </div>

            <p className="text-xl font-semibold text-blue-100 mb-6">
              AI-Powered Fraud Detection Platform
            </p>

            <p className="text-blue-200 leading-relaxed max-w-md text-base xl:text-lg">
              TrustLens is a deep learning fraud detection system that combines
              neural networks, anomaly detection, and explainable AI to help
              analysts detect suspicious financial transactions in real time.
            </p>

            <div className="mt-10 text-sm text-blue-200 space-y-3">
              <p>✔ Real-time fraud monitoring</p>
              <p>✔ Deep learning risk scoring</p>
              <p>✔ Explainable AI insights (XAI)</p>
              <p>✔ Transaction audit intelligence</p>
            </div>

            <div className="mt-12">
              <div className="flex items-center justify-between mb-4">
                <p className="text-xs uppercase text-blue-300 font-bold tracking-widest">
                  System Status
                </p>

                <button
                  onClick={loadHealth}
                  className="text-[11px] font-bold uppercase tracking-wider text-blue-200 hover:text-white transition"
                >
                  Refresh
                </button>
              </div>

              <div className="space-y-3 text-sm text-blue-100">
                <HealthStatusItem
                  label="MLP Model"
                  ok={Boolean(health?.mlp_loaded)}
                  loading={healthLoading}
                />
                <HealthStatusItem
                  label="Autoencoder Engine"
                  ok={Boolean(health?.ae_loaded)}
                  loading={healthLoading}
                />
                <HealthStatusItem
                  label="MongoDB Database"
                  ok={Boolean(health?.mongodb?.connected)}
                  loading={healthLoading}
                />
                <HealthStatusItem
                  label="Explainable AI (SHAP)"
                  ok={Boolean(health?.shap_loaded)}
                  loading={healthLoading}
                />
              </div>

              {healthError && (
                <p className="mt-4 text-xs text-red-200 font-semibold">{healthError}</p>
              )}
            </div>
          </div>
        </div>

        <div className="flex items-center justify-center bg-slate-50 px-6 py-12 lg:px-12">
          <div className="bg-white p-10 rounded-[2rem] shadow-xl border border-slate-200 text-center w-full max-w-md transition hover:shadow-2xl hover:scale-[1.02]">
            <div className="flex justify-center mb-4">
              <div className="bg-slate-900 p-4 rounded-2xl">
                <ShieldAlert className="text-blue-400" size={34} />
              </div>
            </div>

            <h1 className="text-3xl font-black text-slate-800 mb-2">TrustLens</h1>
            <p className="text-slate-500 mb-8">
              Sign in with Google to access the fraud detection dashboard
            </p>

            <div className="flex justify-center">
              <GoogleLoginButton onLoginSuccess={setUser} />
            </div>

            <div className="lg:hidden mt-8 text-left bg-slate-50 rounded-2xl p-5 border border-slate-100">
              <p className="text-sm font-bold text-slate-800 mb-2">
                AI-Powered Fraud Detection Platform
              </p>
              <p className="text-sm text-slate-600 leading-relaxed">
                Real-time fraud monitoring with deep learning, anomaly detection,
                explainable AI, and transaction audit intelligence.
              </p>

              <div className="mt-5">
                <p className="text-[11px] uppercase text-slate-400 font-bold mb-3 tracking-widest">
                  System Status
                </p>

                <div className="space-y-3 text-sm text-slate-700">
                  <HealthStatusItem
                    label="MLP Model"
                    ok={Boolean(health?.mlp_loaded)}
                    loading={healthLoading}
                  />
                  <HealthStatusItem
                    label="Autoencoder Engine"
                    ok={Boolean(health?.ae_loaded)}
                    loading={healthLoading}
                  />
                  <HealthStatusItem
                    label="MongoDB Database"
                    ok={Boolean(health?.mongodb?.connected)}
                    loading={healthLoading}
                  />
                  <HealthStatusItem
                    label="Explainable AI (SHAP)"
                    ok={Boolean(health?.shap_loaded)}
                    loading={healthLoading}
                  />
                </div>

                {healthError && (
                  <p className="mt-4 text-xs text-red-500 font-semibold">{healthError}</p>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-slate-50 flex font-sans">
      <aside className="w-64 bg-slate-900 text-white fixed h-full z-20 shadow-xl flex flex-col">
        <div className="p-5 flex items-center gap-2 border-b border-slate-800">
          <ShieldAlert className="text-blue-400" size={24} />
          <span className="text-lg font-black uppercase tracking-tight">TrustLens</span>
        </div>

        <div className="px-4 py-4 border-b border-slate-800">
          <div className="flex items-center gap-3">
            {user.picture ? (
              <img
                src={user.picture}
                alt={user.name || 'User'}
                className="w-10 h-10 rounded-full border border-slate-700"
              />
            ) : (
              <div className="w-10 h-10 rounded-full bg-slate-700 flex items-center justify-center text-sm font-bold">
                {user.name?.[0] || 'U'}
              </div>
            )}

            <div className="min-w-0">
              <p className="text-sm font-bold text-white truncate">{user.name || 'User'}</p>
              <p className="text-[11px] text-slate-400 truncate">{user.email || ''}</p>
            </div>
          </div>
        </div>

        <nav className="p-3 flex-1 space-y-1 mt-2">
          <button
            onClick={() => setActiveTab('dashboard')}
            className={`w-full flex items-center gap-3 px-4 py-2.5 rounded-xl transition ${
              activeTab === 'dashboard'
                ? 'bg-blue-600 text-white shadow-lg'
                : 'text-slate-400 hover:bg-slate-800'
            }`}
          >
            <LayoutDashboard size={18} /> <span className="text-sm font-bold">Monitor</span>
          </button>

          <button
            onClick={() => setActiveTab('logs')}
            className={`w-full flex items-center gap-3 px-4 py-2.5 rounded-xl transition ${
              activeTab === 'logs'
                ? 'bg-blue-600 text-white shadow-lg'
                : 'text-slate-400 hover:bg-slate-800'
            }`}
          >
            <Database size={18} /> <span className="text-sm font-bold">Audit Logs</span>
          </button>

          <a
            href="/admin"
            className="w-full flex items-center gap-3 px-4 py-2.5 rounded-xl text-slate-400 hover:bg-slate-800 transition"
          >
            <UserCheck size={18} /> <span className="text-sm font-bold">Admin Panel</span>
          </a>
        </nav>

        <div className="p-5 border-t border-slate-800 space-y-3">
          <button
            onClick={handleLogout}
            className="w-full flex items-center justify-center gap-2 bg-slate-800 text-white px-4 py-2.5 rounded-xl text-sm font-bold hover:bg-slate-700 transition"
          >
            <LogOut size={16} />
            Logout
          </button>

          <p className="text-[9px] text-slate-600 font-bold uppercase italic text-center">
            v2.8 | Live Health Monitoring
          </p>
        </div>
      </aside>

      <main className="flex-1 ml-64 p-8 w-full overflow-x-hidden">
        {activeTab === 'dashboard' && (
          <div className="w-full space-y-6 animate-in fade-in duration-500">
            <header className="flex justify-between items-center mb-8 border-b border-slate-200 pb-5">
              <div>
                <h1 className="text-3xl font-black text-slate-800 tracking-tight">
                  Real-time Monitor
                </h1>
                <p className="text-slate-500 text-sm font-medium">
                  Deep Learning Fraud Detection Intelligence
                </p>
              </div>

              <button
                onClick={handleAnalysis}
                disabled={loading}
                className="bg-blue-600 text-white px-8 py-3 rounded-xl font-bold shadow-lg hover:bg-blue-700 transition flex items-center gap-2 active:scale-95"
              >
                <Zap size={18} fill="currentColor" />
                {loading ? 'Analyzing...' : 'Trigger Live Scan'}
              </button>
            </header>

            {!latestTx ? (
              <div className="w-full bg-white border-2 border-dashed border-slate-200 rounded-[2rem] p-24 flex flex-col items-center justify-center text-slate-300">
                <Activity size={64} className="mb-4 animate-pulse" />
                <h2 className="text-xl font-black uppercase tracking-widest">
                  {loadingLogs ? 'Loading Saved Transactions...' : 'System Ready'}
                </h2>
                {logsError && (
                  <p className="mt-3 text-sm text-red-500 font-semibold">{logsError}</p>
                )}
              </div>
            ) : (
              <div className="w-full space-y-6">
                <section
                  className={`w-full p-8 rounded-[2rem] shadow-lg flex items-center justify-between text-white border-b-4 ${
                    latestTx.is_fraud
                      ? 'bg-red-600 border-red-800'
                      : 'bg-green-600 border-green-800'
                  }`}
                >
                  <div className="flex items-center gap-6">
                    <ShieldAlert size={56} className="bg-white/20 p-3 rounded-full shadow-inner" />
                    <div>
                      <h2 className="text-5xl font-black italic uppercase leading-none tracking-tighter">
                        {latestTx.status}
                      </h2>
                      <p className="text-xs font-black opacity-80 uppercase mt-2 tracking-widest">
                        Model Confidence: {latestTx.risk_score}%
                      </p>
                    </div>
                  </div>

                  <div className="text-right bg-black/10 px-6 py-3 rounded-2xl border border-white/10">
                    <p className="text-white/60 text-[10px] font-black uppercase tracking-widest">
                      Captured Amount
                    </p>
                    <p className="text-4xl font-black font-mono leading-none">
                      ${latestTx.Amount}
                    </p>
                  </div>
                </section>

                <section className="bg-white p-10 rounded-[2rem] shadow-sm border border-slate-200 w-full">
                  <div className="flex justify-between items-center mb-10">
                    <h3 className="text-[10px] font-black text-slate-400 uppercase tracking-widest flex items-center gap-2">
                      <MessageSquareText size={16} className="text-blue-500" /> Explainable AI (XAI)
                    </h3>
                  </div>

                  <div className="space-y-12">
                    <div className="space-y-6">
                      <p className="text-3xl font-extrabold text-slate-800 border-l-[12px] border-blue-500 pl-8 leading-[1.15]">
                        {latestTx.explanation}
                      </p>

                      <div className="bg-slate-50 p-6 rounded-2xl border border-slate-100 text-base text-slate-600 font-medium leading-relaxed">
                        The neural network flagged this transaction due to extreme variance in
                        latent PCA components. This interpretability helps analysts verify the
                        decision with more confidence.
                      </div>
                    </div>

                    <div className="pt-10 border-t border-slate-100">
                      <h4 className="text-[10px] font-black text-slate-400 uppercase tracking-widest mb-6">
                        SHAP Feature Contribution Analysis
                      </h4>
                      <div className="bg-slate-50/50 p-8 rounded-[2rem] border border-slate-100">
                        <XAIChart explanationData={latestTx.chartData || []} />
                      </div>
                    </div>
                  </div>
                </section>
              </div>
            )}
          </div>
        )}

        {activeTab === 'logs' && (
          <div className="w-full space-y-6 animate-in slide-in-from-right-4 duration-500">
            <header className="mb-8 border-b border-slate-200 pb-5 flex items-center justify-between">
              <div>
                <h1 className="text-3xl font-black text-slate-800 tracking-tight">
                  Audit Archive
                </h1>
                <p className="text-slate-500 text-sm font-medium tracking-tight">
                  Click a transaction to view full details
                </p>
              </div>

              <button
                onClick={loadTransactions}
                className="bg-slate-900 text-white px-5 py-2 rounded-xl text-sm font-bold hover:bg-slate-700 transition"
              >
                Refresh Logs
              </button>
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
                  {loadingLogs ? (
                    <tr>
                      <td colSpan="4" className="px-10 py-24 text-center text-slate-400 font-bold">
                        Loading transactions...
                      </td>
                    </tr>
                  ) : history.length === 0 ? (
                    <tr>
                      <td colSpan="4" className="px-10 py-24 text-center text-slate-300 italic font-bold">
                        No saved logs found in MongoDB.
                      </td>
                    </tr>
                  ) : (
                    history.map((tx) => (
                      <tr
                        key={tx.id}
                        onClick={() => openTransactionDetails(tx)}
                        className="hover:bg-blue-50/40 transition duration-300 group cursor-pointer"
                      >
                        <td className="px-10 py-5 font-mono text-xs text-slate-400 group-hover:text-blue-600 transition-colors font-bold">
                          {tx.date_captured} {tx.time_captured}
                        </td>

                        <td className="px-10 py-5 text-slate-800 font-black text-base">
                          ${tx.Amount}
                        </td>

                        <td className="px-10 py-5 text-center">
                          <span
                            className={`px-4 py-1 rounded-full text-[10px] font-black uppercase shadow-sm border ${
                              tx.is_fraud
                                ? 'bg-red-50 text-red-600 border-red-100'
                                : 'bg-green-50 text-green-600 border-green-100'
                            }`}
                          >
                            {tx.status}
                          </span>
                        </td>

                        <td className="px-10 py-5 text-right font-mono text-slate-400 group-hover:text-slate-900 transition-all font-black">
                          {tx.risk_score}%{' '}
                          <ArrowRight
                            size={14}
                            className="inline ml-1 opacity-0 group-hover:opacity-100 transition-opacity"
                          />
                        </td>
                      </tr>
                    ))
                  )}
                </tbody>
              </table>
            </div>

            {logsError && <div className="text-red-500 font-semibold text-sm">{logsError}</div>}
          </div>
        )}
      </main>

      {(selectedTx || detailLoading || detailError) && (
        <div className="fixed inset-0 z-50 bg-black/50 flex items-center justify-center p-6">
          <div className="bg-white w-full max-w-5xl rounded-[2rem] shadow-2xl border border-slate-200 max-h-[90vh] overflow-y-auto">
            <div className="flex items-center justify-between px-8 py-6 border-b border-slate-200 sticky top-0 bg-white rounded-t-[2rem]">
              <div>
                <h2 className="text-2xl font-black text-slate-800">Transaction Details</h2>
                <p className="text-sm text-slate-500 font-medium">
                  Full stored record from MongoDB
                </p>
              </div>

              <button
                onClick={closeModal}
                className="p-2 rounded-xl hover:bg-slate-100 transition"
              >
                <X size={22} className="text-slate-600" />
              </button>
            </div>

            <div className="p-8">
              {detailLoading ? (
                <div className="text-center py-16 text-slate-500 font-bold">
                  Loading transaction details...
                </div>
              ) : detailError ? (
                <div className="text-center py-16 text-red-500 font-bold">{detailError}</div>
              ) : selectedTx ? (
                <div className="space-y-8">
                  <section
                    className={`p-6 rounded-[1.5rem] text-white ${
                      selectedTx.is_fraud ? 'bg-red-600' : 'bg-green-600'
                    }`}
                  >
                    <div className="flex items-center justify-between">
                      <div>
                        <div className="text-4xl font-black italic uppercase">
                          {selectedTx.status}
                        </div>
                        <div className="mt-2 text-sm font-bold uppercase tracking-wider opacity-90">
                          Risk Score: {selectedTx.risk_score}%
                        </div>
                      </div>
                      <div className="text-right">
                        <div className="text-xs uppercase font-bold opacity-80">
                          Transaction Amount
                        </div>
                        <div className="text-3xl font-black">${selectedTx.Amount}</div>
                      </div>
                    </div>
                  </section>

                  <section className="grid grid-cols-1 md:grid-cols-2 gap-8">
                    <div className="bg-slate-50 border border-slate-200 rounded-[1.5rem] p-6">
                      <h3 className="text-lg font-black text-slate-800 mb-4">Basic Information</h3>
                      <DetailRow label="Mongo ID" value={selectedTx.mongo_id || '-'} />
                      <DetailRow label="Transaction ID" value={selectedTx.transaction_id || '-'} />
                      <DetailRow
                        label="Created At"
                        value={
                          selectedTx.created_at
                            ? new Date(selectedTx.created_at).toLocaleString()
                            : '-'
                        }
                      />
                      <DetailRow label="Time Feature" value={String(selectedTx.Time)} />
                      <DetailRow label="Amount" value={`$${selectedTx.Amount}`} />
                      <DetailRow label="Status" value={selectedTx.status} />
                      <DetailRow label="Is Fraud" value={selectedTx.is_fraud ? 'Yes' : 'No'} />
                    </div>

                    <div className="bg-slate-50 border border-slate-200 rounded-[1.5rem] p-6">
                      <h3 className="text-lg font-black text-slate-800 mb-4">Model Scores</h3>
                      <DetailRow label="Risk Score" value={`${selectedTx.risk_score}%`} />
                      <DetailRow label="MLP Raw Probability" value={String(selectedTx.mlp_prob_raw)} />
                      <DetailRow label="Fraud Probability" value={String(selectedTx.fraud_prob)} />
                      <DetailRow label="Reconstruction Error" value={String(selectedTx.recon_error)} />
                      <DetailRow label="Anomaly Score" value={String(selectedTx.anomaly_score)} />
                      <DetailRow label="Combined Score" value={String(selectedTx.combined_score)} />
                    </div>
                  </section>

                  <section className="bg-slate-50 border border-slate-200 rounded-[1.5rem] p-6">
                    <h3 className="text-lg font-black text-slate-800 mb-4">Explanation</h3>
                    <p className="text-slate-700 text-base leading-relaxed">
                      {selectedTx.explanation}
                    </p>
                  </section>

                  <section className="bg-slate-50 border border-slate-200 rounded-[1.5rem] p-6">
                    <h3 className="text-lg font-black text-slate-800 mb-4">Thresholds</h3>
                    <pre className="text-sm text-slate-700 whitespace-pre-wrap overflow-x-auto">
                      {JSON.stringify(selectedTx.thresholds || {}, null, 2)}
                    </pre>
                  </section>

                  <section className="bg-slate-50 border border-slate-200 rounded-[1.5rem] p-6">
                    <h3 className="text-lg font-black text-slate-800 mb-4">SHAP Explanation</h3>
                    <div className="mb-6">
                      <XAIChart explanationData={selectedTx.chartData || []} />
                    </div>
                    <pre className="text-sm text-slate-700 whitespace-pre-wrap overflow-x-auto">
                      {JSON.stringify(selectedTx.chartData || [], null, 2)}
                    </pre>
                  </section>

                  <section className="bg-slate-50 border border-slate-200 rounded-[1.5rem] p-6">
                    <h3 className="text-lg font-black text-slate-800 mb-4">
                      Autoencoder Explanation
                    </h3>
                    <pre className="text-sm text-slate-700 whitespace-pre-wrap overflow-x-auto">
                      {JSON.stringify(selectedTx.ae_xai_data || [], null, 2)}
                    </pre>
                  </section>

                  <section className="bg-slate-50 border border-slate-200 rounded-[1.5rem] p-6">
                    <h3 className="text-lg font-black text-slate-800 mb-4">Raw V Features</h3>
                    <pre className="text-sm text-slate-700 whitespace-pre-wrap overflow-x-auto">
                      {JSON.stringify(selectedTx.V_features || [], null, 2)}
                    </pre>
                  </section>
                </div>
              ) : null}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}