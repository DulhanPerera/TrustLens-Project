import React, { useEffect, useMemo, useState } from 'react';
import {
  updateUserRole as updateUserRoleApi,
  markTransactionLegitimate as markTransactionLegitimateApi,
  saveAnalystNote,
  getSystemSettings,
  updateSystemSettings,
  getActivityLogs,
} from '../api';

const API_BASE = 'http://127.0.0.1:8000';

function KPIBox({ title, value }) {
  return (
    <div className="bg-white border border-slate-200 rounded-2xl p-6 shadow-sm">
      <p className="text-xs uppercase tracking-widest text-slate-400 font-bold mb-2">
        {title}
      </p>
      <h3 className="text-3xl font-black text-slate-800">{value}</h3>
    </div>
  );
}

function StatusRow({ label, ok }) {
  return (
    <div className="flex items-center justify-between py-3 border-b border-slate-100 last:border-b-0">
      <span className="text-sm font-semibold text-slate-700">{label}</span>
      <span
        className={`px-3 py-1 rounded-full text-xs font-black uppercase ${
          ok ? 'bg-green-50 text-green-600' : 'bg-red-50 text-red-600'
        }`}
      >
        {ok ? 'Online' : 'Offline'}
      </span>
    </div>
  );
}

function DetailRow({ label, value }) {
  return (
    <div className="grid grid-cols-2 gap-4 py-3 border-b border-slate-100">
      <div className="text-xs font-bold uppercase tracking-wider text-slate-400">
        {label}
      </div>
      <div className="text-sm font-medium text-slate-800 break-words">{value}</div>
    </div>
  );
}

function SectionTab({ label, active, onClick }) {
  return (
    <button
      onClick={onClick}
      className={`px-4 py-2 rounded-xl text-sm font-bold transition ${
        active
          ? 'bg-blue-600 text-white shadow'
          : 'bg-white text-slate-600 border border-slate-200 hover:bg-slate-50'
      }`}
    >
      {label}
    </button>
  );
}

function FilterChip({ label, active, onClick }) {
  return (
    <button
      onClick={onClick}
      className={`px-3 py-2 rounded-xl text-xs font-black uppercase tracking-wider transition ${
        active
          ? 'bg-blue-600 text-white'
          : 'bg-slate-100 text-slate-600 hover:bg-slate-200'
      }`}
    >
      {label}
    </button>
  );
}

function SectionCard({ title, subtitle, children, right }) {
  return (
    <section className="bg-white rounded-2xl border border-slate-200 shadow-sm overflow-hidden">
      <div className="px-6 py-4 border-b border-slate-200 flex items-center justify-between bg-slate-50/70">
        <div>
          <h2 className="text-lg font-black text-slate-800">{title}</h2>
          {subtitle ? <p className="text-sm text-slate-500 mt-1">{subtitle}</p> : null}
        </div>
        {right}
      </div>
      <div>{children}</div>
    </section>
  );
}

export default function AdminPanel() {
  const [users, setUsers] = useState([]);
  const [transactions, setTransactions] = useState([]);
  const [reports, setReports] = useState([]);
  const [health, setHealth] = useState(null);
  const [activityLogs, setActivityLogs] = useState([]);

  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [activeSection, setActiveSection] = useState('overview');

  const [selectedTx, setSelectedTx] = useState(null);
  const [detailLoading, setDetailLoading] = useState(false);
  const [detailError, setDetailError] = useState('');

  const [selectedReport, setSelectedReport] = useState(null);
  const [reportLoading, setReportLoading] = useState(false);
  const [reportError, setReportError] = useState('');

  const [txFilter, setTxFilter] = useState('all');
  const [txSearch, setTxSearch] = useState('');
  const [roleSavingUserId, setRoleSavingUserId] = useState(null);

  const [noteText, setNoteText] = useState('');
  const [noteSaving, setNoteSaving] = useState(false);
  const [markingLegit, setMarkingLegit] = useState(false);

  const [settingsForm, setSettingsForm] = useState({
    site_name: 'TrustLens',
    maintenance_mode: false,
    notifications_enabled: true,
  });
  const [settingsSaving, setSettingsSaving] = useState(false);

  const loadActivityLogsOnly = async () => {
    try {
      const data = await getActivityLogs(50);
      setActivityLogs(data.items || []);
    } catch (err) {
      console.error(err);
    }
  };

  const loadAdminData = async () => {
    setLoading(true);
    setError('');

    try {
      const [usersRes, txRes, reportsRes, healthRes, activityData, settingsData] =
        await Promise.all([
          fetch(`${API_BASE}/users?limit=50`),
          fetch(`${API_BASE}/transactions?limit=50`),
          fetch(`${API_BASE}/reports?limit=50`),
          fetch(`${API_BASE}/health`),
          getActivityLogs(50),
          getSystemSettings(),
        ]);

      if (!usersRes.ok || !txRes.ok || !reportsRes.ok || !healthRes.ok) {
        throw new Error('Failed to load admin data');
      }

      const usersData = await usersRes.json();
      const txData = await txRes.json();
      const reportsData = await reportsRes.json();
      const healthData = await healthRes.json();

      setUsers(usersData.items || []);
      setTransactions(txData.items || []);
      setReports(reportsData.items || []);
      setHealth(healthData || null);
      setActivityLogs(activityData.items || []);

      setSettingsForm({
        site_name: settingsData.site_name ?? 'TrustLens',
        maintenance_mode: settingsData.maintenance_mode ?? false,
        notifications_enabled: settingsData.notifications_enabled ?? true,
      });
    } catch (err) {
      console.error(err);
      setError('Failed to load admin panel data.');
    } finally {
      setLoading(false);
    }
  };

  const handleUpdateUserRole = async (userId, newRole) => {
    setRoleSavingUserId(userId);

    try {
      const data = await updateUserRoleApi(userId, newRole);

      setUsers((prev) =>
        prev.map((user) =>
          user._id === userId ? { ...user, role: data.user?.role || newRole } : user
        )
      );

      await loadActivityLogsOnly();
    } catch (err) {
      console.error(err);
      alert(err.message || 'Failed to update role');
    } finally {
      setRoleSavingUserId(null);
    }
  };

  const openTransactionDetails = async (tx) => {
    if (!tx?._id) return;

    setDetailLoading(true);
    setDetailError('');
    setSelectedTx(null);
    setNoteText('');

    try {
      const res = await fetch(`${API_BASE}/transactions/${tx._id}`);
      if (!res.ok) {
        throw new Error('Failed to load transaction');
      }

      const data = await res.json();
      setSelectedTx(data);
      setNoteText('');
    } catch (err) {
      console.error(err);
      setDetailError('Failed to load transaction details.');
    } finally {
      setDetailLoading(false);
    }
  };

  const openReportDetails = async (report) => {
    if (!report?._id) return;

    setReportLoading(true);
    setReportError('');
    setSelectedReport(null);

    try {
      const res = await fetch(`${API_BASE}/reports/${report._id}`);
      if (!res.ok) {
        throw new Error('Failed to load report');
      }

      const data = await res.json();
      setSelectedReport(data);
    } catch (err) {
      console.error(err);
      setReportError('Failed to load report details.');
    } finally {
      setReportLoading(false);
    }
  };

  const handleMarkTransactionLegitimate = async () => {
    if (!selectedTx?._id) return;

    setMarkingLegit(true);
    try {
      const data = await markTransactionLegitimateApi(selectedTx._id);
      const updatedItem = data.item || null;

      if (updatedItem) {
        setSelectedTx(updatedItem);
        setTransactions((prev) =>
          prev.map((item) => (item._id === updatedItem._id ? updatedItem : item))
        );
      }

      await loadActivityLogsOnly();
    } catch (err) {
      console.error(err);
      alert(err.message || 'Failed to mark transaction as legitimate');
    } finally {
      setMarkingLegit(false);
    }
  };

  const handleAddAnalystNote = async () => {
    if (!selectedTx?._id || !noteText.trim()) return;

    setNoteSaving(true);
    try {
      const data = await saveAnalystNote(selectedTx._id, noteText.trim());
      const updatedItem = data.item || null;

      if (updatedItem) {
        setSelectedTx(updatedItem);
        setTransactions((prev) =>
          prev.map((item) => (item._id === updatedItem._id ? updatedItem : item))
        );
      }

      setNoteText('');
      await loadActivityLogsOnly();
    } catch (err) {
      console.error(err);
      alert(err.message || 'Failed to add note');
    } finally {
      setNoteSaving(false);
    }
  };

  const handleSaveSettings = async () => {
    setSettingsSaving(true);

    try {
      const data = await updateSystemSettings({
        site_name: settingsForm.site_name,
        maintenance_mode: settingsForm.maintenance_mode,
        notifications_enabled: settingsForm.notifications_enabled,
      });

      setSettingsForm({
        site_name: data.settings?.site_name ?? settingsForm.site_name,
        maintenance_mode:
          data.settings?.maintenance_mode ?? settingsForm.maintenance_mode,
        notifications_enabled:
          data.settings?.notifications_enabled ??
          settingsForm.notifications_enabled,
      });

      await loadActivityLogsOnly();
      alert('Settings updated successfully');
    } catch (err) {
      console.error(err);
      alert(err.message || 'Failed to save settings');
    } finally {
      setSettingsSaving(false);
    }
  };

  const closeTransactionModal = () => {
    setSelectedTx(null);
    setDetailLoading(false);
    setDetailError('');
    setNoteText('');
  };

  const closeReportModal = () => {
    setSelectedReport(null);
    setReportLoading(false);
    setReportError('');
  };

  useEffect(() => {
    loadAdminData();
  }, []);

  const fraudCount = useMemo(() => {
    return transactions.filter((t) => t.is_fraud === true).length;
  }, [transactions]);

  const approvedCount = useMemo(() => {
    return transactions.filter((t) => t.is_fraud === false).length;
  }, [transactions]);

  const avgRisk = useMemo(() => {
    if (!transactions.length) return 0;
    const total = transactions.reduce((sum, t) => sum + (t.risk_score || 0), 0);
    return (total / transactions.length).toFixed(1);
  }, [transactions]);

  const filteredTransactions = useMemo(() => {
    let items = [...transactions];

    if (txFilter === 'fraud') {
      items = items.filter((t) => t.is_fraud === true);
    } else if (txFilter === 'approved') {
      items = items.filter((t) => t.is_fraud === false);
    }

    const query = txSearch.trim().toLowerCase();
    if (query) {
      items = items.filter((t) =>
        String(t.transaction_id || t._id || '')
          .toLowerCase()
          .includes(query)
      );
    }

    return items;
  }, [transactions, txFilter, txSearch]);

  return (
    <div className="w-full space-y-8">
      <header className="border-b border-slate-200 pb-5 flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-black text-slate-800 tracking-tight">
            Admin Panel
          </h1>
          <p className="text-slate-500 text-sm font-medium">
            TrustLens system oversight, fraud analytics, reports, and monitoring
          </p>
        </div>

        <button
          onClick={loadAdminData}
          className="bg-slate-900 text-white px-5 py-2 rounded-xl text-sm font-bold hover:bg-slate-700 transition"
        >
          Refresh
        </button>
      </header>

      {error ? (
        <div className="bg-red-50 border border-red-200 text-red-600 px-5 py-4 rounded-2xl font-semibold text-sm">
          {error}
        </div>
      ) : null}

      <div className="flex flex-wrap gap-3">
        <SectionTab
          label="Overview"
          active={activeSection === 'overview'}
          onClick={() => setActiveSection('overview')}
        />
        <SectionTab
          label="Users"
          active={activeSection === 'users'}
          onClick={() => setActiveSection('users')}
        />
        <SectionTab
          label="Transactions"
          active={activeSection === 'transactions'}
          onClick={() => setActiveSection('transactions')}
        />
        <SectionTab
          label="Reports"
          active={activeSection === 'reports'}
          onClick={() => setActiveSection('reports')}
        />
        <SectionTab
          label="System Health"
          active={activeSection === 'health'}
          onClick={() => setActiveSection('health')}
        />
        <SectionTab
          label="Activity Logs"
          active={activeSection === 'activity'}
          onClick={() => setActiveSection('activity')}
        />
        <SectionTab
          label="Settings"
          active={activeSection === 'settings'}
          onClick={() => setActiveSection('settings')}
        />
      </div>

      {activeSection === 'overview' && (
        <div className="space-y-8">
          <section className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-5 gap-5">
            <KPIBox title="Total Users" value={loading ? '...' : users.length} />
            <KPIBox title="Transactions" value={loading ? '...' : transactions.length} />
            <KPIBox title="Fraud Detected" value={loading ? '...' : fraudCount} />
            <KPIBox title="Approved" value={loading ? '...' : approvedCount} />
            <KPIBox title="Average Risk Score" value={loading ? '...' : `${avgRisk}%`} />
          </section>

          <section className="grid grid-cols-1 xl:grid-cols-2 gap-6">
            <SectionCard
              title="System Health"
              subtitle="Live backend service and model readiness"
            >
              <div className="p-6">
                <StatusRow label="MLP Model" ok={Boolean(health?.mlp_loaded)} />
                <StatusRow label="Autoencoder" ok={Boolean(health?.ae_loaded)} />
                <StatusRow label="MongoDB" ok={Boolean(health?.mongodb?.connected)} />
                <StatusRow label="SHAP" ok={Boolean(health?.shap_loaded)} />
              </div>
            </SectionCard>

            <SectionCard
              title="Recent Reports"
              subtitle="Latest generated fraud analysis reports"
            >
              <div className="p-6 space-y-3">
                {reports.length === 0 ? (
                  <p className="text-slate-400 text-sm">No reports found.</p>
                ) : (
                  reports.slice(0, 5).map((r) => (
                    <div
                      key={r._id}
                      onClick={() => openReportDetails(r)}
                      className="flex items-center justify-between p-3 rounded-xl bg-slate-50 border border-slate-100 cursor-pointer hover:bg-blue-50/40 transition"
                    >
                      <div>
                        <p className="text-sm font-bold text-slate-800">
                          {r.transaction_id || r._id}
                        </p>
                        <p className="text-xs text-slate-500">
                          {r.created_at ? new Date(r.created_at).toLocaleString() : '-'}
                        </p>
                      </div>
                      <span className="text-xs font-black uppercase text-blue-600">
                        {r.decision || 'REPORT'}
                      </span>
                    </div>
                  ))
                )}
              </div>
            </SectionCard>
          </section>
        </div>
      )}

      {activeSection === 'users' && (
        <SectionCard
          title="Users"
          subtitle="Registered users with access roles and recent login activity"
        >
          <div className="overflow-x-auto">
            <table className="w-full text-left">
              <thead className="bg-slate-50 text-slate-400 uppercase text-[10px] font-black border-b">
                <tr>
                  <th className="px-6 py-4">Name</th>
                  <th className="px-6 py-4">Email</th>
                  <th className="px-6 py-4">Role</th>
                  <th className="px-6 py-4">Last Login</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100 text-sm">
                {users.length === 0 ? (
                  <tr>
                    <td colSpan="4" className="px-6 py-10 text-center text-slate-400">
                      No users found.
                    </td>
                  </tr>
                ) : (
                  users.map((u) => (
                    <tr key={u._id} className="hover:bg-slate-50">
                      <td className="px-6 py-4 font-semibold text-slate-800">{u.name}</td>
                      <td className="px-6 py-4 text-slate-600">{u.email}</td>
                      <td className="px-6 py-4">
                        <select
                          value={u.role || 'user'}
                          disabled={roleSavingUserId === u._id}
                          onChange={(e) => handleUpdateUserRole(u._id, e.target.value)}
                          className="px-3 py-2 rounded-xl border border-slate-200 text-xs font-black uppercase text-blue-600 bg-blue-50 focus:outline-none focus:ring-2 focus:ring-blue-500"
                        >
                          <option value="user">user</option>
                          <option value="analyst">analyst</option>
                          <option value="admin">admin</option>
                        </select>
                      </td>
                      <td className="px-6 py-4 text-slate-500">
                        {u.last_login_at ? new Date(u.last_login_at).toLocaleString() : '-'}
                      </td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
        </SectionCard>
      )}

      {activeSection === 'transactions' && (
        <SectionCard
          title="Transactions"
          subtitle="Recent fraud monitoring results across the system"
          right={
            <div className="flex flex-wrap items-center gap-3">
              <div className="flex gap-2">
                <FilterChip
                  label="All"
                  active={txFilter === 'all'}
                  onClick={() => setTxFilter('all')}
                />
                <FilterChip
                  label="Fraud Only"
                  active={txFilter === 'fraud'}
                  onClick={() => setTxFilter('fraud')}
                />
                <FilterChip
                  label="Approved Only"
                  active={txFilter === 'approved'}
                  onClick={() => setTxFilter('approved')}
                />
              </div>

              <input
                type="text"
                value={txSearch}
                onChange={(e) => setTxSearch(e.target.value)}
                placeholder="Search transaction ID"
                className="px-3 py-2 rounded-xl border border-slate-200 text-sm text-slate-700 focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>
          }
        >
          <div className="px-6 py-3 border-b border-slate-100 bg-slate-50/50 text-sm text-slate-500 font-medium">
            Showing {filteredTransactions.length} of {transactions.length} transactions
          </div>

          <div className="overflow-x-auto">
            <table className="w-full text-left">
              <thead className="bg-slate-50 text-slate-400 uppercase text-[10px] font-black border-b">
                <tr>
                  <th className="px-6 py-4">Transaction ID</th>
                  <th className="px-6 py-4">Amount</th>
                  <th className="px-6 py-4">Risk Score</th>
                  <th className="px-6 py-4">Status</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100 text-sm">
                {filteredTransactions.length === 0 ? (
                  <tr>
                    <td colSpan="4" className="px-6 py-10 text-center text-slate-400">
                      No matching transactions found.
                    </td>
                  </tr>
                ) : (
                  filteredTransactions.map((t) => (
                    <tr
                      key={t._id}
                      onClick={() => openTransactionDetails(t)}
                      className="hover:bg-blue-50/40 transition cursor-pointer"
                    >
                      <td className="px-6 py-4 font-semibold text-slate-800">
                        {t.transaction_id || t._id}
                      </td>
                      <td className="px-6 py-4 text-slate-700">
                        ${t.input?.Amount ?? '-'}
                      </td>
                      <td className="px-6 py-4 text-slate-700">
                        {t.risk_score ?? '-'}%
                      </td>
                      <td className="px-6 py-4">
                        <span
                          className={`px-3 py-1 rounded-full text-xs font-black uppercase ${
                            t.is_fraud
                              ? 'bg-red-50 text-red-600'
                              : 'bg-green-50 text-green-600'
                          }`}
                        >
                          {t.status || 'UNKNOWN'}
                        </span>
                      </td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
        </SectionCard>
      )}

      {activeSection === 'reports' && (
        <SectionCard title="Reports" subtitle="Saved fraud analysis reports">
          <div className="overflow-x-auto">
            <table className="w-full text-left">
              <thead className="bg-slate-50 text-slate-400 uppercase text-[10px] font-black border-b">
                <tr>
                  <th className="px-6 py-4">Report ID</th>
                  <th className="px-6 py-4">Decision</th>
                  <th className="px-6 py-4">Risk Score</th>
                  <th className="px-6 py-4">Created At</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100 text-sm">
                {reports.length === 0 ? (
                  <tr>
                    <td colSpan="4" className="px-6 py-10 text-center text-slate-400">
                      No reports found.
                    </td>
                  </tr>
                ) : (
                  reports.map((r) => (
                    <tr
                      key={r._id}
                      onClick={() => openReportDetails(r)}
                      className="hover:bg-blue-50/40 transition cursor-pointer"
                    >
                      <td className="px-6 py-4 font-semibold text-slate-800">
                        {r.transaction_id || r._id}
                      </td>
                      <td className="px-6 py-4 text-slate-700">{r.decision || '-'}</td>
                      <td className="px-6 py-4 text-slate-700">
                        {r.risk_score ?? '-'}%
                      </td>
                      <td className="px-6 py-4 text-slate-500">
                        {r.created_at ? new Date(r.created_at).toLocaleString() : '-'}
                      </td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
        </SectionCard>
      )}

      {activeSection === 'health' && (
        <SectionCard
          title="System Health"
          subtitle="Model, database, and service monitoring"
        >
          <div className="p-6">
            <StatusRow label="MLP Model" ok={Boolean(health?.mlp_loaded)} />
            <StatusRow label="Autoencoder" ok={Boolean(health?.ae_loaded)} />
            <StatusRow label="MongoDB" ok={Boolean(health?.mongodb?.connected)} />
            <StatusRow label="SHAP" ok={Boolean(health?.shap_loaded)} />
          </div>
        </SectionCard>
      )}

      {activeSection === 'activity' && (
        <SectionCard
          title="Recent Activity Logs"
          subtitle="Latest backend activity records"
        >
          <div className="p-6 space-y-3">
            {activityLogs.length === 0 ? (
              <p className="text-slate-400 text-sm">No recent activity found.</p>
            ) : (
              activityLogs.map((item) => (
                <div
                  key={item._id}
                  className="p-4 rounded-xl bg-slate-50 border border-slate-100"
                >
                  <p className="text-sm font-semibold text-slate-800">
                    {item.action || 'activity'}
                  </p>
                  <p className="text-xs text-slate-500 mt-1">
                    Actor: {item.actor || 'system'}
                  </p>
                  <p className="text-xs text-slate-500 mt-1">
                    {item.created_at ? new Date(item.created_at).toLocaleString() : '-'}
                  </p>
                  {item.details ? (
                    <pre className="mt-3 text-xs text-slate-700 whitespace-pre-wrap overflow-x-auto">
                      {JSON.stringify(item.details, null, 2)}
                    </pre>
                  ) : null}
                </div>
              ))
            )}
          </div>
        </SectionCard>
      )}

      {activeSection === 'settings' && (
        <SectionCard
          title="System Settings"
          subtitle="Basic platform configuration values"
        >
          <div className="p-6 grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="space-y-5">
              <div>
                <label className="block text-sm font-bold text-slate-700 mb-2">
                  Site Name
                </label>
                <input
                  type="text"
                  value={settingsForm.site_name}
                  onChange={(e) =>
                    setSettingsForm((prev) => ({
                      ...prev,
                      site_name: e.target.value,
                    }))
                  }
                  className="w-full px-4 py-3 rounded-xl border border-slate-200 text-slate-800 focus:outline-none focus:ring-2 focus:ring-blue-500"
                />
              </div>

              <div className="flex items-center justify-between bg-slate-50 border border-slate-200 rounded-xl p-4">
                <div>
                  <p className="text-sm font-bold text-slate-800">Maintenance Mode</p>
                  <p className="text-xs text-slate-500">
                    Temporarily mark system as under maintenance
                  </p>
                </div>
                <input
                  type="checkbox"
                  checked={settingsForm.maintenance_mode}
                  onChange={(e) =>
                    setSettingsForm((prev) => ({
                      ...prev,
                      maintenance_mode: e.target.checked,
                    }))
                  }
                  className="h-5 w-5"
                />
              </div>

              <div className="flex items-center justify-between bg-slate-50 border border-slate-200 rounded-xl p-4">
                <div>
                  <p className="text-sm font-bold text-slate-800">Notifications Enabled</p>
                  <p className="text-xs text-slate-500">
                    Enable automated system notifications
                  </p>
                </div>
                <input
                  type="checkbox"
                  checked={settingsForm.notifications_enabled}
                  onChange={(e) =>
                    setSettingsForm((prev) => ({
                      ...prev,
                      notifications_enabled: e.target.checked,
                    }))
                  }
                  className="h-5 w-5"
                />
              </div>

              <button
                onClick={handleSaveSettings}
                disabled={settingsSaving}
                className="bg-blue-600 text-white px-5 py-3 rounded-xl text-sm font-bold hover:bg-blue-700 transition disabled:opacity-60"
              >
                {settingsSaving ? 'Saving...' : 'Save Settings'}
              </button>
            </div>

            <div className="bg-slate-50 border border-slate-200 rounded-2xl p-5">
              <h3 className="text-lg font-black text-slate-800 mb-4">
                Current Stored Settings
              </h3>
              <pre className="text-sm text-slate-700 whitespace-pre-wrap overflow-x-auto">
                {JSON.stringify(settingsForm || {}, null, 2)}
              </pre>
            </div>
          </div>
        </SectionCard>
      )}

      {(selectedTx || detailLoading || detailError) && (
        <div className="fixed inset-0 z-50 bg-black/50 flex items-center justify-center p-6">
          <div className="bg-white w-full max-w-5xl rounded-[2rem] shadow-2xl border border-slate-200 max-h-[90vh] overflow-y-auto">
            <div className="flex items-center justify-between px-8 py-6 border-b border-slate-200 sticky top-0 bg-white rounded-t-[2rem] z-10">
              <div>
                <h2 className="text-2xl font-black text-slate-800">Transaction Details</h2>
                <p className="text-sm text-slate-500 font-medium">
                  Full stored record from MongoDB
                </p>
              </div>

              <button
                onClick={closeTransactionModal}
                className="px-4 py-2 rounded-xl hover:bg-slate-100 transition text-slate-600 font-semibold"
              >
                Close
              </button>
            </div>

            <div className="p-8">
              {detailLoading ? (
                <div className="text-center py-16 text-slate-500 font-bold">
                  Loading transaction details...
                </div>
              ) : detailError ? (
                <div className="text-center py-16 text-red-500 font-bold">
                  {detailError}
                </div>
              ) : selectedTx ? (
                <div className="space-y-8">
                  <section
                    className={`p-6 rounded-[1.5rem] text-white ${
                      selectedTx.is_fraud ? 'bg-red-600' : 'bg-green-600'
                    }`}
                  >
                    <div className="flex items-center justify-between gap-6">
                      <div>
                        <div className="text-4xl font-black italic uppercase">
                          {selectedTx.status}
                        </div>
                        <div className="mt-2 text-sm font-bold uppercase tracking-wider opacity-90">
                          Risk Score: {selectedTx.risk_score ?? '-'}%
                        </div>
                        {selectedTx.analyst_override ? (
                          <div className="mt-3 inline-block px-3 py-1 rounded-full text-xs font-black uppercase bg-white/20">
                            Analyst Override Applied
                          </div>
                        ) : null}
                      </div>

                      <div className="text-right">
                        <div className="text-xs uppercase font-bold opacity-80">
                          Transaction Amount
                        </div>
                        <div className="text-3xl font-black">
                          ${selectedTx.input?.Amount ?? '-'}
                        </div>
                      </div>
                    </div>
                  </section>

                  <div className="flex flex-wrap gap-3">
                    {selectedTx.is_fraud ? (
                      <button
                        onClick={handleMarkTransactionLegitimate}
                        disabled={markingLegit}
                        className="bg-green-600 text-white px-5 py-3 rounded-xl text-sm font-bold hover:bg-green-700 transition disabled:opacity-60"
                      >
                        {markingLegit ? 'Processing...' : 'Mark as Legitimate'}
                      </button>
                    ) : null}
                  </div>

                  <section className="grid grid-cols-1 md:grid-cols-2 gap-8">
                    <div className="bg-slate-50 border border-slate-200 rounded-[1.5rem] p-6">
                      <h3 className="text-lg font-black text-slate-800 mb-4">
                        Basic Information
                      </h3>
                      <DetailRow label="Mongo ID" value={selectedTx._id || '-'} />
                      <DetailRow
                        label="Transaction ID"
                        value={selectedTx.transaction_id || '-'}
                      />
                      <DetailRow
                        label="Created At"
                        value={
                          selectedTx.created_at
                            ? new Date(selectedTx.created_at).toLocaleString()
                            : '-'
                        }
                      />
                      <DetailRow
                        label="Amount"
                        value={`$${selectedTx.input?.Amount ?? '-'}`}
                      />
                      <DetailRow label="Status" value={selectedTx.status || '-'} />
                      <DetailRow
                        label="Is Fraud"
                        value={selectedTx.is_fraud ? 'Yes' : 'No'}
                      />
                    </div>

                    <div className="bg-slate-50 border border-slate-200 rounded-[1.5rem] p-6">
                      <h3 className="text-lg font-black text-slate-800 mb-4">
                        Model Scores
                      </h3>
                      <DetailRow
                        label="Risk Score"
                        value={`${selectedTx.risk_score ?? '-'}%`}
                      />
                      <DetailRow
                        label="MLP Raw Probability"
                        value={String(selectedTx.mlp_prob_raw ?? '-')}
                      />
                      <DetailRow
                        label="Fraud Probability"
                        value={String(selectedTx.fraud_prob ?? '-')}
                      />
                      <DetailRow
                        label="Reconstruction Error"
                        value={String(selectedTx.recon_error ?? '-')}
                      />
                      <DetailRow
                        label="Anomaly Score"
                        value={String(selectedTx.anomaly_score ?? '-')}
                      />
                      <DetailRow
                        label="Combined Score"
                        value={String(selectedTx.combined_score ?? '-')}
                      />
                    </div>
                  </section>

                  <section className="bg-slate-50 border border-slate-200 rounded-[1.5rem] p-6">
                    <h3 className="text-lg font-black text-slate-800 mb-4">Explanation</h3>
                    <p className="text-slate-700 text-base leading-relaxed">
                      {selectedTx.explanation || 'No explanation available'}
                    </p>
                  </section>

                  <section className="bg-slate-50 border border-slate-200 rounded-[1.5rem] p-6">
                    <h3 className="text-lg font-black text-slate-800 mb-4">
                      Analyst Notes
                    </h3>

                    <div className="space-y-3 mb-5">
                      {Array.isArray(selectedTx.analyst_notes) &&
                      selectedTx.analyst_notes.length > 0 ? (
                        selectedTx.analyst_notes.map((note, index) => (
                          <div
                            key={`${note.created_at || index}-${index}`}
                            className="p-4 rounded-xl bg-white border border-slate-200"
                          >
                            <p className="text-sm text-slate-800 font-medium">
                              {note.note || '-'}
                            </p>
                            <p className="text-xs text-slate-500 mt-2">
                              {note.created_at
                                ? new Date(note.created_at).toLocaleString()
                                : '-'}
                            </p>
                          </div>
                        ))
                      ) : (
                        <p className="text-sm text-slate-400">
                          No analyst notes added yet.
                        </p>
                      )}
                    </div>

                    <div className="space-y-3">
                      <textarea
                        value={noteText}
                        onChange={(e) => setNoteText(e.target.value)}
                        placeholder="Add analyst note..."
                        rows={4}
                        className="w-full px-4 py-3 rounded-xl border border-slate-200 text-slate-800 focus:outline-none focus:ring-2 focus:ring-blue-500"
                      />
                      <button
                        onClick={handleAddAnalystNote}
                        disabled={noteSaving || !noteText.trim()}
                        className="bg-blue-600 text-white px-5 py-3 rounded-xl text-sm font-bold hover:bg-blue-700 transition disabled:opacity-60"
                      >
                        {noteSaving ? 'Saving Note...' : 'Add Note'}
                      </button>
                    </div>
                  </section>

                  <section className="bg-slate-50 border border-slate-200 rounded-[1.5rem] p-6">
                    <h3 className="text-lg font-black text-slate-800 mb-4">Thresholds</h3>
                    <pre className="text-sm text-slate-700 whitespace-pre-wrap overflow-x-auto">
                      {JSON.stringify(selectedTx.thresholds || {}, null, 2)}
                    </pre>
                  </section>

                  <section className="bg-slate-50 border border-slate-200 rounded-[1.5rem] p-6">
                    <h3 className="text-lg font-black text-slate-800 mb-4">Raw Input</h3>
                    <pre className="text-sm text-slate-700 whitespace-pre-wrap overflow-x-auto">
                      {JSON.stringify(selectedTx.input || {}, null, 2)}
                    </pre>
                  </section>
                </div>
              ) : null}
            </div>
          </div>
        </div>
      )}

      {(selectedReport || reportLoading || reportError) && (
        <div className="fixed inset-0 z-50 bg-black/50 flex items-center justify-center p-6">
          <div className="bg-white w-full max-w-5xl rounded-[2rem] shadow-2xl border border-slate-200 max-h-[90vh] overflow-y-auto">
            <div className="flex items-center justify-between px-8 py-6 border-b border-slate-200 sticky top-0 bg-white rounded-t-[2rem] z-10">
              <div>
                <h2 className="text-2xl font-black text-slate-800">Report Details</h2>
                <p className="text-sm text-slate-500 font-medium">
                  Full stored report from MongoDB
                </p>
              </div>

              <button
                onClick={closeReportModal}
                className="px-4 py-2 rounded-xl hover:bg-slate-100 transition text-slate-600 font-semibold"
              >
                Close
              </button>
            </div>

            <div className="p-8">
              {reportLoading ? (
                <div className="text-center py-16 text-slate-500 font-bold">
                  Loading report details...
                </div>
              ) : reportError ? (
                <div className="text-center py-16 text-red-500 font-bold">
                  {reportError}
                </div>
              ) : selectedReport ? (
                <div className="space-y-8">
                  <section
                    className={`p-6 rounded-[1.5rem] text-white ${
                      selectedReport.decision === 'BLOCKED'
                        ? 'bg-red-600'
                        : selectedReport.decision === 'REVIEW'
                        ? 'bg-amber-500'
                        : 'bg-green-600'
                    }`}
                  >
                    <div className="flex items-center justify-between">
                      <div>
                        <div className="text-4xl font-black italic uppercase">
                          {selectedReport.decision || 'REPORT'}
                        </div>
                        <div className="mt-2 text-sm font-bold uppercase tracking-wider opacity-90">
                          Risk Score: {selectedReport.risk_score ?? '-'}%
                        </div>
                      </div>
                      <div className="text-right">
                        <div className="text-xs uppercase font-bold opacity-80">
                          Confidence
                        </div>
                        <div className="text-3xl font-black">
                          {selectedReport.confidence ?? '-'}
                        </div>
                      </div>
                    </div>
                  </section>

                  <section className="grid grid-cols-1 md:grid-cols-2 gap-8">
                    <div className="bg-slate-50 border border-slate-200 rounded-[1.5rem] p-6">
                      <h3 className="text-lg font-black text-slate-800 mb-4">
                        Basic Information
                      </h3>
                      <DetailRow label="Mongo ID" value={selectedReport._id || '-'} />
                      <DetailRow
                        label="Transaction ID"
                        value={selectedReport.transaction_id || '-'}
                      />
                      <DetailRow
                        label="Created At"
                        value={
                          selectedReport.created_at
                            ? new Date(selectedReport.created_at).toLocaleString()
                            : '-'
                        }
                      />
                      <DetailRow label="Decision" value={selectedReport.decision || '-'} />
                      <DetailRow
                        label="Risk Score"
                        value={`${selectedReport.risk_score ?? '-'}%`}
                      />
                      <DetailRow
                        label="Confidence"
                        value={String(selectedReport.confidence ?? '-')}
                      />
                    </div>

                    <div className="bg-slate-50 border border-slate-200 rounded-[1.5rem] p-6">
                      <h3 className="text-lg font-black text-slate-800 mb-4">Signals</h3>
                      <pre className="text-sm text-slate-700 whitespace-pre-wrap overflow-x-auto">
                        {JSON.stringify(selectedReport.signals || {}, null, 2)}
                      </pre>
                    </div>
                  </section>

                  <section className="bg-slate-50 border border-slate-200 rounded-[1.5rem] p-6">
                    <h3 className="text-lg font-black text-slate-800 mb-4">Explanations</h3>
                    <pre className="text-sm text-slate-700 whitespace-pre-wrap overflow-x-auto">
                      {JSON.stringify(selectedReport.explanations || {}, null, 2)}
                    </pre>
                  </section>

                  <section className="bg-slate-50 border border-slate-200 rounded-[1.5rem] p-6">
                    <h3 className="text-lg font-black text-slate-800 mb-4">Thresholds</h3>
                    <pre className="text-sm text-slate-700 whitespace-pre-wrap overflow-x-auto">
                      {JSON.stringify(selectedReport.thresholds || {}, null, 2)}
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