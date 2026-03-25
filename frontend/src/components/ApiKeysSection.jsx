import { useState } from 'react';
import {
  createApiKey,
  getApiKeys,
  updateApiKeyStatus,
  getRequestLogs,
  getActivityLogs,
} from '../api';

function formatDate(value) {
  if (!value) return '-';
  try {
    return new Date(value).toLocaleString();
  } catch {
    return String(value);
  }
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

export default function ApiKeysSection({
  apiKeys,
  setApiKeys,
  requestLogs,
  setRequestLogs,
  setActivityLogs,
}) {
  const [clientName, setClientName] = useState('');
  const [apiKeyCreating, setApiKeyCreating] = useState(false);
  const [apiKeyStatusSavingId, setApiKeyStatusSavingId] = useState(null);
  const [tokenModal, setTokenModal] = useState(null);

  const loadApiKeysOnly = async () => {
    try {
      const data = await getApiKeys(50);
      setApiKeys(data.items || []);
    } catch (err) {
      console.error(err);
      alert('Failed to load API keys');
    }
  };

  const loadRequestLogsOnly = async () => {
    try {
      const data = await getRequestLogs(50);
      setRequestLogs(data.items || []);
    } catch (err) {
      console.error(err);
      alert('Failed to load request logs');
    }
  };

  const loadActivityLogsOnly = async () => {
    try {
      const data = await getActivityLogs(50);
      setActivityLogs(data.items || []);
    } catch (err) {
      console.error(err);
    }
  };

  const handleCreateApiKey = async () => {
    if (!clientName.trim()) {
      alert('Please enter a client name.');
      return;
    }

    setApiKeyCreating(true);
    try {
      const data = await createApiKey(clientName.trim());
      setTokenModal(data);
      setClientName('');
      await loadApiKeysOnly();
      await loadActivityLogsOnly();
    } catch (err) {
      console.error(err);
      alert(err?.response?.data?.detail || err.message || 'Failed to create API key');
    } finally {
      setApiKeyCreating(false);
    }
  };

  const handleToggleApiKeyStatus = async (apiKeyId, currentStatus) => {
    const nextStatus = currentStatus === 'active' ? 'inactive' : 'active';
    setApiKeyStatusSavingId(apiKeyId);

    try {
      const data = await updateApiKeyStatus(apiKeyId, nextStatus);
      const updatedItem = data.item;

      setApiKeys((prev) =>
        prev.map((item) => (item._id === apiKeyId ? updatedItem : item))
      );

      await loadActivityLogsOnly();
      await loadRequestLogsOnly();
    } catch (err) {
      console.error(err);
      alert(err?.response?.data?.detail || err.message || 'Failed to update API key status');
    } finally {
      setApiKeyStatusSavingId(null);
    }
  };

  const copyTokenToClipboard = async () => {
    if (!tokenModal?.token) return;

    try {
      await navigator.clipboard.writeText(tokenModal.token);
      alert('API token copied to clipboard');
    } catch (err) {
      console.error(err);
      alert('Failed to copy token');
    }
  };

  const closeTokenModal = () => {
    setTokenModal(null);
  };

  return (
    <div className="space-y-8">
      <SectionCard
        title="Generate API Key"
        subtitle="Create a new API key for an external client system"
      >
        <div className="p-6 flex flex-col md:flex-row gap-4 items-start md:items-end">
          <div className="w-full md:max-w-md">
            <label className="block text-sm font-bold text-slate-700 mb-2">
              Client Name
            </label>
            <input
              type="text"
              value={clientName}
              onChange={(e) => setClientName(e.target.value)}
              placeholder="e.g. Bank A"
              className="w-full px-4 py-3 rounded-xl border border-slate-200 text-slate-800 focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>

          <button
            onClick={handleCreateApiKey}
            disabled={apiKeyCreating || !clientName.trim()}
            className="bg-blue-600 text-white px-5 py-3 rounded-xl text-sm font-bold hover:bg-blue-700 transition disabled:opacity-60"
          >
            {apiKeyCreating ? 'Generating...' : 'Generate API Key'}
          </button>
        </div>
      </SectionCard>

      <SectionCard
        title="API Keys"
        subtitle="Manage issued API keys for external systems"
        right={
          <button
            onClick={loadApiKeysOnly}
            className="bg-slate-900 text-white px-4 py-2 rounded-xl text-xs font-bold hover:bg-slate-700 transition"
          >
            Refresh Keys
          </button>
        }
      >
        <div className="overflow-x-auto">
          <table className="w-full text-left">
            <thead className="bg-slate-50 text-slate-400 uppercase text-[10px] font-black border-b">
              <tr>
                <th className="px-6 py-4">Client Name</th>
                <th className="px-6 py-4">Token Preview</th>
                <th className="px-6 py-4">Status</th>
                <th className="px-6 py-4">Created At</th>
                <th className="px-6 py-4">Expires At</th>
                <th className="px-6 py-4">Last Used At</th>
                <th className="px-6 py-4">Action</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100 text-sm">
              {apiKeys.length === 0 ? (
                <tr>
                  <td colSpan="7" className="px-6 py-10 text-center text-slate-400">
                    No API keys found.
                  </td>
                </tr>
              ) : (
                apiKeys.map((item) => (
                  <tr key={item._id} className="hover:bg-slate-50">
                    <td className="px-6 py-4 font-semibold text-slate-800">
                      {item.client_name || '-'}
                    </td>
                    <td className="px-6 py-4 text-slate-600 font-mono">
                      {item.token_preview || '-'}
                    </td>
                    <td className="px-6 py-4">
                      <span
                        className={`px-3 py-1 rounded-full text-xs font-black uppercase ${
                          item.status === 'active'
                            ? 'bg-green-50 text-green-600'
                            : 'bg-red-50 text-red-600'
                        }`}
                      >
                        {item.status || '-'}
                      </span>
                    </td>
                    <td className="px-6 py-4 text-slate-500">
                      {formatDate(item.created_at)}
                    </td>
                    <td className="px-6 py-4 text-slate-500">
                      {formatDate(item.expires_at)}
                    </td>
                    <td className="px-6 py-4 text-slate-500">
                      {formatDate(item.last_used_at)}
                    </td>
                    <td className="px-6 py-4">
                      <button
                        onClick={() => handleToggleApiKeyStatus(item._id, item.status)}
                        disabled={apiKeyStatusSavingId === item._id}
                        className={`px-4 py-2 rounded-xl text-xs font-bold transition ${
                          item.status === 'active'
                            ? 'bg-red-50 text-red-600 hover:bg-red-100'
                            : 'bg-green-50 text-green-600 hover:bg-green-100'
                        } disabled:opacity-60`}
                      >
                        {apiKeyStatusSavingId === item._id
                          ? 'Saving...'
                          : item.status === 'active'
                          ? 'Deactivate'
                          : 'Activate'}
                      </button>
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </SectionCard>

      <SectionCard
        title="Request Logs"
        subtitle="Recent API requests made by external clients"
        right={
          <button
            onClick={loadRequestLogsOnly}
            className="bg-slate-900 text-white px-4 py-2 rounded-xl text-xs font-bold hover:bg-slate-700 transition"
          >
            Refresh Logs
          </button>
        }
      >
        <div className="overflow-x-auto">
          <table className="w-full text-left">
            <thead className="bg-slate-50 text-slate-400 uppercase text-[10px] font-black border-b">
              <tr>
                <th className="px-6 py-4">Client</th>
                <th className="px-6 py-4">Endpoint</th>
                <th className="px-6 py-4">Method</th>
                <th className="px-6 py-4">Status</th>
                <th className="px-6 py-4">Time</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100 text-sm">
              {requestLogs.length === 0 ? (
                <tr>
                  <td colSpan="5" className="px-6 py-10 text-center text-slate-400">
                    No request logs found.
                  </td>
                </tr>
              ) : (
                requestLogs.map((log) => (
                  <tr key={log._id} className="hover:bg-slate-50">
                    <td className="px-6 py-4 font-semibold text-slate-800">
                      {log.client_name || '-'}
                    </td>
                    <td className="px-6 py-4 text-slate-700">
                      {log.endpoint || '-'}
                    </td>
                    <td className="px-6 py-4 text-slate-700 uppercase">
                      {log.method || '-'}
                    </td>
                    <td className="px-6 py-4">
                      <span
                        className={`px-3 py-1 rounded-full text-xs font-black uppercase ${
                          log.status === 'success'
                            ? 'bg-green-50 text-green-600'
                            : 'bg-red-50 text-red-600'
                        }`}
                      >
                        {log.status || '-'}
                      </span>
                    </td>
                    <td className="px-6 py-4 text-slate-500">
                      {formatDate(log.created_at)}
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </SectionCard>

      {tokenModal && (
        <div className="fixed inset-0 z-[60] bg-black/60 flex items-center justify-center p-6">
          <div className="bg-white w-full max-w-2xl rounded-[2rem] shadow-2xl border border-slate-200 overflow-hidden">
            <div className="px-8 py-6 border-b border-slate-200 bg-white">
              <h2 className="text-2xl font-black text-slate-800">New API Key Created</h2>
              <p className="text-sm text-slate-500 font-medium mt-1">
                Copy this token now. It will not be shown again.
              </p>
            </div>

            <div className="p-8 space-y-6">
              <div className="bg-amber-50 border border-amber-200 text-amber-700 rounded-2xl p-4 text-sm font-semibold">
                For security reasons, only the token hash is stored in the database. If this token is lost, deactivate the key and generate a new one.
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
                <div className="bg-slate-50 border border-slate-200 rounded-2xl p-4">
                  <p className="text-xs uppercase tracking-widest text-slate-400 font-bold mb-2">
                    Client Name
                  </p>
                  <p className="text-sm font-semibold text-slate-800">
                    {tokenModal.client_name}
                  </p>
                </div>

                <div className="bg-slate-50 border border-slate-200 rounded-2xl p-4">
                  <p className="text-xs uppercase tracking-widest text-slate-400 font-bold mb-2">
                    Expires At
                  </p>
                  <p className="text-sm font-semibold text-slate-800">
                    {formatDate(tokenModal.expires_at)}
                  </p>
                </div>
              </div>

              <div>
                <label className="block text-sm font-bold text-slate-700 mb-2">
                  Full API Token
                </label>
                <textarea
                  readOnly
                  value={tokenModal.token || ''}
                  rows={4}
                  className="w-full px-4 py-3 rounded-2xl border border-slate-200 bg-slate-50 text-slate-800 font-mono text-sm focus:outline-none"
                />
              </div>

              <div className="flex flex-wrap gap-3">
                <button
                  onClick={copyTokenToClipboard}
                  className="bg-blue-600 text-white px-5 py-3 rounded-xl text-sm font-bold hover:bg-blue-700 transition"
                >
                  Copy Token
                </button>

                <button
                  onClick={closeTokenModal}
                  className="bg-slate-900 text-white px-5 py-3 rounded-xl text-sm font-bold hover:bg-slate-700 transition"
                >
                  Close
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
