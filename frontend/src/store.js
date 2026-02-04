/**
 * =============================================================================
 * TrustLens Frontend - State Management Store
 * =============================================================================
 * 
 * This module manages the global application state using Zustand.
 * Zustand is a lightweight state management library that provides:
 * - Simple API without boilerplate
 * - React hooks for accessing state
 * - Built-in performance optimizations
 * 
 * State Structure:
 * - history: Array of recent transaction analysis results (max 10)
 * 
 * File: store.js
 * =============================================================================
 */

import { create } from 'zustand';

/**
 * Transaction Store Hook
 * 
 * Manages the history of analyzed transactions for the monitoring dashboard.
 * Uses Zustand's create function to define state and actions.
 * 
 * @returns {Object} Store object with state and actions:
 *   - history: Transaction[] - Array of recent transactions (newest first)
 *   - addTransaction: (tx) => void - Add a new transaction to history
 * 
 * @example
 * // In a React component:
 * const { history, addTransaction } = useTransactionStore();
 * 
 * // Add a new transaction
 * addTransaction({ id: 1, status: 'APPROVED', risk_score: 15 });
 * 
 * // Access transaction history
 * history.forEach(tx => console.log(tx.status));
 */
export const useTransactionStore = create((set) => ({
  /**
   * Transaction history array.
   * Stores the most recent 10 analyzed transactions.
   * Newest transactions appear first (index 0).
   */
  history: [],
  
  /**
   * Add a new transaction to the history.
   * 
   * The new transaction is prepended to the array (newest first),
   * and the array is trimmed to keep only the 10 most recent entries.
   * This prevents unbounded memory growth during long sessions.
   * 
   * @param {Object} tx - Transaction object from API response
   * @param {number} tx.id - Unique transaction identifier
   * @param {string} tx.status - "APPROVED" or "BLOCKED"
   * @param {number} tx.risk_score - Risk percentage (0-100)
   * @param {number} tx.Amount - Transaction amount
   * @param {string} tx.explanation - Human-readable XAI explanation
   * @param {Array} tx.chartData - SHAP feature data for visualization
   * @param {string} tx.time_captured - Timestamp when analyzed
   */
  addTransaction: (tx) => set((state) => ({ 
    history: [tx, ...state.history].slice(0, 10) // Keep only last 10 transactions
  })),
}));