import { create } from 'zustand';

export const useTransactionStore = create((set) => ({
  history: [], // Initialize as an empty array
  addTransaction: (tx) => set((state) => ({ 
    history: [tx, ...state.history].slice(0, 10) 
  })),
}));