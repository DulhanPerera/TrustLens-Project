/**
 * =============================================================================
 * TrustLens Frontend - XAI (Explainable AI) Chart Component
 * =============================================================================
 * 
 * This component visualizes SHAP (SHapley Additive exPlanations) feature
 * attributions using a horizontal bar chart. It helps users understand
 * which features most influenced the fraud prediction model's decision.
 * 
 * Features:
 * - Horizontal bar chart layout for easy feature name readability
 * - Color-coded bars: Red for high impact (>0.5), Blue for lower impact
 * - Responsive design that adapts to container width
 * - Placeholder state when no data is available
 * 
 * Dependencies:
 * - Recharts: React charting library for data visualization
 * 
 * Use Case Reference: UC-12 (Explainable AI Visualization)
 * 
 * File: components/XAIChart.jsx
 * =============================================================================
 */

import React from 'react';
import { 
  BarChart,           // Main chart component
  Bar,                // Bar element for data visualization
  XAxis,              // Horizontal axis (shows impact values)
  YAxis,              // Vertical axis (shows feature names)
  Tooltip,            // Hover tooltip for detailed values
  ResponsiveContainer,// Makes chart responsive to parent container
  Cell,               // Individual bar cell for custom colors
  CartesianGrid       // Grid lines for better readability
} from 'recharts';

/**
 * XAIChart Component
 * 
 * Renders a horizontal bar chart showing SHAP feature attributions.
 * Each bar represents a feature's contribution to the fraud prediction.
 * 
 * @component
 * @param {Object} props - Component props
 * @param {Array} props.explanationData - SHAP feature data from API
 * @param {string} props.explanationData[].name - Feature name (e.g., "V14", "Amount")
 * @param {number} props.explanationData[].impact - Absolute SHAP value (contribution magnitude)
 * 
 * @returns {JSX.Element} Horizontal bar chart or placeholder message
 * 
 * @example
 * // Sample data from backend /predict endpoint (xai_data field)
 * const shapData = [
 *   { name: "V14", impact: 0.85 },
 *   { name: "V12", impact: 0.62 },
 *   { name: "Amount", impact: 0.45 },
 *   { name: "V10", impact: 0.23 }
 * ];
 * 
 * <XAIChart explanationData={shapData} />
 */
const XAIChart = ({ explanationData }) => {
  // ==========================================================================
  // Empty State Handler
  // ==========================================================================
  // Display a placeholder when no XAI data is available (e.g., before first scan)
  if (!explanationData || explanationData.length === 0) {
    return (
      <div className="h-64 w-full flex items-center justify-center text-slate-300 italic text-sm border-2 border-dashed border-slate-100 rounded-2xl">
        Awaiting model impact values...
      </div>
    );
  }

  // ==========================================================================
  // Chart Rendering
  // ==========================================================================
  return (
    <div className="h-64 w-full bg-transparent">
      {/* ResponsiveContainer ensures the chart scales with its parent element */}
      <ResponsiveContainer width="100%" height="100%">
        <BarChart 
          data={explanationData} 
          layout="vertical"  // Horizontal bars (features on Y-axis, impact on X-axis)
          margin={{ left: 10, right: 30, top: 10, bottom: 10 }}
        >
          {/* Grid lines for easier value reading (horizontal only) */}
          <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke="#f1f5f9" />
          
          {/* X-Axis: Shows impact values (hidden for cleaner look) */}
          <XAxis type="number" domain={[0, 'auto']} hide />
          
          {/* Y-Axis: Shows feature names (V1-V28, Time, Amount) */}
          <YAxis 
            dataKey="name" 
            type="category" 
            width={50} 
            tick={{ fontSize: 11, fontWeight: 800, fill: '#64748b' }} 
            axisLine={false}   // Hide axis line for cleaner look
            tickLine={false}   // Hide tick marks
          />
          
          {/* Tooltip: Shows detailed values on hover */}
          <Tooltip 
            cursor={{ fill: '#f8fafc' }}  // Light hover background
            contentStyle={{ 
              borderRadius: '12px', 
              border: 'none', 
              boxShadow: '0 10px 15px -3px rgb(0 0 0 / 0.1)',
              fontSize: '12px',
              fontWeight: 'bold'
            }}
          />
          
          {/* Bar: The actual data visualization */}
          <Bar dataKey="impact" radius={[0, 10, 10, 0]} barSize={32}>
            {/* 
              Color each bar based on impact value:
              - Red (#ef4444): High impact features (>0.5) - stronger fraud indicators
              - Blue (#3b82f6): Lower impact features - weaker influence on decision
            */}
            {explanationData.map((entry, index) => (
              <Cell 
                key={`cell-${index}`} 
                fill={entry.impact > 0.5 ? '#ef4444' : '#3b82f6'} 
                fillOpacity={0.8}
              />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
};

export default XAIChart;