import React from 'react';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell, CartesianGrid } from 'recharts';

const XAIChart = ({ explanationData }) => {
  // If no data, show a placeholder
  if (!explanationData || explanationData.length === 0) {
    return (
      <div className="h-64 w-full flex items-center justify-center text-slate-300 italic text-sm border-2 border-dashed border-slate-100 rounded-2xl">
        Awaiting model impact values...
      </div>
    );
  }

  return (
    <div className="h-64 w-full bg-transparent">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart 
          data={explanationData} 
          layout="vertical" 
          margin={{ left: 10, right: 30, top: 10, bottom: 10 }}
        >
          <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke="#f1f5f9" />
          <XAxis type="number" domain={[0, 'auto']} hide />
          <YAxis 
            dataKey="name" 
            type="category" 
            width={50} 
            tick={{ fontSize: 11, fontWeight: 800, fill: '#64748b' }} 
            axisLine={false}
            tickLine={false}
          />
          <Tooltip 
            cursor={{ fill: '#f8fafc' }}
            contentStyle={{ 
              borderRadius: '12px', 
              border: 'none', 
              boxShadow: '0 10px 15px -3px rgb(0 0 0 / 0.1)',
              fontSize: '12px',
              fontWeight: 'bold'
            }}
          />
          <Bar dataKey="impact" radius={[0, 10, 10, 0]} barSize={32}>
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