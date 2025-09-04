'use client'
import React from 'react';
import { useUser } from '@clerk/nextjs';
import { UserStats } from '@/services/apiService';

interface SidebarProps {
  activeView: 'chat' | 'search' | 'stats';
  onViewChange: (view: 'chat' | 'search' | 'stats') => void;
  userStats?: UserStats | null;
}

const Sidebar: React.FC<SidebarProps> = ({ activeView, onViewChange, userStats }) => {
  const { user } = useUser();

  const menuItems = [
    {
      id: 'chat' as const,
      icon: '💬',
      label: 'Legal Chat',
      description: 'AI Assistant'
    },
    {
      id: 'search' as const,
      icon: '🔍',
      label: 'Document Search',
      description: 'Search & Manage'
    },
    {
      id: 'stats' as const,
      icon: '📊',
      label: 'Statistics',
      description: 'Usage Analytics'
    }
  ];

  return (
    <div className="bg-white shadow-lg rounded-lg p-4 flex flex-col h-full">
      {/* Navigation Menu */}
      <div className="space-y-2 mb-8">
        {menuItems.map((item) => (
          <button
            key={item.id}
            onClick={() => onViewChange(item.id)}
            className={`w-full text-left p-3 rounded-lg transition-all duration-200 ${
              activeView === item.id
                ? 'bg-gradient-to-r from-blue-500 to-purple-600 text-white shadow-md'
                : 'hover:bg-gray-100 text-gray-700'
            }`}
          >
            <div className="flex items-center space-x-3">
              <span className="text-2xl">{item.icon}</span>
              <div className="flex-1">
                <div className="font-medium">{item.label}</div>
                <div className={`text-xs ${
                  activeView === item.id ? 'text-blue-100' : 'text-gray-500'
                }`}>
                  {item.description}
                </div>
              </div>
            </div>
          </button>
        ))}
      </div>

      {/* Spacer */}
      <div className="flex-1"></div>

      {/* User Profile Section */}
      <div className="border-t border-gray-200 pt-4 mt-4">
        <div className="flex items-center space-x-3 p-3 bg-gray-50 rounded-lg">
          <div className="w-10 h-10 bg-gradient-to-r from-blue-500 to-purple-600 rounded-full flex items-center justify-center text-white font-bold">
            {user?.firstName?.charAt(0) || user?.emailAddresses[0]?.emailAddress?.charAt(0) || 'U'}
          </div>
          <div className="flex-1 min-w-0">
            <div className="font-medium text-gray-900 truncate">
              {user?.firstName || 'User'}
            </div>
            <div className="text-xs text-gray-500 truncate">
              {user?.emailAddresses[0]?.emailAddress}
            </div>
          </div>
        </div>

        {/* Quick Stats */}
        {userStats && (
          <div className="mt-3 space-y-2">
            <div className="flex justify-between items-center text-sm">
              <span className="text-gray-600">Documents:</span>
              <span className="font-semibold text-blue-600">
                {userStats.total_documents}
              </span>
            </div>
            <div className="flex justify-between items-center text-sm">
              <span className="text-gray-600">Storage:</span>
              <span className="font-semibold text-green-600">
                {(userStats.total_file_size / (1024 * 1024)).toFixed(1)}MB
              </span>
            </div>
            <div className="flex justify-between items-center text-sm">
              <span className="text-gray-600">Sessions:</span>
              <span className="font-semibold text-purple-600">
                {userStats.sessions.length}
              </span>
            </div>
          </div>
        )}

        {/* User Actions */}
        <div className="mt-3 pt-3 border-t border-gray-200">
          <button className="w-full text-left p-2 text-sm text-gray-600 hover:text-gray-800 hover:bg-gray-100 rounded transition-colors">
            ⚙️ Settings
          </button>
          <button className="w-full text-left p-2 text-sm text-gray-600 hover:text-gray-800 hover:bg-gray-100 rounded transition-colors">
            📋 Help & Support
          </button>
        </div>
      </div>
    </div>
  );
};

export default Sidebar;