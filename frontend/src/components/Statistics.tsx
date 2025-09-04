'use client'
import React from 'react';
import { UserStats } from '@/services/apiService';

interface StatisticsProps {
  userStats: UserStats | null;
  loading?: boolean;
}

const Statistics: React.FC<StatisticsProps> = ({ userStats, loading }) => {
  if (loading) {
    return (
      <div className="bg-white rounded-lg shadow-lg p-8 h-full flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
          <p className="text-gray-600">Loading statistics...</p>
        </div>
      </div>
    );
  }

  if (!userStats) {
    return (
      <div className="bg-white rounded-lg shadow-lg p-8 h-full flex items-center justify-center">
        <div className="text-center text-gray-500">
          <div className="text-4xl mb-4">📊</div>
          <p>No statistics available</p>
        </div>
      </div>
    );
  }

  // Calculate document type percentages
  const totalDocs = userStats.total_documents;
  const documentTypeData = Object.entries(userStats.document_types).map(([type, count]) => ({
    type: type.toUpperCase(),
    count,
    percentage: totalDocs > 0 ? (count / totalDocs * 100).toFixed(1) : '0'
  }));

  return (
    <div className="bg-white rounded-lg shadow-lg p-8 h-full overflow-y-auto">
      <div className="border-b-2 border-gray-200 pb-4 mb-6">
        <h2 className="text-3xl font-bold text-gray-900 text-center">
          <span className="block text-4xl mb-2">📊</span>
          Usage Analytics
        </h2>
      </div>

      {/* Overview Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
        <div className="bg-gradient-to-r from-blue-500 to-blue-600 text-white rounded-lg p-6">
          <div className="text-3xl font-bold">{userStats.total_documents}</div>
          <div className="text-blue-100 text-sm mt-1">Total Documents</div>
        </div>
        
        <div className="bg-gradient-to-r from-green-500 to-green-600 text-white rounded-lg p-6">
          <div className="text-3xl font-bold">{userStats.total_chunks}</div>
          <div className="text-green-100 text-sm mt-1">Document Chunks</div>
        </div>
        
        <div className="bg-gradient-to-r from-purple-500 to-purple-600 text-white rounded-lg p-6">
          <div className="text-3xl font-bold">{userStats.sessions.length}</div>
          <div className="text-purple-100 text-sm mt-1">Active Sessions</div>
        </div>
        
        <div className="bg-gradient-to-r from-orange-500 to-orange-600 text-white rounded-lg p-6">
          <div className="text-3xl font-bold">
            {(userStats.total_file_size / (1024 * 1024)).toFixed(1)}MB
          </div>
          <div className="text-orange-100 text-sm mt-1">Storage Used</div>
        </div>
      </div>

      {/* Document Types Analysis */}
      <div className="bg-gray-50 rounded-lg p-6 mb-8">
        <h3 className="text-xl font-semibold text-gray-800 mb-4">Document Types</h3>
        
        {documentTypeData.length === 0 ? (
          <div className="text-center text-gray-500 py-8">
            <p>No documents to analyze</p>
          </div>
        ) : (
          <div className="space-y-4">
            {documentTypeData.map(({ type, count, percentage }) => (
              <div key={type} className="flex items-center justify-between">
                <div className="flex items-center space-x-3">
                  <div className="text-2xl">
                    {type === 'PDF' ? '📄' : type === 'DOCX' ? '📝' : '📋'}
                  </div>
                  <div>
                    <div className="font-medium text-gray-800">{type}</div>
                    <div className="text-sm text-gray-600">{count} documents</div>
                  </div>
                </div>
                <div className="flex items-center space-x-3">
                  <div className="w-32 bg-gray-200 rounded-full h-2">
                    <div 
                      className="bg-blue-600 h-2 rounded-full transition-all duration-500"
                      style={{ width: `${percentage}%` }}
                    ></div>
                  </div>
                  <span className="text-sm font-semibold text-gray-700 w-12 text-right">
                    {percentage}%
                  </span>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Session Information */}
      <div className="bg-gray-50 rounded-lg p-6 mb-8">
        <h3 className="text-xl font-semibold text-gray-800 mb-4">Session Details</h3>
        
        {userStats.sessions.length === 0 ? (
          <div className="text-center text-gray-500 py-8">
            <p>No active sessions</p>
          </div>
        ) : (
          <div className="space-y-3">
            {userStats.sessions.map((sessionId, index) => (
              <div key={sessionId} className="bg-white rounded-lg p-4 border border-gray-200">
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-3">
                    <div className="w-8 h-8 bg-purple-100 rounded-full flex items-center justify-center">
                      <span className="text-purple-600 font-semibold text-sm">
                        {index + 1}
                      </span>
                    </div>
                    <div>
                      <div className="font-medium text-gray-800">
                        Session {index + 1}
                      </div>
                      <div className="text-sm text-gray-600">
                        ID: {sessionId.substring(0, 8)}...
                      </div>
                    </div>
                  </div>
                  <span className="px-2 py-1 bg-green-100 text-green-800 rounded-full text-xs font-semibold">
                    Active
                  </span>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Storage Breakdown */}
      <div className="bg-gray-50 rounded-lg p-6">
        <h3 className="text-xl font-semibold text-gray-800 mb-4">Storage Analysis</h3>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="text-center">
            <div className="text-2xl font-bold text-blue-600 mb-2">
              {(userStats.total_file_size / (1024 * 1024)).toFixed(2)} MB
            </div>
            <div className="text-sm text-gray-600">Total Storage Used</div>
          </div>
          
          <div className="text-center">
            <div className="text-2xl font-bold text-green-600 mb-2">
              {totalDocs > 0 ? (userStats.total_file_size / totalDocs / 1024).toFixed(1) : '0'} KB
            </div>
            <div className="text-sm text-gray-600">Average File Size</div>
          </div>
        </div>

        {/* Storage Bar */}
        <div className="mt-6">
          <div className="flex justify-between items-center mb-2">
            <span className="text-sm text-gray-600">Storage Usage</span>
            <span className="text-sm text-gray-600">
              {(userStats.total_file_size / (1024 * 1024)).toFixed(1)} MB / 100 MB
            </span>
          </div>
          <div className="w-full bg-gray-200 rounded-full h-3">
            <div 
              className="bg-gradient-to-r from-blue-500 to-purple-600 h-3 rounded-full transition-all duration-500"
              style={{ 
                width: `${Math.min((userStats.total_file_size / (100 * 1024 * 1024)) * 100, 100)}%` 
              }}
            ></div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Statistics;