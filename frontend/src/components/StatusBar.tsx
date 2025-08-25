import React from 'react';

interface StatusBarProps {
  apiStatus: 'healthy' | 'degraded' | 'unhealthy' | 'unknown';
  documentCount: number;
  lastUpload?: string | null;
}

const StatusBar: React.FC<StatusBarProps> = ({ apiStatus, documentCount, lastUpload }) => {
  const getStatusColor = (status: string) => {
    switch (status) {
      case 'healthy':
        return 'bg-green-500';
      case 'degraded':
        return 'bg-yellow-500';
      case 'unhealthy':
        return 'bg-red-500';
      default:
        return 'bg-gray-500';
    }
  };

  const getStatusText = (status: string) => {
    switch (status) {
      case 'healthy':
        return 'Connected';
      case 'degraded':
        return 'Limited';
      case 'unhealthy':
        return 'Disconnected';
      default:
        return 'Unknown';
    }
  };

  return (
    <div className="bg-white rounded-lg shadow-lg p-4 mb-6">
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-4">
          <div className="flex items-center space-x-2">
            <div className={`w-3 h-3 rounded-full ${getStatusColor(apiStatus)}`}></div>
            <span className="text-sm font-medium">
              API: {getStatusText(apiStatus)}
            </span>
          </div>
          <div className="text-sm text-gray-600">
            📄 {documentCount} documents stored
          </div>
        </div>
        {lastUpload && (
          <div className="text-sm text-gray-500">
            Last upload: {lastUpload}
          </div>
        )}
      </div>
    </div>
  );
};

export default StatusBar;