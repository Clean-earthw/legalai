'use client'
import React, { useState, useEffect, useCallback } from 'react';
import { useUser, useAuth } from '@clerk/nextjs';
import { AuthenticatedApiService, UploadResult, Document, UserStats } from '@/services/apiService';
import FileUpload from '@/components/FileUpload';
import ChatInterface from '@/components/ChatInterface';
import DocumentSearch from '@/components/DocumentSearch';
import Statistics from '@/components/Statistics';
import Sidebar from '@/components/Sidebar';
import DashboardNavbar from '@/components/DashboardNavbar';

interface DashboardState {
  isUploading: boolean;
  documents: Document[];
  userStats: UserStats | null;
  loading: boolean;
  error: string | null;
  activeView: 'chat' | 'search' | 'stats';
}

const Dashboard: React.FC = () => {
  const { user, isLoaded: userLoaded } = useUser();
  const { getToken } = useAuth();
  
  // Consolidated state management
  const [state, setState] = useState<DashboardState>({
    isUploading: false,
    documents: [],
    userStats: null,
    loading: true,
    error: null,
    activeView: 'chat'
  });

  const [apiService, setApiService] = useState<AuthenticatedApiService | null>(null);

  // Initialize API service
  useEffect(() => {
    if (!userLoaded) return;

    if (!user) {
      setState(prev => ({
        ...prev,
        loading: false,
        error: 'User not authenticated'
      }));
      return;
    }

    try {
      const service = new AuthenticatedApiService(
        user.id,
        undefined,
        getToken
      );

      service.checkHealth()
        .then(() => {
          setApiService(service);
        })
        .catch(healthError => {
          setState(prev => ({
            ...prev,
            error: `API service unavailable: ${healthError.message}`,
            loading: false
          }));
        });

    } catch (initError: any) {
      setState(prev => ({
        ...prev,
        error: `Failed to initialize API service: ${initError.message}`,
        loading: false
      }));
    }
  }, [userLoaded, user, getToken]);

  // Load dashboard data
  const loadDashboardData = useCallback(async (service: AuthenticatedApiService) => {
    if (!service || !user) return;

    try {
      setState(prev => ({
        ...prev,
        loading: true,
        error: null
      }));

      let docsResponse: Document[] = [];
      let statsResponse: UserStats | null = null;
      
      try {
        docsResponse = await service.getDocuments();
        
        if (!Array.isArray(docsResponse)) {
          docsResponse = [];
        }
        
      } catch (docError: any) {
        docsResponse = [];
        setState(prev => ({
          ...prev,
          error: `Failed to load documents: ${docError.message}`
        }));
      }

      try {
        statsResponse = await service.getUserStats();
      } catch (statsError) {
        statsResponse = null;
      }
      
      setState(prev => ({
        ...prev,
        documents: docsResponse,
        userStats: statsResponse,
        loading: false
      }));

    } catch (err: any) {
      setState(prev => ({
        ...prev,
        loading: false,
        error: err.message || 'Failed to load dashboard data'
      }));
    }
  }, [user]);

  // Load data when API service is ready
  useEffect(() => {
    if (apiService) {
      loadDashboardData(apiService);
    }
  }, [apiService, loadDashboardData]);

  // Handle successful upload with refresh
  const handleUploadSuccess = useCallback((result: UploadResult) => {
    setState(prev => ({
      ...prev,
      isUploading: false
    }));

    if (apiService) {
      setTimeout(() => {
        loadDashboardData(apiService);
      }, 1000);
    }
  }, [apiService, loadDashboardData]);

  // Handle document deletion
  const handleDeleteDocument = useCallback(async (documentId: string) => {
    if (!apiService) return;

    try {
      await apiService.deleteDocument(documentId);
      await loadDashboardData(apiService);
    } catch (err: any) {
      setState(prev => ({
        ...prev,
        error: `Failed to delete document: ${err.message}`
      }));
    }
  }, [apiService, loadDashboardData]);

  // Handle manual refresh
  const handleRefresh = useCallback(() => {
    if (apiService) {
      loadDashboardData(apiService);
    }
  }, []);

  // Set uploading state
  const setIsUploading = useCallback((uploading: boolean) => {
    setState(prev => ({
      ...prev,
      isUploading: uploading
    }));
  }, []);

  // Handle view changes
  const handleViewChange = useCallback((view: 'chat' | 'search' | 'stats') => {
    setState(prev => ({
      ...prev,
      activeView: view
    }));
  }, []);

  // Loading state
  if (!userLoaded || state.loading) {
    return (
      <div className="min-h-screen bg-gray-50">
        <DashboardNavbar />
        <div className="flex flex-col justify-center items-center h-64">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mb-4"></div>
          <p className="text-gray-600">
            {!userLoaded ? 'Loading user...' : 'Loading dashboard...'}
          </p>
        </div>
      </div>
    );
  }

  // Error state with retry functionality
  if (state.error) {
    return (
      <div className="min-h-screen bg-gray-50">
        <DashboardNavbar />
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          <div className="bg-red-50 border border-red-200 rounded-lg p-6">
            <h3 className="text-lg font-semibold text-red-800 mb-2">Error Loading Dashboard</h3>
            <p className="text-red-700 mb-4">{state.error}</p>
            
            <div className="space-x-2">
              <button 
                onClick={() => apiService && loadDashboardData(apiService)}
                disabled={!apiService}
                className="px-4 py-2 bg-red-600 text-white rounded hover:bg-red-700 disabled:opacity-50"
              >
                Retry Loading Data
              </button>
              
              <button 
                onClick={() => window.location.reload()}
                className="px-4 py-2 bg-gray-600 text-white rounded hover:bg-gray-700"
              >
                Reload Page
              </button>
            </div>
          </div>
        </div>
      </div>
    );
  }

  // Not ready state
  if (!apiService) {
    return (
      <div className="min-h-screen bg-gray-50">
        <DashboardNavbar />
        <div className="flex justify-center items-center h-64">
          <p className="text-gray-600">Initializing API service...</p>
        </div>
      </div>
    );
  }

  const renderMainContent = () => {
    switch (state.activeView) {
      case 'chat':
        return (
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 h-full">
            {/* Left Side - File Upload and Documents */}
            <div className="lg:col-span-1 space-y-6 overflow-y-auto pr-2">
              <FileUpload
                apiService={apiService}
                onUploadSuccess={handleUploadSuccess}
                isUploading={state.isUploading}
                setIsUploading={setIsUploading}
              />

              <div className="bg-white rounded-lg shadow-lg p-6">
                <div className="flex justify-between items-center mb-4">
                  <h3 className="text-xl font-semibold text-gray-800">Documents</h3>
                  <button
                    onClick={handleRefresh}
                    className="text-sm text-blue-600 hover:text-blue-800 transition-colors"
                  >
                    Refresh
                  </button>
                </div>
                
                {state.documents.length === 0 ? (
                  <div className="text-center text-gray-500 py-8">
                    <div className="text-4xl mb-4">📚</div>
                    <p>No documents found.</p>
                    <p className="text-sm mt-2">Upload your first document to get started!</p>
                  </div>
                ) : (
                  <div className="space-y-3 max-h-96 overflow-y-auto pr-2">
                    {state.documents.slice(0, 5).map((doc) => (
                      <div key={doc.document_id} className="border border-gray-200 rounded-lg p-3 hover:bg-gray-50 transition-colors">
                        <div className="flex items-center">
                          <div className="text-xl mr-2">
                            {doc.document_type === 'pdf' ? '📄' : 
                             doc.document_type === 'docx' ? '📝' : '📋'}
                          </div>
                          <div className="flex-1">
                            <div className="text-sm font-medium text-gray-900 truncate">
                              {doc.document_name}
                            </div>
                            <div className="text-xs text-gray-500">
                              {doc.chunks_count} chunks • {doc.document_type.toUpperCase()}
                            </div>
                            {doc.created_at && (
                              <div className="text-xs text-gray-400">
                                {new Date(doc.created_at).toLocaleDateString()}
                              </div>
                            )}
                          </div>
                        </div>
                      </div>
                    ))}
                    
                    {state.documents.length > 5 && (
                      <button 
                        onClick={() => handleViewChange('search')}
                        className="w-full text-center text-blue-600 hover:text-blue-800 text-sm py-2 transition-colors"
                      >
                        View all {state.documents.length} documents →
                      </button>
                    )}
                  </div>
                )}
              </div>
            </div>

            {/* Right Side - Legal Assistant Chat */}
            <div className="lg:col-span-2">
              <div className="bg-white rounded-lg shadow-lg p-8 h-full flex flex-col">
                <div className="mb-6 text-center border-b-2 border-gray-200 pb-4 flex-shrink-0">
                  <h2 className="text-3xl font-bold text-gray-900 tracking-wide">
                    <span className="block text-sm font-normal text-gray-600 mb-2">DIGITAL</span>
                    <span className="block text-4xl font-black tracking-wider">LEGAL COUNSEL</span>
                    <span className="block text-sm font-normal text-gray-600 mt-2 italic">AI-Powered Legal Assistant</span>
                  </h2>
                </div>
                
                <div className="flex-1 flex flex-col min-h-0">
                  <ChatInterface apiService={apiService} />
                </div>
              </div>
            </div>
          </div>
        );
      
      case 'search':
        return (
          <div className="h-full overflow-y-auto">
            <DocumentSearch 
              documents={state.documents}
              apiService={apiService}
              onDocumentDelete={handleDeleteDocument}
              onRefresh={handleRefresh}
            />
          </div>
        );
      
      case 'stats':
        return (
          <div className="h-full overflow-y-auto">
            <Statistics userStats={state.userStats} loading={state.loading} />
          </div>
        );
      
      default:
        return null;
    }
  };

  return (
    <div className="h-screen bg-gray-50 flex flex-col overflow-hidden">
      {/* Fixed Navbar */}
      <div className="flex-shrink-0">
        <DashboardNavbar />
      </div>
      
      {/* Main Layout Container */}
      <div className="flex-1 flex overflow-hidden">
        {/* Fixed Sidebar */}
        <div className="w-64 bg-white border-r border-gray-200 flex-shrink-0 overflow-y-auto">
          <div className="p-4">
            {/* Welcome Section - Fixed at top */}
            <div className="mb-6 pb-6 border-b border-gray-100">
              <h1 className="text-xl font-bold text-gray-900 truncate">
                Welcome, {user?.firstName || 'User'}!
              </h1>
              <p className="text-sm text-gray-600 mt-1">
                Your legal assistant dashboard
              </p>
            </div>
            
            {/* Sidebar Component */}
            <Sidebar 
              activeView={state.activeView}
              onViewChange={handleViewChange}
              userStats={state.userStats}
            />
          </div>
        </div>

        {/* Main Content Area - Scrollable */}
        <div className="flex-1 flex flex-col overflow-hidden">
          {/* Content Header */}
          <div className="flex-shrink-0 p-6 bg-white border-b border-gray-200">
            <h2 className="text-2xl font-semibold text-gray-900 capitalize">
              {state.activeView === 'chat' ? 'Legal Assistant' : 
               state.activeView === 'search' ? 'Document Search' : 
               'Statistics'}
            </h2>
            <p className="text-gray-600 mt-1">
              {state.activeView === 'chat' ? 'Chat with your AI legal assistant and manage documents' : 
               state.activeView === 'search' ? 'Search and manage your uploaded documents' : 
               'View your usage statistics and analytics'}
            </p>
          </div>

          {/* Scrollable Main Content */}
          <div className="flex-1 overflow-y-auto">
            <div className="p-6 h-full">
              {renderMainContent()}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;