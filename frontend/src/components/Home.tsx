'use client'
import React, { useState, useEffect } from 'react';
import Head from 'next/head';
import FileUpload from '../components/FileUpload';
import DocumentManager from './DocumentService';
import { apiService } from '@/lib/service';
import ChatInterface from '../components/ChatInterface';
import StatusBar from '../components/StatusBar';

// Import types from the correct location
import { Document, UploadResult } from '@/lib/service'; // Changed from @/types/types

export default function Home() {
  const [documents, setDocuments] = useState<Document[]>([]);
  const [apiStatus, setApiStatus] = useState<'healthy' | 'degraded' | 'unhealthy' | 'unknown'>('unknown'); // Fixed type
  const [isUploading, setIsUploading] = useState(false);
  const [lastUpload, setLastUpload] = useState<string | null>(null);

  // Check API status and load documents
  const checkApiStatus = async () => {
    try {
      const health = await apiService.checkHealth();
      setApiStatus(health.status === 'healthy' ? 'healthy' : 'degraded');
    } catch (error) {
      setApiStatus('unhealthy');
    }
  };

  const loadDocuments = async () => {
    try {
      const response = await apiService.getDocuments();
      setDocuments(response.documents || []);
    } catch (error) {
      console.error('Failed to load documents:', error);
      setDocuments([]); // Set empty array on error
    }
  };

  const handleUploadSuccess = (result: UploadResult) => {
    console.log('Upload successful:', result);
    setLastUpload(new Date().toLocaleString());
    loadDocuments(); // Refresh document list
    alert(`Document "${result.document_name}" uploaded successfully! ${result.chunks_added} chunks created.`);
  };

  const handleDocumentDelete = (documentId: string) => {
    setDocuments(prev => prev.filter(doc => doc.document_id !== documentId));
  };

  // Initialize on component mount
  useEffect(() => {
    checkApiStatus();
    loadDocuments();
  }, []);

  return (
    <div className="min-h-screen bg-gradient-to-br from-indigo-500 via-purple-600 to-pink-500">
      <Head>
        <title>Legal Support AI</title>
        <meta name="description" content="Upload documents and get intelligent legal assistance" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <link rel="icon" href="/favicon.ico" />
      </Head>

      <div className="container mx-auto px-4 py-8">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-white mb-2">⚖️ Legal Support AI</h1>
          <p className="text-xl text-gray-200">Upload documents and get intelligent legal assistance</p>
        </div>

        {/* Status Bar */}
        <StatusBar 
          apiStatus={apiStatus}
          documentCount={documents.length}
          lastUpload={lastUpload}
        />

        {/* Main Content Grid */}
        <div className="grid lg:grid-cols-2 gap-8">
          {/* Left Column */}
          <div className="space-y-6">
            <FileUpload 
              onUploadSuccess={handleUploadSuccess}
              isUploading={isUploading}
              setIsUploading={setIsUploading}
            />
            <DocumentManager 
              documents={documents}
              onDocumentDelete={handleDocumentDelete}
              onRefresh={loadDocuments}
            />
          </div>

          {/* Right Column */}
          <div>
            <ChatInterface />
          </div>
        </div>

        {/* Footer */}
        <div className="mt-12 text-center text-white">
          <p className="text-sm opacity-80">
            Powered by AI • Supports PDF & DOCX • Vector Search Enabled
          </p>
        </div>
      </div>
    </div>
  );
}