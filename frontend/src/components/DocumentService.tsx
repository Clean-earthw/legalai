'use client'
import React, { useState } from 'react';
import { apiService, Document, SearchResult } from '@/lib/service';

interface DocumentManagerProps {
  documents: Document[];
  onDocumentDelete: (documentId: string) => void;
  onRefresh: () => void;
}

const DocumentManager: React.FC<DocumentManagerProps> = ({ 
  documents, 
  onDocumentDelete, 
  onRefresh 
}) => {
  const [isDeleting, setIsDeleting] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState<SearchResult[]>([]);
  const [isSearching, setIsSearching] = useState(false);

  const handleDelete = async (documentId: string, documentName: string) => {
    if (!confirm(`Are you sure you want to delete "${documentName}"? This action cannot be undone.`)) {
      return;
    }

    setIsDeleting(documentId);
    try {
      await apiService.deleteDocument(documentId);
      onDocumentDelete(documentId);
      alert('Document deleted successfully');
    } catch (error: any) {
      console.error('Delete error:', error);
      alert(`Failed to delete document: ${error.message}`);
    } finally {
      setIsDeleting(null);
    }
  };

  const handleSearch = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!searchQuery.trim()) return;

    setIsSearching(true);
    try {
      const response = await apiService.searchDocuments(searchQuery, 10);
      setSearchResults(response.results);
    } catch (error: any) {
      console.error('Search error:', error);
      alert(`Search failed: ${error.message}`);
    } finally {
      setIsSearching(false);
    }
  };

  const clearSearch = () => {
    setSearchQuery('');
    setSearchResults([]);
  };

  return (
    <div className="bg-white rounded-lg shadow-lg p-6">
      <div className="flex justify-between items-center mb-6">
        <h3 className="text-xl font-semibold text-gray-800">Document Library</h3>
        <button
          onClick={onRefresh}
          className="px-3 py-1 text-sm bg-gray-100 hover:bg-gray-200 rounded-md transition-colors"
        >
          🔄 Refresh
        </button>
      </div>

      {/* Document Search */}
      <div className="mb-6">
        <form onSubmit={handleSearch} className="flex space-x-2">
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="flex-1 px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
            placeholder="Search in documents..."
            disabled={isSearching}
          />
          <button
            type="submit"
            disabled={isSearching || !searchQuery.trim()}
            className="px-4 py-2 bg-green-600 text-white rounded-md hover:bg-green-700 disabled:opacity-50 transition-colors"
          >
            {isSearching ? (
              <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white"></div>
            ) : (
              '🔍 Search'
            )}
          </button>
          {searchResults.length > 0 && (
            <button
              type="button"
              onClick={clearSearch}
              className="px-3 py-2 text-gray-600 hover:text-gray-800 transition-colors"
            >
              Clear
            </button>
          )}
        </form>
      </div>

      {/* Search Results */}
      {searchResults.length > 0 && (
        <div className="mb-6 bg-blue-50 rounded-lg p-4">
          <h4 className="font-semibold text-blue-800 mb-3">
            Search Results ({searchResults.length})
          </h4>
          <div className="space-y-3 max-h-64 overflow-y-auto">
            {searchResults.map((result, index) => (
              <div key={index} className="bg-white rounded p-3 border border-blue-200">
                <div className="flex justify-between items-start mb-2">
                  <h5 className="font-medium text-sm text-blue-700">
                    {result.document_name}
                  </h5>
                  {result.similarity_score && (
                    <span className="text-xs text-gray-500 bg-gray-100 px-2 py-1 rounded">
                      Score: {result.similarity_score.toFixed(2)}
                    </span>
                  )}
                </div>
                {result.section && (
                  <p className="text-xs text-gray-600 mb-1">Section: {result.section}</p>
                )}
                <p className="text-sm text-gray-700">
                  {result.text.length > 150 
                    ? result.text.substring(0, 150) + '...' 
                    : result.text}
                </p>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Document List */}
      <div>
        <h4 className="font-semibold mb-3 text-gray-700">
          All Documents ({documents.length})
        </h4>
        
        {documents.length === 0 ? (
          <div className="text-center text-gray-500 py-8">
            <div className="text-4xl mb-4">📚</div>
            <p>No documents uploaded yet.</p>
            <p className="text-sm mt-2">Upload your first document to get started!</p>
          </div>
        ) : (
          <div className="space-y-3 max-h-64 overflow-y-auto">
            {documents.map((doc) => (
              <div key={doc.document_id} className="flex items-center justify-between p-3 border border-gray-200 rounded-md hover:bg-gray-50 transition-colors">
                <div className="flex-1">
                  <h5 className="font-medium text-gray-800">{doc.document_name}</h5>
                  <p className="text-sm text-gray-600">
                    {doc.chunks_count} chunks • ID: {doc.document_id.substring(0, 8)}...
                  </p>
                </div>
                <button
                  onClick={() => handleDelete(doc.document_id, doc.document_name)}
                  disabled={isDeleting === doc.document_id}
                  className="px-3 py-1 text-red-600 hover:bg-red-50 rounded-md transition-colors disabled:opacity-50"
                >
                  {isDeleting === doc.document_id ? (
                    <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-red-600"></div>
                  ) : (
                    '🗑️ Delete'
                  )}
                </button>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

export default DocumentManager;