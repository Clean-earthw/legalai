'use client'
import React, { useState, useCallback, useEffect } from 'react';
import { AuthenticatedApiService, Document, SearchResult } from '@/services/apiService';

interface DocumentSearchProps {
  documents: Document[];
  apiService: AuthenticatedApiService;
  onDocumentDelete: (documentId: string) => void;
  onRefresh: () => void;
  isLoading?: boolean;
}

const DocumentSearch: React.FC<DocumentSearchProps> = ({ 
  documents, 
  apiService,
  onDocumentDelete, 
  onRefresh,
  isLoading = false
}) => {
  const [isDeleting, setIsDeleting] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState<SearchResult[]>([]);
  const [isSearching, setIsSearching] = useState(false);
  const [searchError, setSearchError] = useState<string | null>(null);
  const [selectedDocument, setSelectedDocument] = useState<string | null>(null);
  const [documentTypes, setDocumentTypes] = useState<string[]>([]);
  const [selectedType, setSelectedType] = useState<string>('all');
  const [sortBy, setSortBy] = useState<'name' | 'date' | 'type' | 'size'>('date');
  const [sortOrder, setSortOrder] = useState<'asc' | 'desc'>('desc');

  // Extract unique document types
  useEffect(() => {
    const types = Array.from(new Set(documents.map(doc => doc.document_type)))
      .filter(type => type && type !== 'unknown')
      .sort();
    setDocumentTypes(types);
  }, [documents]);

  // Filter and sort documents
  const filteredAndSortedDocuments = useCallback(() => {
    let filtered = documents.filter(doc => 
      selectedType === 'all' || doc.document_type === selectedType
    );

    return filtered.sort((a, b) => {
      let aValue: any, bValue: any;
      
      switch (sortBy) {
        case 'name':
          aValue = a.document_name.toLowerCase();
          bValue = b.document_name.toLowerCase();
          break;
        case 'date':
          aValue = new Date(a.created_at || 0);
          bValue = new Date(b.created_at || 0);
          break;
        case 'type':
          aValue = a.document_type;
          bValue = b.document_type;
          break;
        case 'size':
          aValue = a.file_size || 0;
          bValue = b.file_size || 0;
          break;
        default:
          return 0;
      }

      if (aValue < bValue) return sortOrder === 'asc' ? -1 : 1;
      if (aValue > bValue) return sortOrder === 'asc' ? 1 : -1;
      return 0;
    });
  }, [documents, selectedType, sortBy, sortOrder]);

  const handleDelete = async (documentId: string, documentName: string) => {
    if (!confirm(`Are you sure you want to delete "${documentName}"? This action cannot be undone.`)) {
      return;
    }

    setIsDeleting(documentId);
    try {
      await apiService.deleteDocument(documentId);
      onDocumentDelete(documentId);
      // Clear search results if the deleted document was in them
      setSearchResults(current => current.filter(result => result.document_id !== documentId));
    } catch (error: any) {
      console.error('Delete error:', error);
      alert(`Failed to delete document: ${error.message}`);
    } finally {
      setIsDeleting(null);
    }
  };

  const handleSearch = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!searchQuery.trim()) {
      setSearchResults([]);
      return;
    }

    setIsSearching(true);
    setSearchError(null);
    try {
      const response = await apiService.searchDocuments(searchQuery, 20);
      setSearchResults(response.results);
    } catch (error: any) {
      console.error('Search error:', error);
      setSearchError(error.message || 'Search failed');
      setSearchResults([]);
    } finally {
      setIsSearching(false);
    }
  };

  const clearSearch = () => {
    setSearchQuery('');
    setSearchResults([]);
    setSearchError(null);
  };

  const toggleSort = (newSortBy: typeof sortBy) => {
    if (sortBy === newSortBy) {
      setSortOrder(sortOrder === 'asc' ? 'desc' : 'asc');
    } else {
      setSortBy(newSortBy);
      setSortOrder('desc');
    }
  };

  const formatFileSize = (bytes: number | undefined): string => {
    if (!bytes) return 'N/A';
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  const formatDate = (dateString: string | undefined): string => {
    if (!dateString) return 'N/A';
    return new Date(dateString).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  const getDocumentIcon = (type: string): string => {
    switch (type) {
      case 'pdf': return '📄';
      case 'docx': return '📝';
      case 'contract': return '📑';
      case 'agreement': return '🤝';
      case 'lawsuit': return '⚖️';
      default: return '📋';
    }
  };

  return (
    <div className="bg-white rounded-lg shadow-lg p-6 h-full flex flex-col">
      {/* Header */}
      <div className="flex justify-between items-center mb-6 flex-shrink-0">
        <div>
          <h3 className="text-xl font-semibold text-gray-800">Document Library</h3>
          <p className="text-sm text-gray-600 mt-1">
            {documents.length} document{documents.length !== 1 ? 's' : ''} stored
          </p>
        </div>
        <button
          onClick={onRefresh}
          disabled={isLoading}
          className="px-4 py-2 bg-blue-100 text-blue-700 rounded-md hover:bg-blue-200 disabled:opacity-50 transition-colors flex items-center"
        >
          {isLoading ? (
            <>
              <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-700 mr-2"></div>
              Refreshing...
            </>
          ) : (
            'Refresh Documents'
          )}
        </button>
      </div>

      {/* Search Section */}
      <div className="mb-6 flex-shrink-0">
        <form onSubmit={handleSearch} className="space-y-3">
          <div className="flex space-x-2">
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="flex-1 px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              placeholder="Search across all documents..."
              disabled={isSearching}
            />
            <button
              type="submit"
              disabled={isSearching || !searchQuery.trim()}
              className="px-6 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 disabled:opacity-50 transition-colors flex items-center"
            >
              {isSearching ? (
                <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white mr-2"></div>
              ) : (
                <span className="mr-2">🔍</span>
              )}
              Search
            </button>
            {searchQuery && (
              <button
                type="button"
                onClick={clearSearch}
                className="px-4 py-2 text-gray-600 hover:text-gray-800 transition-colors"
              >
                Clear
              </button>
            )}
          </div>
          
          {searchError && (
            <div className="bg-red-50 border border-red-200 rounded-lg p-3">
              <p className="text-red-700 text-sm">{searchError}</p>
            </div>
          )}
        </form>
      </div>

      {/* Search Results */}
      {searchResults.length > 0 && (
        <div className="mb-6 bg-blue-50 rounded-lg p-4 flex-shrink-0">
          <div className="flex justify-between items-center mb-3">
            <h4 className="font-semibold text-blue-800">
              Search Results ({searchResults.length})
            </h4>
            <button
              onClick={clearSearch}
              className="text-blue-600 hover:text-blue-800 text-sm"
            >
              Clear Results
            </button>
          </div>
          <div className="space-y-3 max-h-64 overflow-y-auto">
            {searchResults.map((result, index) => (
              <div 
                key={`${result.document_id}-${index}`} 
                className="bg-white rounded-lg p-4 border border-blue-200 hover:border-blue-300 transition-colors"
              >
                <div className="flex justify-between items-start mb-2">
                  <div className="flex items-center">
                    <span className="text-lg mr-2">
                      {getDocumentIcon(result.document_type)}
                    </span>
                    <h5 className="font-medium text-blue-800">
                      {result.document_name}
                    </h5>
                  </div>
                  {result.similarity_score && (
                    <span className="text-xs font-medium px-2 py-1 rounded-full bg-blue-100 text-blue-700">
                      {Math.round(result.similarity_score * 100)}% match
                    </span>
                  )}
                </div>
                
                {result.section && (
                  <p className="text-xs text-gray-600 mb-2">
                    <span className="font-medium">Section:</span> {result.section}
                  </p>
                )}
                
                {result.page_number && (
                  <p className="text-xs text-gray-600 mb-3">
                    <span className="font-medium">Page:</span> {result.page_number}
                  </p>
                )}
                
                <p className="text-sm text-gray-700 bg-gray-50 p-3 rounded border">
                  {result.text.length > 200 
                    ? result.text.substring(0, 200) + '...' 
                    : result.text}
                </p>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Filters and Sorting */}
      <div className="flex flex-wrap gap-4 mb-4 flex-shrink-0">
        {/* Document Type Filter */}
        <div className="flex items-center space-x-2">
          <label className="text-sm font-medium text-gray-700">Filter:</label>
          <select
            value={selectedType}
            onChange={(e) => setSelectedType(e.target.value)}
            className="px-3 py-1 border border-gray-300 rounded-md text-sm"
          >
            <option value="all">All Types</option>
            {documentTypes.map(type => (
              <option key={type} value={type}>
                {type.charAt(0).toUpperCase() + type.slice(1)}
              </option>
            ))}
          </select>
        </div>

        {/* Sort Options */}
        <div className="flex items-center space-x-2">
          <label className="text-sm font-medium text-gray-700">Sort by:</label>
          <select
            value={sortBy}
            onChange={(e) => setSortBy(e.target.value as any)}
            className="px-3 py-1 border border-gray-300 rounded-md text-sm"
          >
            <option value="date">Date</option>
            <option value="name">Name</option>
            <option value="type">Type</option>
            <option value="size">Size</option>
          </select>
          <button
            onClick={() => setSortOrder(sortOrder === 'asc' ? 'desc' : 'asc')}
            className="p-1 hover:bg-gray-100 rounded"
            title={sortOrder === 'asc' ? 'Ascending' : 'Descending'}
          >
            {sortOrder === 'asc' ? '↑' : '↓'}
          </button>
        </div>
      </div>

      {/* Document List */}
      <div className="flex-1 flex flex-col min-h-0">
        <h4 className="font-semibold mb-3 text-gray-700 flex-shrink-0">
          Documents ({filteredAndSortedDocuments().length})
        </h4>
        
        {filteredAndSortedDocuments().length === 0 ? (
          <div className="text-center text-gray-500 py-12 flex-1 flex flex-col justify-center items-center">
            <div className="text-5xl mb-4">📚</div>
            <p className="text-lg mb-2">No documents found</p>
            <p className="text-sm">
              {selectedType !== 'all' 
                ? `No ${selectedType} documents available.` 
                : 'Upload your first document to get started!'}
            </p>
          </div>
        ) : (
          <div className="space-y-3 overflow-y-auto flex-1 pr-2">
            {filteredAndSortedDocuments().map((doc) => (
              <div 
                key={doc.document_id} 
                className="border border-gray-200 rounded-lg p-4 hover:shadow-md transition-shadow"
              >
                <div className="flex items-start justify-between">
                  <div className="flex items-start flex-1">
                    <div className="text-2xl mr-4 mt-1">
                      {getDocumentIcon(doc.document_type)}
                    </div>
                    <div className="flex-1">
                      <div className="text-sm font-medium text-gray-900 mb-1">
                        {doc.document_name}
                      </div>
                      <div className="flex flex-wrap gap-2 text-xs text-gray-600 mb-2">
                        <span className="px-2 py-1 bg-blue-100 text-blue-700 rounded-full">
                          {doc.document_type.toUpperCase()}
                        </span>
                        <span>{doc.chunks_count} chunks</span>
                        <span>•</span>
                        <span>{formatFileSize(doc.file_size)}</span>
                      </div>
                      <div className="text-xs text-gray-500">
                        Uploaded: {formatDate(doc.created_at)}
                      </div>
                    </div>
                  </div>
                  <button
                    onClick={() => handleDelete(doc.document_id, doc.document_name)}
                    disabled={isDeleting === doc.document_id}
                    className="text-red-600 hover:text-red-800 transition-colors text-sm ml-4 disabled:opacity-50 flex items-center"
                    title="Delete document"
                  >
                    {isDeleting === doc.document_id ? (
                      <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-red-600"></div>
                    ) : (
                      <>
                        <span className="mr-1">🗑️</span>
                        Delete
                      </>
                    )}
                  </button>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

export default DocumentSearch;