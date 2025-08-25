// API Configuration
const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL || 'http://localhost:8000';

export interface UploadResult {
  document_id: string;
  document_name: string;
  chunks_added: number;
  status: string;
}

export interface QueryResult {
  answer: string;
  sources: string[];
  confidence?: number;
  processing_time: number;
}

export interface SearchResult {
  text: string;
  document_name: string;
  document_type: string;
  page_number?: number;
  section?: string;
  similarity_score?: number;
}

export interface SearchResponse {
  query: string;
  results_count: number;
  results: SearchResult[];
}

export interface Document {
  document_id: string;
  document_name: string;
  document_type: string;
  chunks_count: number;
}

export interface DocumentListResponse {
  success: boolean;
  documents: Document[];
}

export interface HealthResponse {
  status: string;
  components: {
    [key: string]: string;
  };
  details?: any;
}

// API Service
export const apiService = {
  async uploadDocument(file: File, documentName: string = '', documentType: string = ''): Promise<UploadResult> {
    const formData = new FormData();
    formData.append('file', file);
    
    if (documentName) {
      formData.append('document_name', documentName);
    }
    
    if (documentType) {
      formData.append('document_type', documentType);
    }

    const response = await fetch(`${API_BASE_URL}/upload`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Upload failed');
    }

    return response.json();
  },

  async queryLegalSupport(question: string, documentType?: string): Promise<QueryResult> {
    const response = await fetch(`${API_BASE_URL}/ask`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ 
        question,
        document_type: documentType 
      }),
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Query failed');
    }

    return response.json();
  },

  async searchDocuments(query: string, limit: number = 5, documentType?: string): Promise<SearchResponse> {
    const response = await fetch(`${API_BASE_URL}/search`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ 
        query, 
        limit,
        document_type: documentType 
      }),
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Search failed');
    }

    return response.json();
  },

  async getDocuments(): Promise<DocumentListResponse> {  // Fixed return type
    const response = await fetch(`${API_BASE_URL}/documents`);

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Failed to fetch documents');
    }

    return response.json();  // Should return { success: boolean, documents: Document[] }
  },

  async deleteDocument(documentId: string): Promise<{ document_id: string; chunks_deleted: number; status: string }> {
    const response = await fetch(`${API_BASE_URL}/documents/${documentId}`, {
      method: 'DELETE',
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Delete failed');
    }

    return response.json();
  },

  async checkHealth(): Promise<HealthResponse> {
    try {
      const response = await fetch(`${API_BASE_URL}/health`);
      
      if (!response.ok) {
        return {
          status: 'unhealthy',
          components: {
            api: 'disconnected',
            document_store: 'unknown'
          }
        };
      }

      return response.json();
    } catch (error) {
      return {
        status: 'unhealthy',
        components: {
          api: 'unreachable',
          document_store: 'unknown'
        },
        details: {
          error: error instanceof Error ? error.message : 'Unknown error'
        }
      };
    }
  },

  async uploadTextContent(text: string, documentName: string, documentType: string = 'text'): Promise<UploadResult> {
    const response = await fetch(`${API_BASE_URL}/upload/text`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        text,
        document_name: documentName,
        document_type: documentType
      }),
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Text upload failed');
    }

    return response.json();
  }
};