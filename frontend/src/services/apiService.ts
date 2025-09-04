// API Configuration
const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL || 'http://localhost:8000';

export interface UploadResult {
  document_id: string;
  document_name: string;
  chunks_added: number;
  status: string;
  user_id: string;
  session_id?: string;
}

export interface QueryResult {
  answer: string;
  sources: string[];
  confidence?: number;
  processing_time: number;
  user_id: string;
  session_id?: string;
}

export interface SearchResult {
  text: string;
  document_name: string;
  document_type: string;
  document_id: string;
  page_number?: number;
  section?: string;
  similarity_score?: number;
  user_id: string;
  session_id?: string;
}

export interface SearchResponse {
  query: string;
  results_count: number;
  results: SearchResult[];
  user_id: string;
  session_id?: string;
}

export interface Document {
  document_id: string;
  document_name: string;
  document_type: string;
  chunks_count: number;
  user_id: string;
  session_id?: string;
  created_at?: string;
  file_size?: number;
}

export interface UserStats {
  user_id: string;
  total_documents: number;
  total_chunks: number;
  sessions: string[];
  document_types: { [key: string]: number };
  total_file_size: number;
}

export interface SessionResponse {
  session_id: string;
  user_id: string;
  status: string;
}

export interface DeleteResponse {
  status: string;
  chunks_deleted: number;
  user_id: string;
  document_id?: string;
  session_id?: string;
}

export interface HealthResponse {
  status: string;
  components: {
    [key: string]: string;
  };
  details?: any;
}

export interface ApiHeaders {
  [key: string]: string;
}

// API Service Class with Authentication Support
export class AuthenticatedApiService {
  private userId?: string;
  private sessionId?: string;
  private getToken?: () => Promise<string | null>;

  constructor(userId?: string, sessionId?: string, getToken?: () => Promise<string | null>) {
    this.userId = userId;
    this.sessionId = sessionId;
    this.getToken = getToken;
  }

  // Update authentication context
  setAuth(userId: string, sessionId?: string, getToken?: () => Promise<string | null>) {
    this.userId = userId;
    this.sessionId = sessionId;
    this.getToken = getToken;
  }

  // Build headers with authentication
  private async buildHeaders(contentType: string = 'application/json'): Promise<ApiHeaders> {
    if (!this.userId) {
      throw new Error('User ID is required for API calls');
    }

    const headers: ApiHeaders = {
      'X-User-Id': this.userId,
    };

    if (this.sessionId) {
      headers['X-Session-Id'] = this.sessionId;
    }

    if (contentType) {
      headers['Content-Type'] = contentType;
    }

    // Add Clerk token if available
    if (this.getToken) {
      try {
        const token = await this.getToken();
        if (token) {
          headers['Authorization'] = `Bearer ${token}`;
        }
      } catch (error) {
        console.warn('Failed to get authentication token:', error);
      }
    }

    return headers;
  }

  // Make authenticated API request
  private async makeRequest<T>(endpoint: string, options: RequestInit = {}): Promise<T> {
    const url = `${API_BASE_URL}${endpoint}`;
    
    try {
      const response = await fetch(url, {
        ...options,
        headers: {
          ...(options.headers || {}),
          ...(options.body instanceof FormData ? {} : await this.buildHeaders()),
        },
      });

      if (!response.ok) {
        let errorMessage = 'API request failed';
        try {
          const error = await response.json();
          errorMessage = error.detail || error.message || errorMessage;
        } catch {
          errorMessage = `HTTP ${response.status}: ${response.statusText}`;
        }
        throw new Error(errorMessage);
      }

      return response.json();
    } catch (error) {
      if (error instanceof Error) {
        throw error;
      }
      throw new Error('Network error occurred');
    }
  }

  // Create new session
  async createSession(sessionName?: string): Promise<SessionResponse> {
    return this.makeRequest<SessionResponse>('/sessions', {
      method: 'POST',
      body: JSON.stringify({ session_name: sessionName }),
    });
  }

  // Upload document file
  async uploadDocument(
    file: File, 
    documentName?: string, 
    documentType?: string
  ): Promise<UploadResult> {
    if (!this.userId) {
      throw new Error('User ID is required for file upload');
    }

    const formData = new FormData();
    formData.append('file', file);
    
    if (documentName) {
      formData.append('document_name', documentName);
    }
    
    if (documentType) {
      formData.append('document_type', documentType);
    }

    // For FormData, we need to build headers manually without Content-Type
    const headers: ApiHeaders = {
      'X-User-Id': this.userId,
    };

    if (this.sessionId) {
      headers['X-Session-Id'] = this.sessionId;
    }

    if (this.getToken) {
      try {
        const token = await this.getToken();
        if (token) {
          headers['Authorization'] = `Bearer ${token}`;
        }
      } catch (error) {
        console.warn('Failed to get authentication token:', error);
      }
    }

    const response = await fetch(`${API_BASE_URL}/upload`, {
      method: 'POST',
      headers,
      body: formData,
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: 'Upload failed' }));
      throw new Error(error.detail || 'Upload failed');
    }

    return response.json();
  }

  // Upload text content directly
  async uploadTextContent(
    text: string, 
    documentName: string, 
    documentType: string = 'text'
  ): Promise<UploadResult> {
    return this.makeRequest<UploadResult>('/upload/text', {
      method: 'POST',
      body: JSON.stringify({
        text,
        document_name: documentName,
        document_type: documentType
      }),
    });
  }

  // Ask question about documents
  async queryLegalSupport(question: string, documentType?: string): Promise<QueryResult> {
    return this.makeRequest<QueryResult>('/ask', {
      method: 'POST',
      body: JSON.stringify({ 
        question,
        document_type: documentType 
      }),
    });
  }

  // Search documents
  async searchDocuments(
    query: string, 
    limit: number = 5, 
    documentType?: string
  ): Promise<SearchResponse> {
    return this.makeRequest<SearchResponse>('/search', {
      method: 'POST',
      body: JSON.stringify({ 
        query, 
        limit,
        document_type: documentType 
      }),
    });
  }

  // Get user documents
  async getDocuments(): Promise<Document[]> {
    return this.makeRequest<Document[]>('/documents');
  }

  // Get user statistics
  async getUserStats(): Promise<UserStats> {
    return this.makeRequest<UserStats>('/stats');
  }

  // Delete document
  async deleteDocument(documentId: string): Promise<DeleteResponse> {
    return this.makeRequest<DeleteResponse>(`/documents/${documentId}`, {
      method: 'DELETE',
    });
  }

  // Delete session documents
  async deleteSessionDocuments(sessionId: string): Promise<DeleteResponse> {
    return this.makeRequest<DeleteResponse>(`/sessions/${sessionId}/documents`, {
      method: 'DELETE',
    });
  }

  // Check system health (no auth required)
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
  }
}

// Hook for using authenticated API service with Clerk
export function useApiService(userId?: string, sessionId?: string, getToken?: () => Promise<string | null>) {
  const apiService = new AuthenticatedApiService(userId, sessionId, getToken);
  
  return {
    apiService,
    // Helper methods
    setAuth: (newUserId: string, newSessionId?: string, newGetToken?: () => Promise<string | null>) => {
      apiService.setAuth(newUserId, newSessionId, newGetToken);
    },
  };
}

// Legacy API service for backward compatibility (without authentication)
export const apiService = {
  async uploadDocument(file: File, documentName: string = '', documentType: string = ''): Promise<UploadResult> {
    console.warn('Using legacy API service without authentication. Consider using AuthenticatedApiService.');
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
    console.warn('Using legacy API service without authentication. Consider using AuthenticatedApiService.');
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
    console.warn('Using legacy API service without authentication. Consider using AuthenticatedApiService.');
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

  async getDocuments(): Promise<Document[]> {  
    console.warn('Using legacy API service without authentication. Consider using AuthenticatedApiService.');
    const response = await fetch(`${API_BASE_URL}/documents`);
     
    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Failed to fetch documents');
    }
    return response.json();
  },

  async deleteDocument(documentId: string): Promise<DeleteResponse> {
    console.warn('Using legacy API service without authentication. Consider using AuthenticatedApiService.');
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
    console.warn('Using legacy API service without authentication. Consider using AuthenticatedApiService.');
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

// Export default authenticated service
export default AuthenticatedApiService;