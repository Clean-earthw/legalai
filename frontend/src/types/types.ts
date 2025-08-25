export interface UploadResult {
  success: boolean;
  filename: string;
  document_name: string;
  document_id: string;
  file_type: string;
  chunks_added: number;
  text_preview: string;
  message: string;
}

export interface QueryResult {
  result: string;
  processing_time?: number;
}

export interface SearchResult {
  text: string;
  document_id: string;
  document_name: string;
  section?: string;
  relevance_score: number;
}

export interface SearchResponse {
  success: boolean;
  query: string;
  results_count: number;
  results: SearchResult[];
}

export interface Document {
  document_id: string;
  document_name: string;
  chunks_count: number;
  upload_date?: string;
}

export interface DocumentListResponse {
  success: boolean;
  document_count: number;
  documents: Document[];
}

export interface HealthResponse {
  status: string;
  timestamp: number;
  version: string;
  services: {
    api: string;
    document_store: string;
  };
  document_count?: number;
}

export interface Message {
  type: 'user' | 'ai';
  content: string;
  timestamp: Date;
  processingTime?: number;
}