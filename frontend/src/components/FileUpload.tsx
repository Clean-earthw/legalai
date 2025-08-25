'use client'
import React, { useState, useRef, useCallback } from 'react';
import { apiService, UploadResult } from '@/lib/service';

interface FileUploadProps {
  onUploadSuccess: (result: UploadResult) => void;
  isUploading: boolean;
  setIsUploading: (uploading: boolean) => void;
}

const FileUpload: React.FC<FileUploadProps> = ({ 
  onUploadSuccess, 
  isUploading, 
  setIsUploading 
}) => {
  const [dragOver, setDragOver] = useState(false);
  const [documentName, setDocumentName] = useState('');
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
  }, []);

  const handleDrop = useCallback(async (e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    
    const files = Array.from(e.dataTransfer.files);
    if (files.length > 0) {
      await uploadFile(files[0]);
    }
  }, [documentName]);

  const handleFileSelect = useCallback(async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files && files.length > 0) {
      await uploadFile(files[0]);
    }
  }, [documentName]);

  const uploadFile = async (file: File) => {
    // Validate file type
    const allowedTypes = ['.pdf', '.docx'];
    const fileExtension = '.' + file.name.split('.').pop()?.toLowerCase();
    
    if (!allowedTypes.includes(fileExtension)) {
      alert(`Please select a valid file type: ${allowedTypes.join(', ')}`);
      return;
    }

    // Validate file size (50MB limit)
    const maxSize = 50 * 1024 * 1024; // 50MB
    if (file.size > maxSize) {
      alert('File size must be less than 50MB');
      return;
    }

    setIsUploading(true);
    try {
      const result = await apiService.uploadDocument(file, documentName);
      onUploadSuccess(result);
      setDocumentName('');
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
    } catch (error: any) {
      console.error('Upload error:', error);
      alert(`Upload failed: ${error.message}`);
    } finally {
      setIsUploading(false);
    }
  };

  return (
    <div className="bg-white rounded-lg shadow-lg p-6">
      <h3 className="text-xl font-semibold mb-4 text-gray-800">Upload Legal Document</h3>
      
      <div className="mb-4">
        <label className="block text-sm font-medium text-gray-700 mb-2">
          Document Name (Optional)
        </label>
        <input
          type="text"
          value={documentName}
          onChange={(e) => setDocumentName(e.target.value)}
          className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
          placeholder="Enter document name..."
          disabled={isUploading}
        />
      </div>

      <div
        className={`border-2 border-dashed ${dragOver ? 'border-blue-500 bg-blue-50' : 'border-gray-300'} 
          rounded-lg p-8 text-center cursor-pointer transition-all duration-300 
          ${isUploading ? 'opacity-50 cursor-not-allowed' : 'hover:border-blue-400'}`}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={() => !isUploading && fileInputRef.current?.click()}
      >
        <input
          ref={fileInputRef}
          type="file"
          className="hidden"
          accept=".pdf,.docx"
          onChange={handleFileSelect}
          disabled={isUploading}
        />
        
        {isUploading ? (
          <div className="flex flex-col items-center">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mb-4"></div>
            <p className="text-gray-600">Uploading and processing...</p>
          </div>
        ) : (
          <>
            <div className="text-4xl mb-4">📄</div>
            <p className="text-lg font-medium text-gray-700 mb-2">
              Drop your document here or click to browse
            </p>
            <p className="text-sm text-gray-500">
              Supports PDF and DOCX files (max 50MB)
            </p>
          </>
        )}
      </div>
    </div>
  );
};

export default FileUpload;