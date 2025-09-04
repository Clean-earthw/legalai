'use client'
import React, { useState, useEffect, useRef } from 'react';
import { AuthenticatedApiService } from '@/services/apiService';

interface Message {
  type: 'user' | 'ai';
  content: string;
  timestamp: Date;
  processingTime?: number;
  sources?: string[];
}

interface ChatInterfaceProps {
  apiService: AuthenticatedApiService;
}

const ChatInterface: React.FC<ChatInterfaceProps> = ({ apiService }) => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(scrollToBottom, [messages]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!inputValue.trim() || isLoading) return;

    const userMessage: Message = { 
      type: 'user', 
      content: inputValue, 
      timestamp: new Date() 
    };
    setMessages(prev => [...prev, userMessage]);
    setInputValue('');
    setIsLoading(true);

    try {
      const response = await apiService.queryLegalSupport(inputValue);
      const aiMessage: Message = { 
        type: 'ai', 
        content: response.answer,
        timestamp: new Date(),
        processingTime: response.processing_time,
        sources: response.sources
      };
      setMessages(prev => [...prev, aiMessage]);
    } catch (error: any) {
      console.error('Query error:', error);
      const errorMessage: Message = { 
        type: 'ai', 
        content: `I apologize, but I encountered an error: ${error.message}. Please try again.`, 
        timestamp: new Date() 
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="flex flex-col h-full">
      {/* Messages Area */}
      <div className="flex-1 overflow-y-auto mb-6 space-y-4 pr-2">
        {messages.length === 0 ? (
          <div className="text-center text-gray-500 mt-16">
            <div className="text-6xl mb-6">⚖️</div>
            <p className="text-xl font-medium mb-2">Welcome to your Legal Counsel</p>
            <p className="text-lg">Ask me anything about your legal documents!</p>
            <p className="text-sm mt-4 text-gray-400">
              I specialize in employment law, compliance, contract analysis, and equity management.
            </p>
          </div>
        ) : (
          messages.map((message, index) => (
            <div
              key={index}
              className={`flex ${message.type === 'user' ? 'justify-end' : 'justify-start'}`}
            >
              <div
                className={`max-w-2xl px-6 py-4 rounded-lg shadow-sm ${
                  message.type === 'user' 
                    ? 'bg-gradient-to-r from-blue-600 to-purple-700 text-white' 
                    : 'bg-gray-50 text-gray-800 border border-gray-200'
                }`}
              >
                <p className="whitespace-pre-wrap leading-relaxed text-base">{message.content}</p>
                
                {/* Show sources if available */}
                {message.sources && message.sources.length > 0 && (
                  <div className="mt-4 pt-3 border-t border-gray-300">
                    <p className="text-sm font-semibold opacity-80 mb-2">📚 Referenced Sources:</p>
                    <div className="text-sm opacity-70 space-y-1">
                      {message.sources.slice(0, 3).map((source, idx) => (
                        <div key={idx} className="flex items-center">
                          <span className="mr-2">📄</span>
                          <span className="truncate">{source}</span>
                        </div>
                      ))}
                      {message.sources.length > 3 && (
                        <div className="text-sm italic">
                          +{message.sources.length - 3} additional sources referenced
                        </div>
                      )}
                    </div>
                  </div>
                )}
                
                {message.processingTime && (
                  <p className="text-xs mt-3 opacity-60">
                    ⏱️ Processed in {message.processingTime.toFixed(2)}s
                  </p>
                )}
              </div>
            </div>
          ))
        )}
        
        {isLoading && (
          <div className="flex justify-start">
            <div className="bg-gray-100 border border-gray-200 max-w-2xl px-6 py-4 rounded-lg shadow-sm">
              <div className="flex items-center space-x-3">
                <div className="flex space-x-1">
                  <div className="w-3 h-3 bg-gray-500 rounded-full animate-pulse"></div>
                  <div className="w-3 h-3 bg-gray-500 rounded-full animate-pulse" style={{animationDelay: '0.2s'}}></div>
                  <div className="w-3 h-3 bg-gray-500 rounded-full animate-pulse" style={{animationDelay: '0.4s'}}></div>
                </div>
                <span className="text-base text-gray-600">Legal analysis in progress...</span>
              </div>
            </div>
          </div>
        )}
        
        <div ref={messagesEndRef} />
      </div>

      {/* Input Area */}
      <div className="border-t border-gray-200 pt-4">
        <form onSubmit={handleSubmit} className="flex space-x-4">
          <input
            type="text"
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            className="flex-1 px-4 py-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent text-base"
            placeholder="Ask your legal question here..."
            disabled={isLoading}
            maxLength={2000}
          />
          <button
            type="submit"
            disabled={isLoading || !inputValue.trim()}
            className="px-6 py-3 bg-gradient-to-r from-blue-600 to-purple-700 text-white font-medium rounded-lg hover:from-blue-700 hover:to-purple-800 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200 shadow-md hover:shadow-lg"
          >
            {isLoading ? (
              <div className="flex items-center space-x-2">
                <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white"></div>
                <span>Analyzing</span>
              </div>
            ) : (
              <span>Submit Query</span>
            )}
          </button>
        </form>
        <p className="text-xs text-gray-500 mt-2">
          Press Enter to submit • {inputValue.length}/2000 characters
        </p>
      </div>
    </div>
  );
};

export default ChatInterface;