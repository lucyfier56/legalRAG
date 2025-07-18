import { useState, useEffect, useRef } from "react";
import { Search, FileText, Download, ExternalLink, Loader2, Sparkles, MessageSquare, Trash2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import { useToast } from "@/hooks/use-toast";
import { PDFViewer } from "@/components/PDFViewer";
import { SidebarProvider } from "@/components/ui/sidebar";
import { AppSidebar } from "@/components/AppSidebar";
import { Toaster } from "@/components/ui/toaster";
import { ChatHistory } from "@/components/ChatHistory";
import { ChatMessage } from "@/components/ChatMessage";

interface QueryResponse {
  query: string;
  answer: string;
  method: string;
  query_type: string;
  intent: string;
  entities: any;
  results_count: number;
  sources: Array<{
    document: string;
    pages: number[];
    chunks: number;
    max_score: number;
    file_path: string;
    downloadable: boolean;
  }>;
  total_documents?: number;
  total_mentions?: number;
  entity_name?: string;
  confidence_metrics?: any;
}

interface ChatEntry {
  id: string;
  type: 'user' | 'assistant';
  content: string;
  timestamp: string;
  sources?: any[];
  queryType?: string;
  intent?: string;
}

export function RAGSystemPage() {
  const [query, setQuery] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [chatHistory, setChatHistory] = useState<ChatEntry[]>([]);
  const [selectedPDF, setSelectedPDF] = useState<string | null>(null);
  const [showHistory, setShowHistory] = useState(false);
  const { toast } = useToast();
  const scrollAreaRef = useRef<HTMLDivElement>(null);

  // Load chat history from localStorage on component mount
  useEffect(() => {
    const savedHistory = localStorage.getItem('rag-chat-history');
    if (savedHistory) {
      try {
        setChatHistory(JSON.parse(savedHistory));
      } catch (error) {
        console.error('Error loading chat history:', error);
      }
    }
  }, []);

  // Save chat history to localStorage whenever it changes
  useEffect(() => {
    localStorage.setItem('rag-chat-history', JSON.stringify(chatHistory));
  }, [chatHistory]);

  // Auto-scroll to bottom when new messages are added
  useEffect(() => {
    if (scrollAreaRef.current) {
      const scrollContainer = scrollAreaRef.current.querySelector('[data-radix-scroll-area-viewport]');
      if (scrollContainer) {
        scrollContainer.scrollTop = scrollContainer.scrollHeight;
      }
    }
  }, [chatHistory]);

  const handleQuery = async () => {
    if (!query.trim()) {
      toast({
        title: "Error",
        description: "Please enter a query",
        variant: "destructive",
      });
      return;
    }

    setIsLoading(true);
    
    // Add user message to chat history
    const userEntry: ChatEntry = {
      id: `user-${Date.now()}`,
      type: 'user',
      content: query,
      timestamp: new Date().toISOString()
    };
    
    setChatHistory(prev => [...prev, userEntry]);
    
    try {
      const response = await fetch("http://localhost:8000/rag/query-detailed", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          query: query,
          debug: false,
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      console.log("Received response:", data);
      
      if (data && typeof data === 'object') {
        // Add assistant response to chat history
        const assistantEntry: ChatEntry = {
          id: `assistant-${Date.now()}`,
          type: 'assistant',
          content: data.answer || "No answer available.",
          timestamp: new Date().toISOString(),
          sources: data.sources || [],
          queryType: data.query_type,
          intent: data.intent
        };
        
        setChatHistory(prev => [...prev, assistantEntry]);
        
        toast({
          title: "Query Processed",
          description: `Found ${data.results_count || 0} relevant results`,
        });
      } else {
        throw new Error("Invalid response format");
      }
    } catch (error) {
      console.error("Query failed:", error);
      toast({
        title: "Query Failed",
        description: "Failed to process your query. Please try again.",
        variant: "destructive",
      });
      
      // Add error message to chat history
      const errorEntry: ChatEntry = {
        id: `error-${Date.now()}`,
        type: 'assistant',
        content: "I'm sorry, I encountered an error processing your query. Please try again.",
        timestamp: new Date().toISOString()
      };
      
      setChatHistory(prev => [...prev, errorEntry]);
    } finally {
      setIsLoading(false);
      setQuery(""); // Clear the input after sending
    }
  };

  const handleSourceClick = (source: any) => {
    if (source.downloadable && source.file_path) {
      const filename = source.document + ".pdf";
      setSelectedPDF(filename);
    } else {
      toast({
        title: "PDF Not Available",
        description: "This document is not available for viewing",
        variant: "destructive",
      });
    }
  };

  const clearChatHistory = () => {
    setChatHistory([]);
    toast({
      title: "History Cleared",
      description: "Chat history has been cleared",
    });
  };

  const toggleHistory = () => {
    setShowHistory(!showHistory);
  };

  return (
    <SidebarProvider>
      <div className="min-h-screen flex w-full bg-background">
        <AppSidebar />
        <main className="flex-1 overflow-hidden">
          <div className="flex h-screen">
            {/* Main Content Area */}
            <div className={`${selectedPDF && !showHistory ? 'w-1/2' : showHistory && !selectedPDF ? 'w-1/2' : selectedPDF && showHistory ? 'w-1/3' : 'w-full'} flex flex-col transition-all duration-300`}>
              
              {/* Header with History Toggle */}
              <div className="p-4 border-b border-border flex items-center justify-between">
                <h1 className="text-2xl font-bold">RAG System</h1>
                <div className="flex gap-2">
                  <Button variant="outline" onClick={toggleHistory}>
                    <MessageSquare className="h-4 w-4 mr-2" />
                    {showHistory ? 'Hide History' : 'Show History'}
                  </Button>
                  {chatHistory.length > 0 && (
                    <Button variant="outline" onClick={clearChatHistory}>
                      <Trash2 className="h-4 w-4 mr-2" />
                      Clear All
                    </Button>
                  )}
                </div>
              </div>

              {/* Conversation Area - Show All History */}
              <ScrollArea className="flex-1 p-6" ref={scrollAreaRef}>
                {/* Empty State - Only show when no history */}
                {chatHistory.length === 0 && !isLoading && (
                  <div className="flex flex-col items-center justify-center h-full text-center max-w-4xl mx-auto px-8">
                    <div className="p-6 rounded-full bg-primary/10 mb-6">
                      <Search className="h-16 w-16 text-primary" />
                    </div>
                    <h2 className="text-3xl font-bold mb-4">RAG System</h2>
                    <p className="text-lg text-muted-foreground mb-8">
                      Query legal documents with AI-powered search. Enter your legal question below to get started.
                    </p>
                  </div>
                )}

                {/* Show All Chat History */}
                {chatHistory.length > 0 && (
                  <div className="space-y-4 mb-4">
                    {chatHistory.map((entry) => (
                      <ChatMessage
                        key={entry.id}
                        type={entry.type}
                        content={entry.content}
                        timestamp={entry.timestamp}
                        sources={entry.sources}
                        queryType={entry.queryType}
                        intent={entry.intent}
                        onSourceClick={handleSourceClick}
                      />
                    ))}
                  </div>
                )}

                {/* Loading State */}
                {isLoading && (
                  <div className="flex flex-col items-center justify-center py-12 text-center">
                    <div className="relative">
                      <div className="w-20 h-20 border-4 border-primary/20 border-t-primary rounded-full animate-spin mb-6"></div>
                      <Sparkles className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 h-8 w-8 text-primary animate-pulse" />
                    </div>
                    <h3 className="text-xl font-semibold mb-2">Processing your query</h3>
                    <p className="text-muted-foreground">AI is analyzing legal documents...</p>
                  </div>
                )}
              </ScrollArea>

              {/* Search Input - Always at bottom */}
              <div className="border-t border-border p-6 bg-background/95 backdrop-blur-sm">
                <div className="flex gap-2 w-full">
                  <Input
                    placeholder="Enter your legal query..."
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                    onKeyPress={(e) => e.key === "Enter" && !isLoading && handleQuery()}
                    className="flex-1 h-12"
                    disabled={isLoading}
                  />
                  <Button onClick={handleQuery} disabled={isLoading} size="lg">
                    {isLoading ? (
                      <div className="flex items-center gap-2">
                        <Loader2 className="h-4 w-4 animate-spin" />
                        <span>Searching...</span>
                      </div>
                    ) : (
                      <Search className="h-5 w-5" />
                    )}
                  </Button>
                </div>
              </div>
            </div>

            {/* Chat History Panel */}
            {showHistory && (
              <div className={`${selectedPDF ? 'w-1/3' : 'w-1/2'} border-l border-border transition-all duration-300`}>
                <ChatHistory
                  chatHistory={chatHistory}
                  onSourceClick={handleSourceClick}
                  onClearHistory={clearChatHistory}
                />
              </div>
            )}

            {/* PDF Viewer Panel */}
            {selectedPDF && (
              <div className={`${showHistory ? 'w-1/3' : 'w-1/2'} flex flex-col border-l border-border transition-all duration-300`}>
                <div className="p-4 border-b border-border flex items-center justify-between bg-background/95 backdrop-blur-sm">
                  <h2 className="text-lg font-semibold">
                    Viewing: {selectedPDF}
                  </h2>
                  <Button variant="outline" size="sm" onClick={() => setSelectedPDF(null)}>
                    Close PDF
                  </Button>
                </div>
                <div className="flex-1">
                  <PDFViewer filename={selectedPDF} />
                </div>
              </div>
            )}
          </div>
        </main>
        <Toaster />
      </div>
    </SidebarProvider>
  );
}
