
import { useState } from "react";
import { Search, Loader2, Sparkles, Trash2, Send } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import { useToast } from "@/hooks/use-toast";
import { ChatMessage } from "@/components/ChatMessage";

interface Collection {
  id: string;
  name: string;
  file_count: number;
  created_date: string;
  total_size_mb: number;
}

interface ChatEntry {
  id: string;
  type: 'user' | 'assistant';
  content: string;
  timestamp: string;
  sources?: any[];
  collectionId?: string;
  collectionName?: string;
}

interface ChatInterfaceProps {
  selectedCollection: Collection;
  chatHistory: ChatEntry[];
  setChatHistory: React.Dispatch<React.SetStateAction<ChatEntry[]>>;
  onSourceClick: (source: any) => void;
  onClearHistory: () => void;
}

export function ChatInterface({
  selectedCollection,
  chatHistory,
  setChatHistory,
  onSourceClick,
  onClearHistory
}: ChatInterfaceProps) {
  const [query, setQuery] = useState("");
  const [isQuerying, setIsQuerying] = useState(false);
  const { toast } = useToast();

  const handleQuery = async () => {
    if (!query.trim()) {
      toast({
        title: "Error",
        description: "Please enter a query",
        variant: "destructive",
      });
      return;
    }

    setIsQuerying(true);
    
    const userEntry: ChatEntry = {
      id: `user-${Date.now()}`,
      type: 'user',
      content: query,
      timestamp: new Date().toISOString()
    };
    
    setChatHistory(prev => [...prev, userEntry]);
    
    try {
      const response = await fetch("http://localhost:8000/collections/query", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          query: query,
          collection_id: selectedCollection.id,
          debug: false,
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      
      if (data && typeof data === 'object') {
        const assistantEntry: ChatEntry = {
          id: `assistant-${Date.now()}`,
          type: 'assistant',
          content: data.answer || "No answer available.",
          timestamp: new Date().toISOString(),
          sources: data.sources || [],
          collectionId: data.collection_id,
          collectionName: data.collection_name
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
      
      const errorEntry: ChatEntry = {
        id: `error-${Date.now()}`,
        type: 'assistant',
        content: "I'm sorry, I encountered an error processing your query. Please try again.",
        timestamp: new Date().toISOString()
      };
      
      setChatHistory(prev => [...prev, errorEntry]);
    } finally {
      setIsQuerying(false);
      setQuery("");
    }
  };

  return (
    <div className="flex flex-col h-full overflow-hidden">
      {/* Header - Fixed */}
      <div className="flex-shrink-0 p-4 border-b border-border bg-background">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-lg font-semibold">Query Collection</h2>
            <p className="text-sm text-muted-foreground">
              Querying: <Badge variant="outline" className="ml-1">{selectedCollection.name}</Badge>
            </p>
          </div>
          {chatHistory.length > 0 && (
            <Button variant="outline" onClick={onClearHistory} size="sm">
              <Trash2 className="h-4 w-4 mr-2" />
              Clear
            </Button>
          )}
        </div>
      </div>

      {/* Chat Messages - Scrollable */}
      <div className="flex-1 overflow-hidden">
        <ScrollArea className="h-full">
          <div className="p-4">
            {chatHistory.length === 0 && !isQuerying && (
              <div className="flex flex-col items-center justify-center h-full text-center min-h-[400px]">
                <div className="p-8 rounded-full bg-primary/10 mb-6">
                  <Search className="h-12 w-12 text-primary" />
                </div>
                <h3 className="text-xl font-semibold mb-2">Start Querying</h3>
                <p className="text-muted-foreground max-w-md">
                  Ask questions about the documents in "{selectedCollection.name}". I'll search through all {selectedCollection.file_count} files to find relevant information.
                </p>
              </div>
            )}

            {chatHistory.length > 0 && (
              <div className="space-y-4">
                {chatHistory.map((entry) => (
                  <ChatMessage
                    key={entry.id}
                    type={entry.type}
                    content={entry.content}
                    timestamp={entry.timestamp}
                    sources={entry.sources}
                    onSourceClick={onSourceClick}
                  />
                ))}
              </div>
            )}

            {isQuerying && (
              <div className="flex flex-col items-center justify-center py-12 text-center">
                <div className="relative mb-6">
                  <div className="w-16 h-16 border-4 border-primary/20 border-t-primary rounded-full animate-spin"></div>
                  <Sparkles className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 h-6 w-6 text-primary animate-pulse" />
                </div>
                <h3 className="text-lg font-semibold mb-2">Processing Query</h3>
                <p className="text-muted-foreground">Searching through {selectedCollection.name}...</p>
              </div>
            )}
          </div>
        </ScrollArea>
      </div>

      {/* Query Input - Fixed */}
      <div className="flex-shrink-0 border-t border-border p-4 bg-background">
        <div className="flex gap-2">
          <Input
            placeholder={`Ask about "${selectedCollection.name}"...`}
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyPress={(e) => e.key === "Enter" && !isQuerying && handleQuery()}
            className="flex-1"
            disabled={isQuerying}
          />
          <Button onClick={handleQuery} disabled={isQuerying || !query.trim()}>
            {isQuerying ? (
              <Loader2 className="h-4 w-4 animate-spin" />
            ) : (
              <Send className="h-4 w-4" />
            )}
          </Button>
        </div>
      </div>
    </div>
  );
}
