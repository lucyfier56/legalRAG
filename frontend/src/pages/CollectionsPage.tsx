
import { useState, useRef, useEffect } from "react";
import { Upload, Search, FileText, Folder, Loader2, Sparkles, Trash2 } from "lucide-react";
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
import { CollectionsList } from "@/components/CollectionsList";
import { CollectionUpload } from "@/components/CollectionUpload";
import { ChatInterface } from "@/components/ChatInterface";

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

export function CollectionsPage() {
  const [collections, setCollections] = useState<Collection[]>([]);
  const [selectedCollection, setSelectedCollection] = useState<Collection | null>(null);
  const [isLoadingCollections, setIsLoadingCollections] = useState(false);
  const [chatHistory, setChatHistory] = useState<ChatEntry[]>([]);
  const [selectedPDF, setSelectedPDF] = useState<string | null>(null);
  const { toast } = useToast();

  const loadCollections = async () => {
    setIsLoadingCollections(true);
    try {
      const response = await fetch("http://localhost:8000/collections/list");
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const data = await response.json();
      if (data.success) {
        setCollections(data.collections);
      } else {
        throw new Error(data.error || "Failed to load collections");
      }
    } catch (error) {
      console.error("Failed to load collections:", error);
      toast({
        title: "Loading Failed",
        description: "Failed to load collections. Please try again.",
        variant: "destructive",
      });
    } finally {
      setIsLoadingCollections(false);
    }
  };

  const deleteCollection = async (collection: Collection) => {
    try {
      const response = await fetch(`http://localhost:8000/collections/${collection.id}`, {
        method: "DELETE",
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      
      if (data.success) {
        toast({
          title: "Collection Deleted",
          description: `Successfully deleted collection "${collection.name}"`,
        });
        
        if (selectedCollection?.id === collection.id) {
          setSelectedCollection(null);
          setChatHistory([]);
        }
        
        await loadCollections();
      } else {
        throw new Error(data.error || "Collection deletion failed");
      }
    } catch (error) {
      console.error("Collection deletion failed:", error);
      toast({
        title: "Deletion Failed",
        description: "Failed to delete collection. Please try again.",
        variant: "destructive",
      });
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

  useEffect(() => {
    loadCollections();
  }, []);

  return (
    <SidebarProvider>
      <div className="min-h-screen flex w-full bg-background">
        <AppSidebar />
        <main className="flex-1 flex flex-col min-h-screen">
          <div className="flex-1 flex overflow-hidden">
            {/* Left Panel - Collections and Chat */}
            <div className={`${selectedPDF ? 'w-2/3' : 'w-full'} flex flex-col border-r border-border`}>
              
              {/* Header */}
              <div className="flex-shrink-0 p-6 border-b border-border bg-background">
                <h1 className="text-3xl font-bold flex items-center gap-3">
                  <Folder className="h-8 w-8 text-primary" />
                  Document Collections
                </h1>
                <p className="text-muted-foreground mt-2">
                  Upload folders of documents and query them independently
                </p>
              </div>

              <div className="flex-1 flex overflow-hidden">
                {/* Collections Sidebar */}
                <div className="w-80 flex flex-col border-r border-border bg-muted/20 overflow-hidden">
                  <CollectionsList
                    collections={collections}
                    selectedCollection={selectedCollection}
                    isLoading={isLoadingCollections}
                    onSelectCollection={setSelectedCollection}
                    onDeleteCollection={deleteCollection}
                    onRefresh={loadCollections}
                  />
                  
                  <CollectionUpload onUploadSuccess={loadCollections} />
                </div>

                {/* Chat Area */}
                <div className="flex-1 flex flex-col overflow-hidden">
                  {selectedCollection ? (
                    <ChatInterface
                      selectedCollection={selectedCollection}
                      chatHistory={chatHistory}
                      setChatHistory={setChatHistory}
                      onSourceClick={handleSourceClick}
                      onClearHistory={clearChatHistory}
                    />
                  ) : (
                    <div className="flex-1 flex items-center justify-center">
                      <div className="text-center max-w-md">
                        <div className="p-8 rounded-full bg-primary/10 mb-6 mx-auto w-fit">
                          <Folder className="h-16 w-16 text-primary" />
                        </div>
                        <h3 className="text-2xl font-semibold mb-3">Select a Collection</h3>
                        <p className="text-muted-foreground leading-relaxed">
                          Choose a collection from the sidebar to start querying your documents, or create a new collection by uploading PDF files.
                        </p>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </div>

            {/* Right Panel - PDF Viewer */}
            {selectedPDF && (
              <div className="w-1/3 flex flex-col">
                <div className="flex-shrink-0 p-4 border-b border-border flex items-center justify-between bg-background">
                  <h2 className="text-lg font-semibold truncate">
                    {selectedPDF}
                  </h2>
                  <Button variant="outline" size="sm" onClick={() => setSelectedPDF(null)}>
                    Close
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
