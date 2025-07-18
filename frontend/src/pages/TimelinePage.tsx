import { useState, useEffect } from "react";
import { FileText, Folder, Clock, Loader2, Download, Sparkles, Upload, History } from "lucide-react";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { useToast } from "@/hooks/use-toast";
import { PDFViewer } from "@/components/PDFViewer";
import { FileUpload } from "@/components/FileUpload";
import { FolderSelector } from "@/components/FolderSelector";
import { TimelineViewer } from "@/components/TimelineViewer";
import { TimelineHistory } from "@/components/TimelineHistory";
import { PDFListSidebar } from "@/components/PDFListSidebar";
import { SidebarProvider } from "@/components/ui/sidebar";
import { AppSidebar } from "@/components/AppSidebar";
import { Toaster } from "@/components/ui/toaster";

interface TimelineEvent {
  date: string;
  event: string;
  page: number;
  context: string;
  confidence: number;
}

interface TimelineHistoryItem {
  id: string;
  pdfName: string;
  documentName: string;
  timestamp: string;
  eventCount: number;
  timelineData: TimelineEvent[];
  wordDocPath?: string;
}

interface BackendTimelineEvent {
  date: string;
  event_details?: string;
  event?: string;
  page_numbers?: number[];
  source_pages?: string;
  source_document?: string;
  pdf_filename?: string;
}

interface TimelineResponse {
  success: boolean;
  timeline_data: BackendTimelineEvent[];
  total_events: number;
  document_name: string;
  word_document_path: string;
  source_pdf_path: string;
  error?: string;
}

interface PDFFile {
  name: string;
  size_mb: number;
  pages: number;
  relative_path: string;
  display_name: string;
  session_id?: string;
}

interface FolderUploadResponse {
  success: boolean;
  uploaded_files: string[];
  failed_files: string[];
  total_files: number;
  folder_session_id: string;
  error?: string;
}

export function TimelinePage() {
  const [uploadMode, setUploadMode] = useState<"single" | "folder">("single");
  const [isProcessing, setIsProcessing] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [timelineData, setTimelineData] = useState<TimelineEvent[]>([]);
  const [selectedPDF, setSelectedPDF] = useState<string | null>(null);
  const [currentDocument, setCurrentDocument] = useState<string | null>(null);
  const [wordDocPath, setWordDocPath] = useState<string | null>(null);
  const [selectedPage, setSelectedPage] = useState<number>(1);
  const [folderFiles, setFolderFiles] = useState<PDFFile[]>([]);
  const [isPDFSidebarOpen, setIsPDFSidebarOpen] = useState(false);
  const [selectedFiles, setSelectedFiles] = useState<string[]>([]);
  const [currentSessionId, setCurrentSessionId] = useState<string | null>(null);
  const [hasFolderUploaded, setHasFolderUploaded] = useState(false);
  
  const [timelineHistory, setTimelineHistory] = useState<TimelineHistoryItem[]>([]);
  const [showHistory, setShowHistory] = useState(false);
  const [selectedHistoryId, setSelectedHistoryId] = useState<string | null>(null);
  
  const { toast } = useToast();

  useEffect(() => {
    const savedHistory = localStorage.getItem('timeline-history');
    if (savedHistory) {
      try {
        setTimelineHistory(JSON.parse(savedHistory));
      } catch (error) {
        console.error('Error loading timeline history:', error);
      }
    }
  }, []);

  useEffect(() => {
    localStorage.setItem('timeline-history', JSON.stringify(timelineHistory));
  }, [timelineHistory]);

  const saveToHistory = (
    pdfName: string,
    documentName: string,
    timelineData: TimelineEvent[],
    wordDocPath?: string
  ) => {
    const historyItem: TimelineHistoryItem = {
      id: `timeline-${Date.now()}`,
      pdfName,
      documentName,
      timestamp: new Date().toISOString(),
      eventCount: timelineData.length,
      timelineData,
      wordDocPath
    };

    setTimelineHistory(prev => [historyItem, ...prev]);
  };

  const loadFromHistory = (item: TimelineHistoryItem) => {
    setTimelineData(item.timelineData);
    setCurrentDocument(item.documentName);
    setSelectedPDF(item.pdfName);
    setWordDocPath(item.wordDocPath || null);
    setSelectedHistoryId(item.id);
    setSelectedPage(1);
    
    setShowHistory(false);
    
    toast({
      title: "Timeline Loaded",
      description: `Loaded timeline for ${item.pdfName}`,
    });
  };

  const clearHistory = () => {
    setTimelineHistory([]);
    setSelectedHistoryId(null);
    toast({
      title: "History Cleared",
      description: "Timeline history has been cleared",
    });
  };

  const handleSingleFileUpload = async (file: File) => {
    setIsProcessing(true);
    try {
      const formData = new FormData();
      formData.append("file", file);
      formData.append("document_name", file.name.replace(".pdf", ""));

      const response = await fetch("http://localhost:8000/timeline/extract", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data: TimelineResponse = await response.json();
      
      if (data.success && data.timeline_data) {
        const transformedData = data.timeline_data.map((item: BackendTimelineEvent) => ({
          date: item.date || '',
          event: item.event_details || item.event || '',
          page: item.page_numbers?.[0] || 1,
          context: item.source_pages || '',
          confidence: 0.8 // Default confidence since backend doesn't provide this
        }));

        setTimelineData(transformedData);
        setCurrentDocument(data.document_name);
        setWordDocPath(data.word_document_path);
        setSelectedPDF(file.name);
        setSelectedHistoryId(null);
        
        saveToHistory(file.name, data.document_name, transformedData, data.word_document_path);
        
        toast({
          title: "Timeline Extracted",
          description: `Found ${data.total_events} timeline events`,
        });
      } else {
        throw new Error(data.error || "Timeline extraction failed");
      }
    } catch (error) {
      console.error("Timeline extraction failed:", error);
      toast({
        title: "Extraction Failed",
        description: "Failed to extract timeline. Please try again.",
        variant: "destructive",
      });
    } finally {
      setIsProcessing(false);
    }
  };

  const handleFolderUpload = async (files: FileList) => {
    if (!files || files.length === 0) {
      toast({
        title: "No Files Selected",
        description: "Please select PDF files to upload.",
        variant: "destructive",
      });
      return;
    }

    setIsUploading(true);
    try {
      const formData = new FormData();
      
      const pdfFiles = Array.from(files).filter(file => 
        file.name.toLowerCase().endsWith('.pdf')
      );

      if (pdfFiles.length === 0) {
        toast({
          title: "No PDF Files",
          description: "Please select PDF files only.",
          variant: "destructive",
        });
        return;
      }

      pdfFiles.forEach(file => {
        formData.append("files", file);
      });

      const response = await fetch("http://localhost:8000/timeline/upload-folder", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data: FolderUploadResponse = await response.json();
      
      if (data.success) {
        setCurrentSessionId(data.folder_session_id);
        setHasFolderUploaded(true);
        
        await loadFolderContents(data.folder_session_id);
        
        toast({
          title: "Folder Uploaded",
          description: `Uploaded ${data.total_files} PDF files successfully`,
        });
      } else {
        throw new Error(data.error || "Folder upload failed");
      }
    } catch (error) {
      console.error("Folder upload failed:", error);
      toast({
        title: "Upload Failed",
        description: "Failed to upload folder. Please try again.",
        variant: "destructive",
      });
    } finally {
      setIsUploading(false);
    }
  };

  const loadFolderContents = async (sessionId: string) => {
    try {
      const response = await fetch(`http://localhost:8000/timeline/folder-contents?session_id=${sessionId}`);
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const data = await response.json();
      if (data.success) {
        setFolderFiles(data.files);
        setIsPDFSidebarOpen(true);
      } else {
        throw new Error(data.error || "Failed to load folder contents");
      }
    } catch (error) {
      console.error("Failed to load folder contents:", error);
      toast({
        title: "Error",
        description: "Failed to load uploaded files",
        variant: "destructive",
      });
    }
  };

  const handlePDFSelect = async (file: PDFFile) => {
    setIsProcessing(true);
    setIsPDFSidebarOpen(false);
    
    try {
      const response = await fetch(`http://localhost:8000/timeline/process-selected?session_id=${currentSessionId}`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          selected_files: [file.relative_path],
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      
      if (data.success && data.results && data.results.length > 0) {
        const result = data.results[0];
        
        const transformedData = result.timeline_data.map((item: BackendTimelineEvent) => ({
          date: item.date || '',
          event: item.event_details || item.event || '',
          page: item.page_numbers?.[0] || 1,
          context: item.source_pages || '',
          confidence: 0.8 // Default confidence
        }));

        setTimelineData(transformedData);
        setCurrentDocument(result.document_name);
        setWordDocPath(result.word_document_path);
        setSelectedPDF(file.name);
        setSelectedHistoryId(null);
        
        saveToHistory(file.name, result.document_name, transformedData, result.word_document_path);
        
        toast({
          title: "Timeline Extracted",
          description: `Found ${result.timeline_data.length} timeline events`,
        });
      } else {
        throw new Error("No timeline data found");
      }
    } catch (error) {
      console.error("Processing failed:", error);
      toast({
        title: "Processing Failed",
        description: "Failed to process document. Please try again.",
        variant: "destructive",
      });
    } finally {
      setIsProcessing(false);
    }
  };

  const handleTimelineClick = (event: TimelineEvent) => {
    if (selectedPDF) {
      setSelectedPage(event.page);
      toast({
        title: "Navigating to Page",
        description: `Jumping to page ${event.page}`,
      });
    }
  };

  const downloadWordDoc = () => {
    if (wordDocPath) {
      const filename = wordDocPath.split('/').pop() || 'timeline.docx';
      window.open(`http://localhost:8000/timeline/download/${filename}`, '_blank');
    }
  };

  const resetToUpload = () => {
    setHasFolderUploaded(false);
    setCurrentSessionId(null);
    setFolderFiles([]);
    setIsPDFSidebarOpen(false);
    setTimelineData([]);
    setSelectedPDF(null);
    setCurrentDocument(null);
    setWordDocPath(null);
    setSelectedHistoryId(null);
  };

  const hasTimelineData = timelineData.length > 0;

  return (
    <SidebarProvider>
      <div className="min-h-screen flex w-full bg-background">
        <AppSidebar />
        <main className="flex-1 overflow-hidden">
          <div className="flex h-screen">
            <div className={`flex-1 flex flex-col transition-all duration-300 ${showHistory ? 'mr-80' : ''}`}>
              
              <div className="p-4 border-b border-border flex items-center justify-between">
                <h1 className="text-2xl font-bold">Timeline Extraction</h1>
                <div className="flex gap-2">
                  <Button 
                    variant="outline" 
                    onClick={() => setShowHistory(!showHistory)}
                    className="flex items-center gap-2"
                  >
                    <History className="h-4 w-4" />
                    {showHistory ? 'Hide History' : 'Show History'}
                  </Button>
                  {hasTimelineData && (
                    <Button 
                      variant="ghost" 
                      onClick={resetToUpload}
                      size="sm"
                    >
                      Process Another Document
                    </Button>
                  )}
                </div>
              </div>

              <div className="flex-1 flex">
                <div className={`${hasTimelineData ? 'w-1/2' : 'w-full'} flex flex-col ${hasTimelineData ? 'border-r border-border' : ''}`}>
                  <ScrollArea className="flex-1 p-8">
                    {!hasTimelineData && !isProcessing && !isUploading && (
                      <div className="flex flex-col items-center justify-center h-full text-center max-w-2xl mx-auto">
                        <div className="p-6 rounded-full bg-primary/10 mb-6">
                          <Clock className="h-16 w-16 text-primary" />
                        </div>
                        <h2 className="text-3xl font-bold mb-4">Timeline Extraction</h2>
                        <p className="text-lg text-muted-foreground mb-8">
                          Extract chronological timelines from legal documents. Upload a single file or upload multiple files to browse and select.
                        </p>
                        
                        <div className="flex gap-2 mb-6">
                          <Button
                            variant={uploadMode === "single" ? "default" : "outline"}
                            onClick={() => setUploadMode("single")}
                            className="flex items-center gap-2"
                            size="lg"
                          >
                            <FileText className="h-5 w-5" />
                            Single File
                          </Button>
                          <Button
                            variant={uploadMode === "folder" ? "default" : "outline"}
                            onClick={() => setUploadMode("folder")}
                            className="flex items-center gap-2"
                            size="lg"
                          >
                            <Folder className="h-5 w-5" />
                            Multiple Files
                          </Button>
                        </div>

                        <div className="w-full max-w-md">
                          {uploadMode === "single" ? (
                            <FileUpload 
                              onFileUpload={handleSingleFileUpload}
                              disabled={isProcessing}
                            />
                          ) : (
                            <div className="space-y-4">
                              {!hasFolderUploaded ? (
                                <div>
                                  <input
                                    type="file"
                                    multiple
                                    accept=".pdf"
                                    onChange={(e) => e.target.files && handleFolderUpload(e.target.files)}
                                    className="hidden"
                                    id="folder-upload"
                                    disabled={isUploading}
                                  />
                                  <label
                                    htmlFor="folder-upload"
                                    className={`w-full h-32 border-2 border-dashed border-primary/30 bg-primary/5 hover:bg-primary/10 flex flex-col gap-2 items-center justify-center rounded-lg cursor-pointer transition-colors ${isUploading ? 'opacity-50 cursor-not-allowed' : ''}`}
                                  >
                                    <Upload className="h-8 w-8 text-primary" />
                                    <span className="font-medium">Upload Multiple PDFs</span>
                                    <span className="text-sm text-muted-foreground">Select multiple PDF files to process</span>
                                  </label>
                                </div>
                              ) : (
                                <div className="text-center space-y-4">
                                  <div className="p-4 bg-green-50 border border-green-200 rounded-lg">
                                    <p className="text-green-800 font-medium">Files Uploaded Successfully!</p>
                                    <p className="text-green-600 text-sm mt-1">
                                      {folderFiles.length} PDF files ready for processing
                                    </p>
                                  </div>
                                  <div className="flex gap-2">
                                    <Button 
                                      onClick={() => setIsPDFSidebarOpen(true)}
                                      className="flex-1"
                                    >
                                      <FileText className="h-4 w-4 mr-2" />
                                      Browse & Select Files
                                    </Button>
                                    <Button 
                                      variant="outline"
                                      onClick={resetToUpload}
                                    >
                                      Upload Different Files
                                    </Button>
                                  </div>
                                </div>
                              )}
                            </div>
                          )}
                        </div>
                      </div>
                    )}

                    {(isProcessing || isUploading) && (
                      <div className="flex flex-col items-center justify-center h-full text-center">
                        <div className="relative">
                          <div className="w-20 h-20 border-4 border-primary/20 border-t-primary rounded-full animate-spin mb-6"></div>
                          <Sparkles className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 h-8 w-8 text-primary animate-pulse" />
                        </div>
                        <h3 className="text-xl font-semibold mb-2">
                          {isUploading ? "Uploading files..." : "Processing timeline..."}
                        </h3>
                        <p className="text-muted-foreground">
                          {isUploading ? "Uploading your PDF files to the server..." : "Extracting chronological events from document..."}
                        </p>
                      </div>
                    )}

                    {hasTimelineData && (
                      <div className="space-y-6">
                        <div className="bg-muted/30 rounded-xl p-6">
                          <div className="flex items-center justify-between">
                            <div>
                              <h2 className="text-xl font-bold flex items-center gap-2">
                                <Clock className="h-5 w-5 text-primary" />
                                {currentDocument}
                              </h2>
                              <p className="text-muted-foreground mt-1">
                                {timelineData.length} timeline events found
                                {selectedHistoryId && (
                                  <span className="ml-2 text-xs bg-primary/10 text-primary px-2 py-1 rounded">
                                    From History
                                  </span>
                                )}
                              </p>
                            </div>
                            <div className="flex gap-2">
                              {wordDocPath && (
                                <Button onClick={downloadWordDoc} variant="outline">
                                  <Download className="h-4 w-4 mr-2" />
                                  Download Word
                                </Button>
                              )}
                            </div>
                          </div>
                        </div>

                        <TimelineViewer 
                          events={timelineData} 
                          onEventClick={handleTimelineClick}
                        />
                      </div>
                    )}
                  </ScrollArea>
                </div>

                {hasTimelineData && (
                  <div className="w-1/2 flex flex-col">
                    <div className="p-4 border-b border-border bg-background/95 backdrop-blur-sm">
                      <h2 className="text-lg font-semibold">
                        {selectedPDF ? `Viewing: ${selectedPDF}` : "Document Viewer"}
                      </h2>
                    </div>
                    <div className="flex-1">
                      {selectedPDF ? (
                        <PDFViewer filename={selectedPDF} initialPage={selectedPage} />
                      ) : (
                        <div className="flex items-center justify-center h-full text-center">
                          <div>
                            <FileText className="h-12 w-12 text-muted-foreground mb-4 mx-auto" />
                            <h3 className="text-lg font-medium mb-2">No Document Selected</h3>
                            <p className="text-muted-foreground">
                              Upload a document to view it here
                            </p>
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                )}
              </div>
            </div>

            {showHistory && (
              <div className="fixed right-0 top-0 h-full w-80 bg-background border-l border-border shadow-lg z-50">
                <TimelineHistory
                  history={timelineHistory}
                  onSelectHistory={loadFromHistory}
                  onClearHistory={clearHistory}
                  selectedHistoryId={selectedHistoryId}
                />
              </div>
            )}
          </div>
        </main>
        
        <PDFListSidebar
          files={folderFiles}
          isOpen={isPDFSidebarOpen}
          onToggle={() => setIsPDFSidebarOpen(!isPDFSidebarOpen)}
          onSelectPDF={handlePDFSelect}
          selectedPDF={selectedPDF}
        />
        
        <Toaster />
      </div>
    </SidebarProvider>
  );
}

export default TimelinePage;
