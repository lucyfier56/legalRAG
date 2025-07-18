
import { useState, useEffect } from "react";
import { Folder, FileText, RefreshCw, Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Checkbox } from "@/components/ui/checkbox";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Badge } from "@/components/ui/badge";
import { useToast } from "@/hooks/use-toast";

interface PDFFile {
  file_path: string;
  relative_path: string;
  display_name: string;
  size: number;
  modified_time: string;
  page_count?: number;
}

interface FolderSelectorProps {
  onFilesSelected: (selectedFiles: string[]) => void;
  disabled?: boolean;
}

export function FolderSelector({ onFilesSelected, disabled = false }: FolderSelectorProps) {
  const [files, setFiles] = useState<PDFFile[]>([]);
  const [selectedFiles, setSelectedFiles] = useState<string[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const { toast } = useToast();

  const loadFolderContents = async () => {
    setIsLoading(true);
    try {
      const response = await fetch("http://localhost:8000/timeline/folder-contents");
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const data = await response.json();
      if (data.success) {
        setFiles(data.files);
        toast({
          title: "Folder Loaded",
          description: `Found ${data.total_files} PDF files`,
        });
      } else {
        throw new Error(data.error || "Failed to load folder contents");
      }
    } catch (error) {
      console.error("Failed to load folder contents:", error);
      toast({
        title: "Loading Failed",
        description: "Failed to load folder contents. Please try again.",
        variant: "destructive",
      });
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    loadFolderContents();
  }, []);

  const handleFileToggle = (filePath: string) => {
    setSelectedFiles(prev => 
      prev.includes(filePath)
        ? prev.filter(f => f !== filePath)
        : [...prev, filePath]
    );
  };

  const handleSelectAll = () => {
    if (selectedFiles.length === files.length) {
      setSelectedFiles([]);
    } else {
      setSelectedFiles(files.map(f => f.relative_path));
    }
  };

  const handleProcessSelected = () => {
    if (selectedFiles.length > 0) {
      onFilesSelected(selectedFiles);
    }
  };

  const formatFileSize = (bytes: number) => {
    return (bytes / 1024 / 1024).toFixed(2) + " MB";
  };

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString();
  };

  return (
    <div className="space-y-4">
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <CardTitle className="flex items-center gap-2">
              <Folder className="h-5 w-5" />
              Document Folder
            </CardTitle>
            <Button
              variant="outline"
              size="sm"
              onClick={loadFolderContents}
              disabled={isLoading}
            >
              {isLoading ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                <RefreshCw className="h-4 w-4" />
              )}
            </Button>
          </div>
        </CardHeader>
        <CardContent>
          {isLoading ? (
            <div className="flex items-center justify-center py-8">
              <Loader2 className="h-8 w-8 animate-spin" />
              <span className="ml-2">Loading files...</span>
            </div>
          ) : files.length > 0 ? (
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Checkbox
                    checked={selectedFiles.length === files.length}
                    onCheckedChange={handleSelectAll}
                    disabled={disabled}
                  />
                  <span className="text-sm font-medium">
                    Select All ({files.length} files)
                  </span>
                </div>
                <Badge variant="outline">
                  {selectedFiles.length} selected
                </Badge>
              </div>

              <ScrollArea className="h-64 border rounded-md p-4">
                <div className="space-y-2">
                  {files.map((file, index) => (
                    <div
                      key={index}
                      className="flex items-center space-x-3 p-2 rounded-lg hover:bg-muted/50"
                    >
                      <Checkbox
                        checked={selectedFiles.includes(file.relative_path)}
                        onCheckedChange={() => handleFileToggle(file.relative_path)}
                        disabled={disabled}
                      />
                      <FileText className="h-4 w-4 text-primary" />
                      <div className="flex-1 min-w-0">
                        <p className="text-sm font-medium truncate">
                          {file.display_name}
                        </p>
                        <p className="text-xs text-muted-foreground">
                          {formatFileSize(file.size)} • {formatDate(file.modified_time)}
                          {file.page_count && ` • ${file.page_count} pages`}
                        </p>
                      </div>
                    </div>
                  ))}
                </div>
              </ScrollArea>

              <Button
                onClick={handleProcessSelected}
                disabled={selectedFiles.length === 0 || disabled}
                className="w-full"
              >
                Process Selected Files ({selectedFiles.length})
              </Button>
            </div>
          ) : (
            <div className="text-center py-8">
              <Folder className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
              <p className="text-lg font-medium mb-2">No PDF Files Found</p>
              <p className="text-sm text-muted-foreground">
                No PDF files were found in the documents folder
              </p>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
