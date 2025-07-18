
import { useState, useRef } from "react";
import { Upload, Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { useToast } from "@/hooks/use-toast";

interface CollectionUploadProps {
  onUploadSuccess: () => void;
}

export function CollectionUpload({ onUploadSuccess }: CollectionUploadProps) {
  const [isUploading, setIsUploading] = useState(false);
  const [collectionName, setCollectionName] = useState("");
  const fileInputRef = useRef<HTMLInputElement>(null);
  const { toast } = useToast();

  const handleFolderUpload = async (files: FileList) => {
    if (!files || files.length === 0) {
      toast({
        title: "No Files Selected",
        description: "Please select PDF files to upload.",
        variant: "destructive",
      });
      return;
    }

    if (!collectionName.trim()) {
      toast({
        title: "Collection Name Required",
        description: "Please enter a name for your collection.",
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
        setIsUploading(false);
        return;
      }

      pdfFiles.forEach(file => {
        formData.append("files", file);
      });
      formData.append("collection_name", collectionName.trim());

      const response = await fetch("http://localhost:8000/collections/upload", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      
      if (data.success) {
        toast({
          title: "Collection Created",
          description: `Created "${data.collection_name}" with ${data.file_count} files`,
        });
        
        setCollectionName("");
        if (fileInputRef.current) {
          fileInputRef.current.value = "";
        }
        onUploadSuccess();
      } else {
        throw new Error(data.error || "Collection upload failed");
      }
    } catch (error) {
      console.error("Collection upload failed:", error);
      toast({
        title: "Upload Failed",
        description: error instanceof Error ? error.message : "Failed to upload collection",
        variant: "destructive",
      });
    } finally {
      setIsUploading(false);
    }
  };

  return (
    <div className="flex-shrink-0 p-4 border-t border-border bg-background">
      <h3 className="text-sm font-semibold mb-3">Create Collection</h3>
      <div className="space-y-3">
        <Input
          placeholder="Collection name..."
          value={collectionName}
          onChange={(e) => setCollectionName(e.target.value)}
          disabled={isUploading}
          className="text-sm"
        />
        
        <div>
          <input
            ref={fileInputRef}
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
            className={`w-full h-20 border-2 border-dashed border-primary/30 bg-primary/5 hover:bg-primary/10 flex flex-col gap-1 items-center justify-center rounded-lg cursor-pointer transition-colors text-sm ${isUploading ? 'opacity-50 cursor-not-allowed' : ''}`}
          >
            {isUploading ? (
              <>
                <Loader2 className="h-5 w-5 text-primary animate-spin" />
                <span className="font-medium">Creating...</span>
              </>
            ) : (
              <>
                <Upload className="h-5 w-5 text-primary" />
                <span className="font-medium">Upload PDFs</span>
                <span className="text-xs text-muted-foreground">Select multiple files</span>
              </>
            )}
          </label>
        </div>
      </div>
    </div>
  );
}
