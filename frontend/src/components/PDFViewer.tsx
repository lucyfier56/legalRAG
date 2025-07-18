
import { useState, useEffect } from "react";
import { ChevronLeft, ChevronRight, ZoomIn, ZoomOut, Download } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card } from "@/components/ui/card";
import { useToast } from "@/hooks/use-toast";

interface PDFViewerProps {
  filename: string;
  initialPage?: number;
}

export function PDFViewer({ filename, initialPage = 1 }: PDFViewerProps) {
  const [currentPage, setCurrentPage] = useState(initialPage);
  const [totalPages, setTotalPages] = useState(1);
  const [zoom, setZoom] = useState(100);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const { toast } = useToast();

  const pdfUrl = `http://localhost:8000/pdf/serve/${filename}`;
  const downloadUrl = `http://localhost:8000/pdf/download/${filename}`;

  useEffect(() => {
    if (initialPage !== currentPage) {
      setCurrentPage(initialPage);
    }
  }, [initialPage]);

  useEffect(() => {
    setIsLoading(true);
    setError(null);
  }, [filename]);

  const handlePageChange = (page: number) => {
    if (page >= 1 && page <= totalPages) {
      setCurrentPage(page);
    }
  };

  const handleZoomIn = () => {
    setZoom(prev => Math.min(prev + 25, 200));
  };

  const handleZoomOut = () => {
    setZoom(prev => Math.max(prev - 25, 50));
  };

  const handleDownload = () => {
    window.open(downloadUrl, '_blank');
    toast({
      title: "Download Started",
      description: `Downloading ${filename}`,
    });
  };

  return (
    <div className="flex flex-col h-full bg-gray-50">
      {/* PDF Controls */}
      <div className="flex items-center justify-between p-4 bg-white border-b border-border">
        <div className="flex items-center gap-2">
          <Button
            variant="outline"
            size="sm"
            onClick={() => handlePageChange(currentPage - 1)}
            disabled={currentPage <= 1}
          >
            <ChevronLeft className="h-4 w-4" />
          </Button>
          
          <div className="flex items-center gap-2">
            <Input
              type="number"
              value={currentPage}
              onChange={(e) => handlePageChange(parseInt(e.target.value) || 1)}
              className="w-16 text-center"
              min={1}
              max={totalPages}
            />
            <span className="text-sm text-muted-foreground">of {totalPages}</span>
          </div>
          
          <Button
            variant="outline"
            size="sm"
            onClick={() => handlePageChange(currentPage + 1)}
            disabled={currentPage >= totalPages}
          >
            <ChevronRight className="h-4 w-4" />
          </Button>
        </div>

        <div className="flex items-center gap-2">
          <Button variant="outline" size="sm" onClick={handleZoomOut}>
            <ZoomOut className="h-4 w-4" />
          </Button>
          <span className="text-sm text-muted-foreground w-12 text-center">
            {zoom}%
          </span>
          <Button variant="outline" size="sm" onClick={handleZoomIn}>
            <ZoomIn className="h-4 w-4" />
          </Button>
          
          <Button variant="outline" size="sm" onClick={handleDownload}>
            <Download className="h-4 w-4" />
          </Button>
        </div>
      </div>

      {/* PDF Display */}
      <div className="flex-1 overflow-auto">
        {error ? (
          <div className="flex items-center justify-center h-full">
            <Card className="p-6 text-center">
              <p className="text-red-500 mb-2">Error loading PDF</p>
              <p className="text-sm text-muted-foreground">{error}</p>
            </Card>
          </div>
        ) : (
          <div className="flex justify-center p-4">
            <iframe
              src={`${pdfUrl}#page=${currentPage}&zoom=${zoom}`}
              width="100%"
              height="800px"
              style={{ 
                border: "1px solid #e2e8f0",
                borderRadius: "8px",
                backgroundColor: "white"
              }}
              onLoad={() => setIsLoading(false)}
              onError={() => {
                setIsLoading(false);
                setError("Failed to load PDF");
              }}
            />
          </div>
        )}
      </div>
    </div>
  );
}
