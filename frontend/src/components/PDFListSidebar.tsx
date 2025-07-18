import { useState } from "react";
import { ChevronRight, ChevronLeft, FileText, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { cn } from "@/lib/utils";

interface PDFFile {
  name: string;
  size_mb: number;
  pages: number;
  relative_path: string;
}

interface PDFListSidebarProps {
  files: PDFFile[];
  isOpen: boolean;
  onToggle: () => void;
  onSelectPDF: (file: PDFFile) => void;
  selectedPDF?: string;
}

export function PDFListSidebar({ files, isOpen, onToggle, onSelectPDF, selectedPDF }: PDFListSidebarProps) {
  return (
    <>
      {/* Sidebar Toggle Button */}
      <Button
        onClick={onToggle}
        variant="outline"
        size="sm"
        className={cn(
          "fixed top-1/2 -translate-y-1/2 z-50 transition-all duration-300",
          isOpen ? "right-80" : "right-4"
        )}
      >
        {isOpen ? <ChevronRight className="h-4 w-4" /> : <ChevronLeft className="h-4 w-4" />}
      </Button>

      {/* Sidebar */}
      <div
        className={cn(
          "fixed top-0 right-0 h-full w-80 bg-background border-l border-border z-40 transition-transform duration-300",
          isOpen ? "translate-x-0" : "translate-x-full"
        )}
      >
        <div className="flex items-center justify-between p-4 border-b border-border">
          <h3 className="font-semibold text-lg">PDF Documents</h3>
          <Button onClick={onToggle} variant="ghost" size="sm">
            <X className="h-4 w-4" />
          </Button>
        </div>

        <ScrollArea className="h-[calc(100vh-80px)]">
          <div className="p-4 space-y-2">
            {files.map((file) => (
              <div
                key={file.name}
                className={cn(
                  "p-3 rounded-lg border cursor-pointer transition-all duration-200 hover:bg-muted/50",
                  selectedPDF === file.name 
                    ? "bg-primary/10 border-primary" 
                    : "bg-card border-border hover:border-primary/50"
                )}
                onClick={() => onSelectPDF(file)}
              >
                <div className="flex items-start gap-3">
                  <FileText className="h-5 w-5 text-primary mt-0.5 flex-shrink-0" />
                  <div className="flex-1 min-w-0">
                    <h4 className="font-medium text-sm leading-tight truncate">
                      {file.name.replace('.pdf', '')}
                    </h4>
                    <div className="flex items-center gap-2 mt-1 text-xs text-muted-foreground">
                      <span>{file.pages} pages</span>
                      <span>•</span>
                      <span>{file.size_mb} MB</span>
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </ScrollArea>
      </div>

      {/* Backdrop */}
      {isOpen && (
        <div
          className="fixed inset-0 bg-black/20 z-30"
          onClick={onToggle}
        />
      )}
    </>
  );
}