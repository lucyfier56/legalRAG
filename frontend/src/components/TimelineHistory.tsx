
import { useState } from "react";
import { Clock, FileText, Calendar, Trash2, ChevronRight } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";

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

interface TimelineHistoryProps {
  history: TimelineHistoryItem[];
  onSelectHistory: (item: TimelineHistoryItem) => void;
  onClearHistory: () => void;
  selectedHistoryId?: string;
}

export function TimelineHistory({ 
  history, 
  onSelectHistory, 
  onClearHistory,
  selectedHistoryId 
}: TimelineHistoryProps) {
  const formatTimestamp = (timestamp: string) => {
    return new Date(timestamp).toLocaleString();
  };

  return (
    <Card className="h-full">
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <CardTitle className="text-lg flex items-center gap-2">
            <Clock className="h-5 w-5" />
            Timeline History
          </CardTitle>
          {history.length > 0 && (
            <Button variant="outline" size="sm" onClick={onClearHistory}>
              <Trash2 className="h-4 w-4 mr-1" />
              Clear All
            </Button>
          )}
        </div>
      </CardHeader>
      <CardContent className="p-0 flex-1">
        <ScrollArea className="h-[400px]">
          {history.length === 0 ? (
            <div className="p-6 text-center text-muted-foreground">
              <Clock className="h-12 w-12 mx-auto mb-4 opacity-50" />
              <p className="text-sm">No timeline history yet.</p>
              <p className="text-xs mt-1">Process documents to see your timeline history here.</p>
            </div>
          ) : (
            <div className="p-4 space-y-3">
              {history.map((item) => (
                <Card 
                  key={item.id} 
                  className={`cursor-pointer transition-all hover:shadow-md ${
                    selectedHistoryId === item.id ? 'ring-2 ring-primary' : ''
                  }`}
                  onClick={() => onSelectHistory(item)}
                >
                  <CardContent className="p-4">
                    <div className="flex items-start justify-between mb-2">
                      <div className="flex-1">
                        <div className="flex items-center gap-2 mb-1">
                          <FileText className="h-4 w-4 text-muted-foreground" />
                          <h4 className="font-medium text-sm truncate">{item.pdfName}</h4>
                        </div>
                        <p className="text-xs text-muted-foreground mb-2">
                          {item.documentName}
                        </p>
                        <div className="flex items-center gap-2 text-xs text-muted-foreground">
                          <Calendar className="h-3 w-3" />
                          <span>{formatTimestamp(item.timestamp)}</span>
                        </div>
                      </div>
                      <div className="flex flex-col items-end gap-1">
                        <Badge variant="secondary" className="text-xs">
                          {item.eventCount} events
                        </Badge>
                        <ChevronRight className="h-4 w-4 text-muted-foreground" />
                      </div>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          )}
        </ScrollArea>
      </CardContent>
    </Card>
  );
}
