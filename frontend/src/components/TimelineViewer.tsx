import { Calendar, MapPin, Clock, FileText } from "lucide-react";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";

interface TimelineEvent {
  date: string;
  event: string;
  page: number;
  context: string;
  confidence: number;
}

interface TimelineViewerProps {
  events: TimelineEvent[];
  onEventClick: (event: TimelineEvent) => void;
}

export function TimelineViewer({ events, onEventClick }: TimelineViewerProps) {
  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.8) return "bg-green-100 text-green-800 border-green-200";
    if (confidence >= 0.6) return "bg-yellow-100 text-yellow-800 border-yellow-200";
    return "bg-red-100 text-red-800 border-red-200";
  };

  const getConfidenceText = (confidence: number) => {
    if (confidence >= 0.8) return "High";
    if (confidence >= 0.6) return "Medium";
    return "Low";
  };

  const formatDate = (dateString: string) => {
    try {
      // Handle various date formats that might come from the backend
      const date = new Date(dateString);
      if (isNaN(date.getTime())) {
        // If it's not a valid date, return the original string
        return dateString;
      }
      return date.toLocaleDateString('en-US', {
        year: 'numeric',
        month: 'long',
        day: 'numeric'
      });
    } catch {
      return dateString;
    }
  };

  const sortedEvents = [...events].sort((a, b) => {
    try {
      const dateA = new Date(a.date);
      const dateB = new Date(b.date);
      
      // If both dates are valid, sort by date
      if (!isNaN(dateA.getTime()) && !isNaN(dateB.getTime())) {
        return dateA.getTime() - dateB.getTime();
      }
      
      // Otherwise, sort alphabetically
      return a.date.localeCompare(b.date);
    } catch {
      return a.date.localeCompare(b.date);
    }
  });

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-2 mb-6">
        <Clock className="h-5 w-5 text-primary" />
        <h2 className="text-lg font-semibold">Timeline Events</h2>
        <Badge variant="outline">{events.length} events</Badge>
      </div>

      <div className="relative">
        {/* Timeline line */}
        <div className="absolute left-6 top-0 bottom-0 w-px bg-border"></div>

        {sortedEvents.map((event, index) => (
          <div key={index} className="relative flex items-start gap-4 pb-8">
            {/* Timeline dot */}
            <div className="relative z-10 flex h-12 w-12 items-center justify-center rounded-full bg-primary text-primary-foreground">
              <Calendar className="h-5 w-5" />
            </div>

            {/* Event card */}
            <Card className="flex-1 cursor-pointer hover:shadow-md transition-shadow" onClick={() => onEventClick(event)}>
              <CardContent className="p-4">
                <div className="flex items-start justify-between mb-2">
                  <div className="flex-1">
                    <div className="flex items-center gap-2 mb-1">
                      <h3 className="font-semibold text-lg">
                        {formatDate(event.date)}
                      </h3>
                      <Badge 
                        variant="outline" 
                        className={getConfidenceColor(event.confidence)}
                      >
                        {getConfidenceText(event.confidence)}
                      </Badge>
                    </div>
                    <p className="text-sm text-muted-foreground mb-2">
                      {event.event}
                    </p>
                  </div>
                  <div className="flex items-center gap-1 text-sm text-muted-foreground">
                    <FileText className="h-3 w-3" />
                    <span>Page {event.page}</span>
                  </div>
                </div>

                {event.context && (
                  <div className="mt-3 p-3 bg-muted/50 rounded-md">
                    <p className="text-sm italic text-muted-foreground">
                      "{event.context}"
                    </p>
                  </div>
                )}

                <div className="mt-3 flex items-center justify-between">
                  <div className="flex items-center gap-2 text-xs text-muted-foreground">
                    <MapPin className="h-3 w-3" />
                    <span>Confidence: {(event.confidence * 100).toFixed(1)}%</span>
                  </div>
                  <Button variant="ghost" size="sm" onClick={() => onEventClick(event)}>
                    Go to Page
                  </Button>
                </div>
              </CardContent>
            </Card>
          </div>
        ))}
      </div>

      {events.length === 0 && (
        <div className="text-center py-12">
          <Clock className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
          <h3 className="text-lg font-medium mb-2">No Timeline Events</h3>
          <p className="text-muted-foreground">
            No timeline events have been extracted yet
          </p>
        </div>
      )}
    </div>
  );
}
