
import { ScrollArea } from "@/components/ui/scroll-area";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Trash2, MessageSquare } from "lucide-react";
import { ChatMessage } from "./ChatMessage";

interface ChatEntry {
  id: string;
  type: 'user' | 'assistant';
  content: string;
  timestamp: string;
  sources?: any[];
  queryType?: string;
  intent?: string;
}

interface ChatHistoryProps {
  chatHistory: ChatEntry[];
  onSourceClick: (source: any) => void;
  onClearHistory: () => void;
}

export function ChatHistory({ chatHistory, onSourceClick, onClearHistory }: ChatHistoryProps) {
  return (
    <div className="h-full flex flex-col">
      <Card className="flex-1 rounded-none border-0 border-b">
        <CardHeader className="p-4 border-b">
          <div className="flex items-center justify-between">
            <CardTitle className="text-lg flex items-center gap-2">
              <MessageSquare className="h-5 w-5" />
              Chat History
            </CardTitle>
            {chatHistory.length > 0 && (
              <Button
                variant="outline"
                size="sm"
                onClick={onClearHistory}
                className="text-xs"
              >
                <Trash2 className="h-3 w-3 mr-1" />
                Clear
              </Button>
            )}
          </div>
        </CardHeader>
        <CardContent className="p-0 flex-1">
          <ScrollArea className="h-full">
            {chatHistory.length === 0 ? (
              <div className="p-6 text-center text-muted-foreground">
                <MessageSquare className="h-12 w-12 mx-auto mb-4 opacity-50" />
                <p className="text-sm">No conversation history yet.</p>
                <p className="text-xs mt-1">Start asking questions to see your chat history here.</p>
              </div>
            ) : (
              <div className="p-4 space-y-4">
                {chatHistory.map((entry) => (
                  <ChatMessage
                    key={entry.id}
                    type={entry.type}
                    content={entry.content}
                    timestamp={entry.timestamp}
                    sources={entry.sources}
                    queryType={entry.queryType}
                    intent={entry.intent}
                    onSourceClick={onSourceClick}
                  />
                ))}
              </div>
            )}
          </ScrollArea>
        </CardContent>
      </Card>
    </div>
  );
}
