
import { Loader2, Folder, Trash2, RefreshCw } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { ScrollArea } from "@/components/ui/scroll-area";

interface Collection {
  id: string;
  name: string;
  file_count: number;
  created_date: string;
  total_size_mb: number;
}

interface CollectionsListProps {
  collections: Collection[];
  selectedCollection: Collection | null;
  isLoading: boolean;
  onSelectCollection: (collection: Collection) => void;
  onDeleteCollection: (collection: Collection) => void;
  onRefresh: () => void;
}

export function CollectionsList({
  collections,
  selectedCollection,
  isLoading,
  onSelectCollection,
  onDeleteCollection,
  onRefresh
}: CollectionsListProps) {
  return (
    <div className="flex flex-col h-full">
      <div className="flex-shrink-0 p-4 border-b border-border">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-lg font-semibold">Collections</h2>
          <Button onClick={onRefresh} variant="outline" size="sm" disabled={isLoading}>
            {isLoading ? (
              <Loader2 className="h-4 w-4 animate-spin" />
            ) : (
              <RefreshCw className="h-4 w-4" />
            )}
          </Button>
        </div>
      </div>

      <ScrollArea className="flex-1">
        <div className="p-4">
          {isLoading ? (
            <div className="flex items-center justify-center py-12">
              <div className="text-center">
                <Loader2 className="h-8 w-8 animate-spin mx-auto mb-4" />
                <span className="text-sm text-muted-foreground">Loading collections...</span>
              </div>
            </div>
          ) : collections.length > 0 ? (
            <div className="space-y-3">
              {collections.map((collection) => (
                <Card 
                  key={collection.id}
                  className={`cursor-pointer transition-all duration-200 hover:shadow-md ${
                    selectedCollection?.id === collection.id 
                      ? 'ring-2 ring-primary bg-primary/5' 
                      : 'hover:bg-muted/50'
                  }`}
                  onClick={() => onSelectCollection(collection)}
                >
                  <CardContent className="p-4">
                    <div className="flex items-start justify-between">
                      <div className="flex-1 min-w-0">
                        <h3 className="font-medium truncate mb-2">{collection.name}</h3>
                        <div className="space-y-1 text-sm text-muted-foreground">
                          <div className="flex items-center justify-between">
                            <span>{collection.file_count} files</span>
                            <span>{collection.total_size_mb.toFixed(1)} MB</span>
                          </div>
                          <div className="text-xs">
                            {new Date(collection.created_date).toLocaleDateString()}
                          </div>
                        </div>
                      </div>
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={(e) => {
                          e.stopPropagation();
                          onDeleteCollection(collection);
                        }}
                        className="text-destructive hover:text-destructive ml-2"
                      >
                        <Trash2 className="h-4 w-4" />
                      </Button>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          ) : (
            <div className="text-center py-12">
              <Folder className="h-12 w-12 mx-auto mb-4 opacity-50" />
              <p className="text-sm text-muted-foreground">No collections found</p>
              <p className="text-xs text-muted-foreground mt-1">Create your first collection below</p>
            </div>
          )}
        </div>
      </ScrollArea>
    </div>
  );
}
