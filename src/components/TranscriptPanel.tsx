'use client';

import { ScrollArea } from '@/components/ui/scroll-area';
import { Badge } from '@/components/ui/badge';
import { useEffect, useRef } from 'react';

interface TranscriptEntry {
  text: string;
  start: number;
  end: number;
}

interface TranscriptPanelProps {
  transcript: TranscriptEntry[];
  currentTime: number;
  onTimeClick: (time: number) => void;
}

export function TranscriptPanel({ transcript, currentTime, onTimeClick }: TranscriptPanelProps) {
  const formatTime = (seconds: number) => {
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = Math.floor(seconds % 60);
    return `${minutes}:${remainingSeconds.toString().padStart(2, '0')}`;
  };

  // Refs for autoscroll
  const lineRefs = useRef<(HTMLDivElement | null)[]>([]);

  useEffect(() => {
    const activeIndex = transcript.findIndex(
      (entry) => currentTime >= entry.start && currentTime <= entry.end
    );
    if (activeIndex !== -1 && lineRefs.current[activeIndex]) {
      lineRefs.current[activeIndex]?.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
  }, [currentTime, transcript]);

  return (
    <ScrollArea className="h-[400px] w-full rounded-md border p-4">
      <div className="space-y-4">
        {transcript.map((entry, index) => (
          <div
            key={index}
            ref={el => { lineRefs.current[index] = el; }}
            className={`flex items-start gap-2 p-2 rounded-lg ${
              currentTime >= entry.start && currentTime <= entry.end
                ? 'bg-primary/10'
                : 'hover:bg-muted'
            }`}
          >
            <Badge
              variant="outline"
              className="cursor-pointer"
              onClick={() => onTimeClick(entry.start)}
            >
              {formatTime(entry.start)}
            </Badge>
            <p className="text-sm">{entry.text}</p>
          </div>
        ))}
      </div>
    </ScrollArea>
  );
}