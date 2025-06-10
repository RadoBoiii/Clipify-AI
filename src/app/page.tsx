'use client';

import { useState, useRef, useEffect } from 'react';
import { VideoUploader } from '../components/VideoUploader';
import { TranscriptPanel } from '../components/TranscriptPanel';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import YouTubePlayer, { YouTubePlayerHandle } from '../components/YouTubePlayer';

interface TranscriptSegment {
  text: string;
  start: number;
  end: number;
  metadata?: Record<string, unknown>;
}

interface Chapter {
  title: string;
  start_time: number;
  end_time: number;
  summary: string;
}

export default function Home() {
  const [videoId, setVideoId] = useState<string>('');
  const [currentTime, setCurrentTime] = useState(0);
  const [transcript, setTranscript] = useState<TranscriptSegment[]>([]);
  const [chapters, setChapters] = useState<Chapter[]>([]);
  const [ragResponse, setRagResponse] = useState<string>('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [chatMessages, setChatMessages] = useState<{ role: 'user' | 'agent'; content: string }[]>([]);
  const [chatInput, setChatInput] = useState('');
  const [chatLoading, setChatLoading] = useState(false);

  const playerRef = useRef<YouTubePlayerHandle>(null);

  useEffect(() => {
    setChatMessages([]);
    setChatInput('');
  }, [videoId]);

  const handleVideoSubmit = async (url: string) => {
    try {
      setIsLoading(true);
      setError(null);
      
      // Extract video ID from URL
      const videoId = url.split('v=')[1]?.split('&')[0];
      if (!videoId) throw new Error('Invalid YouTube URL');
      setVideoId(videoId);

      // Call API to analyze video
      const response = await fetch('/api/analyze-video', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ url }),
      });
      
      const data = await response.json();
      
      if (!response.ok) {
        throw new Error(data.error || 'Failed to analyze video');
      }
      
      if (!data.success) {
        throw new Error(data.error || 'Analysis failed');
      }
      
      // Update state with analysis results
      setTranscript(data.transcript || []);
      setChapters(data.chapters || []);
      setRagResponse(data.rag_response || '');
      
    } catch (error) {
      console.error('Error analyzing video:', error);
      setError(error instanceof Error ? error.message : 'An error occurred');
      // Clear previous results on error
      setTranscript([]);
      setChapters([]);
      setRagResponse('');
    } finally {
      setIsLoading(false);
    }
  };

  const handleTranscriptTimeClick = (time: number) => {
    playerRef.current?.seekTo(time);
  };

  const handleChapterClick = (startTime: number) => {
    playerRef.current?.seekTo(startTime);
  };

  const handleSendChat = async () => {
    if (!chatInput.trim()) return;
    const userMessage: { role: 'user' | 'agent'; content: string } = { role: 'user', content: chatInput };
    setChatMessages((prev) => [...prev, userMessage]);
    setChatInput('');
    setChatLoading(true);
    try {
      const res = await fetch('/api/video-chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          question: userMessage.content,
          video: {
            metadata: transcript.length ? transcript[0].metadata : {},
            transcript,
            chapters,
          },
        }),
      });
      const data = await res.json();
      setChatMessages((prev) => [...prev, { role: 'agent' as const, content: data.answer || data.error || 'No response.' }]);
    } catch {
      setChatMessages((prev) => [...prev, { role: 'agent' as const, content: 'Error contacting agent.' }]);
    } finally {
      setChatLoading(false);
    }
  };

  return (
    <main className="container mx-auto p-4 space-y-8">
      <VideoUploader onVideoSubmit={handleVideoSubmit} />
      
      {error && (
        <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded relative" role="alert">
          <strong className="font-bold">Error: </strong>
          <span className="block sm:inline">{error}</span>
        </div>
      )}
      
      {isLoading && (
        <div className="flex justify-center items-center">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-gray-900"></div>
          <span className="ml-2">Analyzing video...</span>
        </div>
      )}
      
      {videoId && !isLoading && !error && (
        <div key={videoId} className="flex flex-col items-center gap-8">
          {/* YouTube Video Embed */}
          <div className="w-full aspect-video max-w-2xl mx-auto mb-4 rounded-lg overflow-hidden shadow">
            <YouTubePlayer
              ref={playerRef}
              videoId={videoId}
              onTimeUpdate={setCurrentTime}
            />
          </div>
          {/* Tabs Card */}
          <div className="w-full max-w-2xl bg-white rounded-lg shadow p-4">
            <Tabs defaultValue="transcript" className="w-full">
              <TabsList className="mb-4">
                <TabsTrigger value="transcript">Transcript</TabsTrigger>
                <TabsTrigger value="chapters">Chapters</TabsTrigger>
                <TabsTrigger value="analysis">Analysis</TabsTrigger>
              </TabsList>
              <TabsContent value="transcript">
                <TranscriptPanel
                  transcript={transcript}
                  currentTime={currentTime}
                  onTimeClick={handleTranscriptTimeClick}
                />
              </TabsContent>
              <TabsContent value="chapters">
                <div className="space-y-4">
                  {chapters.map((chapter, index) => (
                    <div
                      key={index}
                      className="p-4 border rounded-lg bg-gray-50 cursor-pointer hover:bg-primary/10"
                      onClick={() => handleChapterClick(chapter.start_time)}
                    >
                      <h3 className="font-bold">{chapter.title}</h3>
                      <p className="text-sm text-gray-600">
                        {Math.floor(chapter.start_time / 60)}:{(chapter.start_time % 60).toString().padStart(2, '0')} - 
                        {Math.floor(chapter.end_time / 60)}:{(chapter.end_time % 60).toString().padStart(2, '0')}
                      </p>
                      <p className="mt-2">{chapter.summary}</p>
                    </div>
                  ))}
                </div>
              </TabsContent>
              <TabsContent value="analysis">
                <div className="p-4 border rounded-lg bg-gray-50">
                  <h3 className="font-bold mb-2">Video Analysis</h3>
                  <p className="whitespace-pre-wrap mb-4">{ragResponse}</p>
                  {/* Chat UI */}
                  <div className="border rounded-lg bg-white p-4 max-h-96 overflow-y-auto mb-4" style={{ minHeight: 200 }}>
                    {chatMessages.length === 0 && <div className="text-gray-400">Ask anything about this video...</div>}
                    {chatMessages.map((msg, idx) => (
                      <div key={idx} className={`mb-2 flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                        <div className={`px-3 py-2 rounded-lg ${msg.role === 'user' ? 'bg-blue-100 text-right' : 'bg-gray-200 text-left'}`}>{msg.content}</div>
                      </div>
                    ))}
                    {chatLoading && <div className="text-gray-400">Agent is typing...</div>}
                  </div>
                  <div className="flex gap-2">
                    <input
                      className="flex-1 border rounded px-3 py-2"
                      type="text"
                      placeholder="Ask about the video..."
                      value={chatInput}
                      onChange={e => setChatInput(e.target.value)}
                      onKeyDown={e => { if (e.key === 'Enter') handleSendChat(); }}
                      disabled={chatLoading}
                    />
                    <button
                      className="bg-blue-600 text-white px-4 py-2 rounded disabled:opacity-50"
                      onClick={handleSendChat}
                      disabled={chatLoading || !chatInput.trim()}
                    >Send</button>
                  </div>
                </div>
              </TabsContent>
            </Tabs>
          </div>
        </div>
      )}
    </main>
  );
}
