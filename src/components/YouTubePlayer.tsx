import React, { useEffect, useRef, useImperativeHandle, forwardRef } from 'react';

interface YouTubePlayerProps {
  videoId: string;
  onTimeUpdate?: (time: number) => void;
}

export interface YouTubePlayerHandle {
  seekTo: (seconds: number) => void;
}

const YouTubePlayer = forwardRef<YouTubePlayerHandle, YouTubePlayerProps>(({ videoId, onTimeUpdate }, ref) => {
  const playerRef = useRef<any>(null);
  const iframeRef = useRef<HTMLDivElement>(null);
  const intervalRef = useRef<NodeJS.Timeout | null>(null);

  useImperativeHandle(ref, () => ({
    seekTo: (seconds: number) => {
      if (playerRef.current) {
        playerRef.current.seekTo(seconds, true);
      }
    },
  }));

  useEffect(() => {
    // Load YouTube IFrame API if not already loaded
    if (!(window as any).YT) {
      const tag = document.createElement('script');
      tag.src = 'https://www.youtube.com/iframe_api';
      document.body.appendChild(tag);
    }

    (window as any).onYouTubeIframeAPIReady = () => {
      createPlayer();
    };

    // If already loaded
    if ((window as any).YT && (window as any).YT.Player) {
      createPlayer();
    }

    function createPlayer() {
      if (!iframeRef.current) return;
      playerRef.current = new (window as any).YT.Player(iframeRef.current, {
        height: '360',
        width: '640',
        videoId,
        events: {
          onReady: () => {},
          onStateChange: () => {},
        },
      });
    }

    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
      if (playerRef.current && playerRef.current.destroy) {
        playerRef.current.destroy();
      }
    };
    // eslint-disable-next-line
  }, [videoId]);

  // Poll current time for autoscroll
  useEffect(() => {
    if (!onTimeUpdate) return;
    intervalRef.current = setInterval(() => {
      if (playerRef.current && playerRef.current.getCurrentTime) {
        const time = playerRef.current.getCurrentTime();
        onTimeUpdate(time);
      }
    }, 500);
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, [onTimeUpdate]);

  return <div ref={iframeRef} className="w-full aspect-video rounded-lg overflow-hidden" />;
});

YouTubePlayer.displayName = 'YouTubePlayer';
export default YouTubePlayer; 