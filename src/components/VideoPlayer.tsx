'use client';

import { useEffect, useRef } from 'react';

interface VideoPlayerProps {
  videoId: string;
  onTimeUpdate?: (time: number) => void;
}

export function VideoPlayer({ videoId, onTimeUpdate }: VideoPlayerProps) {
  const playerRef = useRef<HTMLIFrameElement>(null);

  useEffect(() => {
    // Load YouTube IFrame API
    const tag = document.createElement('script');
    tag.src = 'https://www.youtube.com/iframe_api';
    const firstScriptTag = document.getElementsByTagName('script')[0];
    firstScriptTag.parentNode?.insertBefore(tag, firstScriptTag);

    // Initialize player when API is ready
    window.onYouTubeIframeAPIReady = () => {
      new window.YT.Player(playerRef.current!, {
        videoId,
        playerVars: {
          autoplay: 0,
          controls: 1,
        },
        events: {
          onStateChange: (event: { data: number; target: { getCurrentTime: () => number } }) => {
            if (event.data === window.YT.PlayerState.PLAYING) {
              const interval = setInterval(() => {
                const currentTime = event.target.getCurrentTime();
                onTimeUpdate?.(currentTime);
              }, 1000);

              return () => clearInterval(interval);
            }
          },
        },
      });
    };
  }, [videoId, onTimeUpdate]);

  return (
    <div className="relative w-full aspect-video">
      <iframe
        ref={playerRef}
        className="absolute top-0 left-0 w-full h-full"
        allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
        allowFullScreen
      />
    </div>
  );
}