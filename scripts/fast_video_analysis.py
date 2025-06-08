#!/usr/bin/env python3
"""
Complete Fast Video Analysis Script with Intelligent Chapter Generation
Optimized for speed with LLM-powered content analysis
"""

import sys
import json
import asyncio
import os
import tempfile
import time
import re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Any

# Core dependencies
import yt_dlp
import whisper
from youtube_transcript_api import YouTubeTranscriptApi
from moviepy.editor import VideoFileClip
import numpy as np
from PIL import Image
import openai

class FastVideoAnalyzer:
    def __init__(self):
        self.temp_dir = tempfile.mkdtemp()
        self.whisper_model = None  # Load only if needed
        
        # Initialize OpenAI client
        self.openai_client = openai.AsyncOpenAI(
            api_key=os.getenv('OPENAI_API_KEY')
        )
        
        # Optimized yt-dlp configuration
        self.ydl_opts_base = {
            'quiet': True,
            'no_warnings': True,
            'extract_flat': False,
            'writesubtitles': False,
            'writeautomaticsub': False,
            'progress': False,  # Disable progress output
            'noprogress': True,  # Disable progress output (alternative)
            'logtostderr': True,  # Redirect all output to stderr
        }

    def clean_text_for_json(self, text: str) -> str:
        """Clean text to ensure it's JSON-safe"""
        if not text:
            return ""
        
        # Remove or replace problematic characters
        text = text.replace('\u200b', '')  # Remove zero-width space
        text = text.replace('\u200c', '')  # Remove zero-width non-joiner
        text = text.replace('\u200d', '')  # Remove zero-width joiner
        text = text.replace('\u200e', '')  # Remove left-to-right mark
        text = text.replace('\u200f', '')  # Remove right-to-left mark
        text = text.replace('…', '...')    # Replace ellipsis
        
        # Remove emojis and other problematic Unicode characters
        # Keep basic Latin, Latin Extended, and common punctuation
        text = re.sub(r'[^\x00-\x7F\u00A0-\u024F\u1E00-\u1EFF\u2000-\u206F\u2070-\u209F\u20A0-\u20CF\u2100-\u214F]', '', text)
        
        return text.strip()

    async def analyze_video(self, youtube_url: str) -> Dict[str, Any]:
        """
        Complete video analysis pipeline with intelligent chapters
        Returns: metadata, transcript, chapters, and key frames
        """
        start_time = time.time()
        
        try:
            print("🚀 Starting video analysis...", file=sys.stderr)
            
            # Extract video ID
            video_id = self.extract_video_id(youtube_url)
            if not video_id:
                raise ValueError("Invalid YouTube URL")
            
            print(f"📹 Video ID: {video_id}", file=sys.stderr)
            
            # Step 1: Run parallel tasks for speed
            print("⚡ Running parallel extraction...", file=sys.stderr)
            metadata_task = self.get_metadata_fast(youtube_url)
            transcript_task = self.get_transcript_fast(video_id)
            download_task = self.download_video_optimized(youtube_url)
            
            metadata, transcript, video_path = await asyncio.gather(
                metadata_task, transcript_task, download_task
            )
            
            print(f"✅ Metadata: {metadata['title'][:50]}...", file=sys.stderr)
            print(f"✅ Transcript: {len(transcript)} segments", file=sys.stderr)
            print(f"✅ Video downloaded: {bool(video_path)}", file=sys.stderr)
            
            # Step 2: Fallback to Whisper if no transcript
            if not transcript and video_path:
                print("🔄 Falling back to Whisper transcription...", file=sys.stderr)
                transcript = await self.transcribe_with_whisper(video_path)
                print(f"✅ Whisper transcript: {len(transcript)} segments", file=sys.stderr)
            
            # Step 3: Extract key frames (parallel)
            key_frames = []
            if video_path and os.path.exists(video_path):
                print("🖼️ Extracting key frames...", file=sys.stderr)
                key_frames = await self.extract_key_frames(video_path)
                print(f"✅ Key frames: {len(key_frames)}", file=sys.stderr)
            
            # Step 4: Intelligent chapter generation with LLM
            print("🧠 Creating intelligent chapters based on content...", file=sys.stderr)
            chapters = await self.analyze_content_and_create_chapters(transcript, metadata)
            print(f"✅ Intelligent chapters: {len(chapters)}", file=sys.stderr)
            
            # Step 5: Prepare final result
            result = {
                'success': True,
                'video_id': video_id,
                'metadata': metadata,
                'transcript': transcript,
                'chapters': chapters,
                'keyFrames': key_frames,
                'processing_time': time.time() - start_time,
                'stats': {
                    'transcript_segments': len(transcript),
                    'chapters_generated': len(chapters),
                    'key_frames_extracted': len(key_frames),
                    'transcript_source': 'youtube_api' if transcript and not self.whisper_model else 'whisper',
                    'chapter_method': 'llm_content_analysis'
                }
            }
            
            print(f"🎉 Analysis complete in {result['processing_time']:.1f}s", file=sys.stderr)
            return result
            
        except Exception as e:
            error_result = {
                'success': False,
                'error': str(e),
                'video_id': video_id if 'video_id' in locals() else '',
                'processing_time': time.time() - start_time
            }
            print(f"❌ Analysis failed: {e}", file=sys.stderr)
            return error_result
            
        finally:
            self.cleanup()

    def extract_video_id(self, url: str) -> Optional[str]:
        """Extract YouTube video ID from various URL formats"""
        patterns = [
            r'(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/)([^&\n?#]+)',
            r'youtube\.com\/watch\?.*v=([^&\n?#]+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        return None

    async def get_metadata_fast(self, url: str) -> Dict[str, Any]:
        """Extract metadata without downloading video"""
        loop = asyncio.get_event_loop()
        
        def _extract_metadata():
            try:
                ydl_opts = {
                    **self.ydl_opts_base,
                    'skip_download': True,
                }
                
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(url, download=False)
                    
                    # Clean description text
                    description = info.get('description', '') or ''
                    description = self.clean_text_for_json(description)
                    
                    return {
                        'id': info.get('id', ''),
                        'title': self.clean_text_for_json(info.get('title', 'Unknown Title')),
                        'author': self.clean_text_for_json(info.get('uploader', 'Unknown Author')),
                        'duration': info.get('duration', 0),
                        'view_count': info.get('view_count', 0),
                        'upload_date': info.get('upload_date', ''),
                        'description': description,
                        'thumbnail': info.get('thumbnail', ''),
                        'tags': [self.clean_text_for_json(tag) for tag in (info.get('tags', []) or [])[:10]],
                    }
            except Exception as e:
                print(f"Metadata extraction error: {e}", file=sys.stderr)
                return {
                    'id': self.extract_video_id(url) or '',
                    'title': 'Unknown Title',
                    'author': 'Unknown Author',
                    'duration': 0,
                    'view_count': 0,
                    'upload_date': '',
                    'description': '',
                    'thumbnail': '',
                    'tags': [],
                }
        
        return await loop.run_in_executor(None, _extract_metadata)

    async def get_transcript_fast(self, video_id: str) -> List[Dict[str, Any]]:
        """Get transcript using YouTube API (fastest method)"""
        loop = asyncio.get_event_loop()
        
        def _get_transcript():
            try:
                transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
                
                # Try different transcript types in order of preference
                transcript = None
                for lang_code in ['en', 'en-US', 'en-GB']:
                    try:
                        transcript = transcript_list.find_generated_transcript([lang_code])
                        break
                    except:
                        try:
                            transcript = transcript_list.find_transcript([lang_code])
                            break
                        except:
                            continue
                
                if not transcript:
                    # Try any available transcript
                    available = list(transcript_list)
                    if available:
                        transcript = available[0]
                
                if transcript:
                    segments = transcript.fetch()
                    return [
                        {
                            'text': self.clean_text_for_json(segment['text'].strip()),
                            'start': segment['start'],
                            'end': segment['start'] + segment['duration'],
                            'confidence': 1.0
                        }
                        for segment in segments
                        if segment['text'].strip()
                    ]
                
                return []
                
            except Exception as e:
                print(f"YouTube transcript error: {e}", file=sys.stderr)
                return []
        
        return await loop.run_in_executor(None, _get_transcript)

    async def download_video_optimized(self, url: str) -> Optional[str]:
        """Download video with speed optimizations"""
        loop = asyncio.get_event_loop()
        
        def _download():
            try:
                video_path = os.path.join(self.temp_dir, 'video.%(ext)s')
                
                ydl_opts = {
                    **self.ydl_opts_base,
                    'format': 'worst[height<=480]/worst[height<=720]/worst',
                    'outtmpl': video_path,
                    'merge_output_format': 'mp4',
                    'progress': False,  # Disable progress output
                    'noprogress': True,  # Disable progress output (alternative)
                    'logtostderr': True,  # Redirect all output to stderr
                    'verbose': True,  # Add verbose logging
                }
                
                print(f"Attempting to download video to: {video_path}", file=sys.stderr)
                print(f"Using yt-dlp options: {ydl_opts}", file=sys.stderr)
                
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    try:
                        ydl.download([url])
                    except Exception as e:
                        print(f"yt-dlp download error: {str(e)}", file=sys.stderr)
                        raise
                
                # Find downloaded file
                print(f"Checking for downloaded file in: {self.temp_dir}", file=sys.stderr)
                for file in os.listdir(self.temp_dir):
                    print(f"Found file: {file}", file=sys.stderr)
                    if file.startswith('video.') and file.endswith(('.mp4', '.webm', '.mkv')):
                        full_path = os.path.join(self.temp_dir, file)
                        print(f"Found video file: {full_path}", file=sys.stderr)
                        return full_path
                
                print("No video file found after download attempt", file=sys.stderr)
                return None
                
            except Exception as e:
                print(f"Download error details: {str(e)}", file=sys.stderr)
                import traceback
                print(f"Traceback: {traceback.format_exc()}", file=sys.stderr)
                return None
        
        return await loop.run_in_executor(None, _download)

    async def transcribe_with_whisper(self, video_path: str) -> List[Dict[str, Any]]:
        """Fallback transcription using Whisper"""
        loop = asyncio.get_event_loop()
        
        def _transcribe():
            try:
                # Load model only when needed
                if not self.whisper_model:
                    print("Loading Whisper model...", file=sys.stderr)
                    self.whisper_model = whisper.load_model("base")
                
                print("Transcribing with Whisper...", file=sys.stderr)
                result = self.whisper_model.transcribe(
                    video_path,
                    language='en',
                    fp16=False
                )
                
                return [
                    {
                        'text': self.clean_text_for_json(segment['text'].strip()),
                        'start': segment['start'],
                        'end': segment['end'],
                        'confidence': 0.8
                    }
                    for segment in result['segments']
                    if segment['text'].strip()
                ]
                
            except Exception as e:
                print(f"Whisper transcription error: {e}", file=sys.stderr)
                return []
        
        return await loop.run_in_executor(None, _transcribe)

    async def extract_key_frames(self, video_path: str) -> List[Dict[str, Any]]:
        """Extract key frames for visual search"""
        loop = asyncio.get_event_loop()
        
        def _extract():
            try:
                clip = VideoFileClip(video_path)
                duration = clip.duration
                
                # Smart frame interval based on video length
                if duration < 120:  # < 2 minutes
                    interval = 15
                elif duration < 600:  # < 10 minutes
                    interval = 30
                else:
                    interval = 60
                
                frame_times = np.arange(0, duration, interval)[:20]  # Max 20 frames
                key_frames = []
                
                for i, t in enumerate(frame_times):
                    try:
                        frame = clip.get_frame(t)
                        img = Image.fromarray(frame)
                        
                        # Resize for web
                        img.thumbnail((640, 360), Image.Resampling.LANCZOS)
                        
                        frame_filename = f'frame_{i}_{int(t)}.jpg'
                        frame_path = os.path.join(self.temp_dir, frame_filename)
                        img.save(frame_path, quality=75, optimize=True)
                        
                        key_frames.append({
                            'timestamp': float(t),
                            'filename': frame_filename,
                            'description': f'Frame at {int(t//60)}:{int(t%60):02d}',
                            'path': frame_path
                        })
                        
                    except Exception as e:
                        print(f"Frame extraction error at {t}s: {e}", file=sys.stderr)
                
                clip.close()
                return key_frames
                
            except Exception as e:
                print(f"Key frame extraction error: {e}", file=sys.stderr)
                return []
        
        return await loop.run_in_executor(None, _extract)

    async def analyze_content_and_create_chapters(self, transcript: List[Dict], metadata: Dict) -> List[Dict[str, Any]]:
        """Analyze video content with LLM to create natural chapter breaks"""
        if not transcript:
            return []
        
        print("🧠 Analyzing content structure with LLM...", file=sys.stderr)
        
        # Step 1: Create a condensed transcript with timestamps
        condensed_transcript = self.create_condensed_transcript(transcript)
        
        # Step 2: Use LLM to identify natural chapter breaks
        chapter_structure = await self.identify_chapter_breaks_with_llm(condensed_transcript, metadata)
        
        # Step 3: Create detailed chapters based on LLM analysis
        detailed_chapters = await self.create_detailed_chapters(chapter_structure, transcript, metadata)
        
        return detailed_chapters

    def create_condensed_transcript(self, transcript: List[Dict], max_length: int = 4000) -> str:
        """Create a condensed version of the transcript for LLM analysis"""
        
        # Combine transcript with timestamps
        full_text = ""
        for segment in transcript:
            timestamp = f"[{int(segment['start']//60)}:{int(segment['start']%60):02d}]"
            full_text += f"{timestamp} {segment['text']} "
        
        # If transcript is too long, create summary chunks
        if len(full_text) > max_length:
            # Take samples from beginning, middle, and end
            chunk_size = max_length // 4
            beginning = full_text[:chunk_size]
            middle_start = len(full_text) // 2 - chunk_size // 2
            middle = full_text[middle_start:middle_start + chunk_size]
            end = full_text[-chunk_size:]
            
            condensed = f"{beginning}\n\n[... MIDDLE SECTION ...]\n{middle}\n\n[... END SECTION ...]\n{end}"
        else:
            condensed = full_text
        
        return condensed

    async def identify_chapter_breaks_with_llm(self, condensed_transcript: str, metadata: Dict) -> List[Dict]:
        """Use LLM to identify natural topic transitions and chapter breaks"""
        
        video_context = f"""
Video Title: {metadata.get('title', 'Unknown')}
Author: {metadata.get('author', 'Unknown')}
Duration: {metadata.get('duration', 0)} seconds ({metadata.get('duration', 0)//60}:{metadata.get('duration', 0)%60:02d})
Description: {metadata.get('description', '')[:300]}
"""

        prompt = f"""{video_context}

Analyze this video transcript and identify natural chapter breaks based on topic changes, content shifts, and logical segments.

Transcript with timestamps:
{condensed_transcript}

Instructions:
1. Identify natural chapter breaks based on content topics. Analyze the transcript and the video description and title to identify the chapter breaks.
2. Each chapter should cover a distinct topic or theme. Make sure theres no overlap between chapters.
3. Look for transitions like "Now let's talk about...", "Moving on to...", topic changes, etc.
4. Chapters should be at least 30 seconds long
5. Create engaging, descriptive titles that capture the essence of each section

Respond in JSON format with an array of chapters:
{{
  "chapters": [
    {{
      "start_timestamp": "0:00",
      "end_timestamp": "2:15", 
      "title": "Engaging Chapter Title",
      "main_topic": "Brief description of what this section covers",
      "key_points": ["point1", "point2", "point3"]
    }},
    {{
      "start_timestamp": "2:15",
      "end_timestamp": "5:30",
      "title": "Next Chapter Title", 
      "main_topic": "Next section description",
      "key_points": ["point1", "point2"]
    }}
  ]
}}

Make sure timestamps don't overlap and cover the entire video duration."""

        try:
            response = await self.openai_client.chat.completions.create(
                model="gpt-4o-mini",  # Good for analysis tasks
                messages=[
                    {
                        "role": "system", 
                        "content": "You are an expert video content analyzer who creates logical chapter divisions based on content flow and topic changes. You always respond with valid JSON."
                    },
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1500,
                temperature=0.3  # Lower temperature for more consistent structure
            )
            
            content = response.choices[0].message.content.strip()
            
            # Remove markdown code blocks if present
            if content.startswith('```'):
                content = content.split('```')[1]
                if content.startswith('json'):
                    content = content[4:]
            
            result = json.loads(content)
            return result.get('chapters', [])
            
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}", file=sys.stderr)
            print(f"LLM response: {content[:200]}...", file=sys.stderr)
            # Fallback to time-based chapters
            return self.create_fallback_chapters(metadata.get('duration', 0))
            
        except Exception as e:
            print(f"LLM chapter analysis error: {e}", file=sys.stderr)
            return self.create_fallback_chapters(metadata.get('duration', 0))

    def timestamp_to_seconds(self, timestamp: str) -> float:
        """Convert timestamp string like '2:15' to seconds"""
        try:
            if ':' in timestamp:
                parts = timestamp.split(':')
                if len(parts) == 2:
                    minutes, seconds = parts
                    return int(minutes) * 60 + int(seconds)
                elif len(parts) == 3:
                    hours, minutes, seconds = parts
                    return int(hours) * 3600 + int(minutes) * 60 + int(seconds)
            return float(timestamp)
        except:
            return 0.0

    async def create_detailed_chapters(self, chapter_structure: List[Dict], transcript: List[Dict], metadata: Dict) -> List[Dict[str, Any]]:
        """Create detailed chapters with summaries based on LLM structure and full transcript"""
        
        if not chapter_structure:
            print("No chapter structure from LLM, using fallback", file=sys.stderr)
            return self.create_fallback_chapters(metadata.get('duration', 0))
        
        detailed_chapters = []
        
        for idx, chapter_info in enumerate(chapter_structure):
            try:
                # Convert timestamps to seconds
                start_time = self.timestamp_to_seconds(chapter_info.get('start_timestamp', '0:00'))
                end_time = self.timestamp_to_seconds(chapter_info.get('end_timestamp', '0:00'))
                
                # Extract transcript segments for this chapter
                chapter_segments = [
                    seg for seg in transcript 
                    if start_time <= seg['start'] < end_time
                ]
                
                chapter_text = ' '.join([seg['text'] for seg in chapter_segments])
                
                # Generate detailed summary for this chapter
                chapter_summary = await self.generate_chapter_summary(
                    chapter_text, 
                    chapter_info, 
                    metadata
                )
                
                detailed_chapters.append({
                    'id': f'chapter_{idx}',
                    'title': self.clean_text_for_json(chapter_info.get('title', f'Chapter {idx + 1}'))[:80],
                    'start_time': start_time,
                    'end_time': end_time,
                    'summary': chapter_summary.get('summary', chapter_info.get('main_topic', ''))[:200],
                    'key_topics': chapter_summary.get('key_topics', chapter_info.get('key_points', []))[:5],
                    'word_count': len(chapter_text.split()) if chapter_text else 0,
                    'main_topic': self.clean_text_for_json(chapter_info.get('main_topic', ''))[:100]
                })
                
                print(f"✅ Chapter {idx + 1}: {detailed_chapters[-1]['title']}", file=sys.stderr)
                
            except Exception as e:
                print(f"Error creating chapter {idx + 1}: {e}", file=sys.stderr)
                continue
        
        return detailed_chapters

    async def generate_chapter_summary(self, chapter_text: str, chapter_info: Dict, metadata: Dict) -> Dict[str, Any]:
        """Generate a detailed summary for a specific chapter"""
        
        if not chapter_text.strip():
            return {
                'summary': chapter_info.get('main_topic', 'No content available'),
                'key_topics': chapter_info.get('key_points', [])
            }
        
        prompt = f"""Analyze this chapter from "{metadata.get('title', 'Unknown Video')}" and create:

1. A concise summary (max 200 characters) of what's discussed
2. 3-5 key topics/themes covered

Chapter Title: {chapter_info.get('title', 'Unknown')}
Expected Topic: {chapter_info.get('main_topic', 'Unknown')}

Chapter Content:
{chapter_text[:1200]}

Respond in JSON format:
{{"summary": "Detailed summary here", "key_topics": ["topic1", "topic2", "topic3"]}}"""

        try:
            response = await self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system", 
                        "content": "You create concise, informative summaries that capture the key points discussed in video segments."
                    },
                    {"role": "user", "content": prompt}
                ],
                max_tokens=300,
                temperature=0.5
            )
            
            content = response.choices[0].message.content.strip()
            
            # Clean up markdown if present
            if content.startswith('```'):
                content = content.split('```')[1]
                if content.startswith('json'):
                    content = content[4:]
            
            result = json.loads(content)
            
            return {
                'summary': self.clean_text_for_json(result.get('summary', chapter_info.get('main_topic', '')))[:200],
                'key_topics': [self.clean_text_for_json(topic)[:30] for topic in result.get('key_topics', [])][:5]
            }
            
        except Exception as e:
            print(f"Summary generation error: {e}", file=sys.stderr)
            return {
                'summary': chapter_info.get('main_topic', 'Content analysis unavailable'),
                'key_topics': chapter_info.get('key_points', [])[:5]
            }

    def create_fallback_chapters(self, duration: float) -> List[Dict[str, Any]]:
        """Fallback to time-based chapters if LLM analysis fails"""
        if duration <= 0:
            return []
        
        # Create 4-6 chapters based on duration
        if duration < 300:  # < 5 minutes
            num_chapters = 3
        elif duration < 900:  # < 15 minutes
            num_chapters = 4
        elif duration < 1800:  # < 30 minutes
            num_chapters = 6
        else:
            num_chapters = 8
        
        chapter_length = duration / num_chapters
        chapters = []
        
        for i in range(num_chapters):
            start_time = i * chapter_length
            end_time = min((i + 1) * chapter_length, duration)
            
            chapters.append({
                'id': f'chapter_{i}',
                'title': f'Chapter {i + 1}',
                'start_time': start_time,
                'end_time': end_time,
                'summary': f'Content from {int(start_time//60)}:{int(start_time%60):02d} to {int(end_time//60)}:{int(end_time%60):02d}',
                'key_topics': [],
                'word_count': 0,
                'main_topic': 'Time-based chapter'
            })
        
        return chapters

    def cleanup(self):
        """Clean up temporary files"""
        try:
            import shutil
            if os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
        except Exception as e:
            print(f"Cleanup error: {e}", file=sys.stderr)

async def main():
    """Main entry point"""
    if len(sys.argv) != 2:
        print(json.dumps({
            'success': False,
            'error': 'Usage: python fast_video_analysis.py <youtube_url>'
        }, ensure_ascii=True))
        sys.exit(1)
    
    youtube_url = sys.argv[1]
    
    # Validate URL
    if 'youtube.com' not in youtube_url and 'youtu.be' not in youtube_url:
        print(json.dumps({
            'success': False,
            'error': 'Invalid YouTube URL'
        }, ensure_ascii=True))
        sys.exit(1)
    
    try:
        analyzer = FastVideoAnalyzer()
        result = await analyzer.analyze_video(youtube_url)
        
        # Output JSON result with proper encoding
        json_str = json.dumps(result, ensure_ascii=True, separators=(',', ':'))
        print(json_str)
        
        # Exit with appropriate code
        sys.exit(0 if result.get('success') else 1)
        
    except KeyboardInterrupt:
        print(json.dumps({
            'success': False,
            'error': 'Analysis interrupted by user'
        }, ensure_ascii=True))
        sys.exit(1)
    except Exception as e:
        print(json.dumps({
            'success': False,
            'error': f'Unexpected error: {str(e)}'
        }, ensure_ascii=True))
        sys.exit(1)

if __name__ == '__main__':
    asyncio.run(main()) 