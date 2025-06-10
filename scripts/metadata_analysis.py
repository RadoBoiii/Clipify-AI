#!/usr/bin/env python3
"""
Enhanced Metadata-Only Video Analysis with Supadata Transcript API
Uses external API for better transcript quality
"""

import sys
import json
import asyncio
import os
import time
import re
import aiohttp
from typing import Dict, List, Optional, Any

# Core dependencies
import yt_dlp
import openai
from dotenv import load_dotenv

class EnhancedMetadataAnalyzer:
    def __init__(self):
        # Initialize OpenAI client
        load_dotenv()
        self.openai_client = openai.AsyncOpenAI(
            api_key=os.getenv('OPENAI_API_KEY')
        )
        print(os.getenv('SUPADATA_API_KEY'))
        
        # Supadata API configuration
        # print("🔍 Checking environment variables...", file=sys.stderr)
        # print(f"Environment variables available: {list(os.environ.keys())}", file=sys.stderr)
        self.supadata_api_key = os.getenv('SUPADATA_API_KEY')  # Use the same API key as OpenAI
        print(f"SUPADATA_API_KEY present: {bool(self.supadata_api_key)}", file=sys.stderr)
        if not self.supadata_api_key:
            raise ValueError("SUPADATA_API_KEY environment variable is not set")
        self.supadata_base_url = 'https://api.supadata.ai/v1/youtube'
        
        # yt-dlp configuration - METADATA ONLY
        self.ydl_opts = {
            'quiet': True,
            'no_warnings': True,
            'extract_flat': False,
            'skip_download': True,
            'writesubtitles': False,
            'writeautomaticsub': False,
            'writeinfojson': False,
            'writethumbnail': False,
            'progress': False,
            'noprogress': True,
        }

    def clean_text_for_json(self, text: str) -> str:
        """Clean text to ensure it's JSON-safe"""
        if not text:
            return ""
        
        # Remove problematic characters
        text = text.replace('\u200b', '').replace('\u200c', '').replace('\u200d', '')
        text = text.replace('\u200e', '').replace('\u200f', '').replace('…', '...')
        
        # Remove emojis and keep basic characters
        text = re.sub(r'[^\x00-\x7F\u00A0-\u024F\u1E00-\u1EFF\u2000-\u206F\u2070-\u209F\u20A0-\u20CF\u2100-\u214F]', '', text)
        
        return text.strip()

    async def analyze_video_enhanced(self, youtube_url: str) -> Dict[str, Any]:
        """
        Enhanced video analysis using Supadata API for transcripts
        """
        start_time = time.time()
        
        try:
            print("🚀 Starting enhanced metadata analysis...", file=sys.stderr)
            
            # Extract video ID
            video_id = self.extract_video_id(youtube_url)
            if not video_id:
                raise ValueError("Invalid YouTube URL")
            
            print(f"📹 Video ID: {video_id}", file=sys.stderr)
            
            # Run metadata and transcript extraction in parallel
            print("⚡ Running parallel extraction...", file=sys.stderr)
            metadata_task = self.get_metadata_only(youtube_url)
            transcript_task = self.get_supadata_transcript(youtube_url)
            
            metadata, transcript = await asyncio.gather(metadata_task, transcript_task)
            
            print(f"✅ Metadata: {metadata['title'][:50]}...", file=sys.stderr)
            print(f"✅ Transcript: {len(transcript)} segments", file=sys.stderr)
            
            # Create intelligent chapters
            print("🧠 Creating intelligent chapters based on content...", file=sys.stderr)
            chapters = await self.create_smart_chapters(transcript, metadata)
            print(f"✅ Intelligent chapters: {len(chapters)}", file=sys.stderr)
            
            # Prepare final result
            result = {
                'success': True,
                'video_id': video_id,
                'metadata': metadata,
                'transcript': transcript,
                'chapters': chapters,
                'keyFrames': [],  # Not available without video download
                'processing_time': time.time() - start_time,
                'analysis_method': 'enhanced_metadata_supadata',
                'stats': {
                    'transcript_segments': len(transcript),
                    'chapters_generated': len(chapters),
                    'key_frames_extracted': 0,
                    'transcript_source': 'supadata_api',
                    'chapter_method': 'smart_content_analysis',
                    'video_downloaded': False
                }
            }
            
            print(f"🎉 Analysis complete in {result['processing_time']:.1f}s", file=sys.stderr)
            return result
            
        except Exception as e:
            error_result = {
                'success': False,
                'error': str(e),
                'video_id': video_id if 'video_id' in locals() else '',
                'processing_time': time.time() - start_time,
                'analysis_method': 'enhanced_metadata_supadata'
            }
            print(f"❌ Analysis failed: {e}", file=sys.stderr)
            return error_result

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

    async def get_metadata_only(self, url: str) -> Dict[str, Any]:
        """Extract metadata without any downloads"""
        loop = asyncio.get_event_loop()
        
        def _extract_metadata():
            try:
                print("🔍 Extracting video metadata...", file=sys.stderr)
                
                with yt_dlp.YoutubeDL(self.ydl_opts) as ydl:
                    info = ydl.extract_info(url, download=False)
                    
                    # Clean and extract relevant data
                    description = self.clean_text_for_json(info.get('description', '') or '')
                    
                    metadata = {
                        'id': info.get('id', ''),
                        'title': self.clean_text_for_json(info.get('title', 'Unknown Title')),
                        'author': self.clean_text_for_json(info.get('uploader', 'Unknown Author')),
                        'channel': self.clean_text_for_json(info.get('channel', '')),
                        'duration': info.get('duration', 0),
                        'view_count': info.get('view_count', 0),
                        'like_count': info.get('like_count', 0),
                        'upload_date': info.get('upload_date', ''),
                        'description': description,
                        'thumbnail': info.get('thumbnail', ''),
                        'tags': [self.clean_text_for_json(tag) for tag in (info.get('tags', []) or [])[:15]],
                        'category': info.get('category', ''),
                        'webpage_url': info.get('webpage_url', ''),
                    }
                    
                    print(f"📊 Metadata extracted: {metadata['title']}", file=sys.stderr)
                    return metadata
                    
            except Exception as e:
                print(f"Metadata extraction error: {e}", file=sys.stderr)
                # Return minimal metadata if extraction fails
                return {
                    'id': self.extract_video_id(url) or '',
                    'title': 'Video Analysis',
                    'author': 'Unknown',
                    'channel': '',
                    'duration': 0,
                    'view_count': 0,
                    'like_count': 0,
                    'upload_date': '',
                    'description': '',
                    'thumbnail': '',
                    'tags': [],
                    'category': '',
                    'webpage_url': url,
                }
        
        return await loop.run_in_executor(None, _extract_metadata)

    async def get_supadata_transcript(self, youtube_url: str) -> List[Dict[str, Any]]:
        """Get transcript using Supadata API"""
        try:
            print("📜 Fetching transcript via Supadata API...", file=sys.stderr)
            
            # Prepare API request
            headers = {
                'x-api-key': self.supadata_api_key,
                'Content-Type': 'application/json'
            }
            
            # URL encode the YouTube URL
            import urllib.parse
            encoded_url = urllib.parse.quote(youtube_url, safe='')
            api_url = f"{self.supadata_base_url}/transcript?url={encoded_url}"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(api_url, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        print("successfully got the transcript")
                        
                        # Parse Supadata response format
                        return self.parse_supadata_transcript(data)
                    else:
                        error_text = await response.text()
                        print(f"Supadata API error {response.status}: {error_text}", file=sys.stderr)
                        return []
                        
        except Exception as e:
            print(f"Supadata transcript error: {e}", file=sys.stderr)
            return []

    def parse_supadata_transcript(self, api_response: Dict) -> List[Dict[str, Any]]:
        """
        Parse Supadata API response into our transcript format
        
        Supadata format:
        {
            "lang": "en",
            "availableLangs": ["en"],
            "content": [
                {
                    "text": "Hello everybody. Welcome to my Premier",
                    "duration": 3761,
                    "offset": 719,
                    "lang": "en"
                }
            ]
        }
        """
        transcript_segments = []
        
        try:
            content = api_response.get('content', [])
            
            for segment in content:
                text = segment.get('text', '').strip()
                if not text:
                    continue
                
                # Convert milliseconds to seconds
                start_time = segment.get('offset', 0) / 1000.0
                duration_ms = segment.get('duration', 0)
                end_time = start_time + (duration_ms / 1000.0)
                
                # Clean the text
                clean_text = self.clean_text_for_json(text)
                
                if clean_text:
                    # Format exactly like fast_video_analysis.py
                    transcript_segments.append({
                        'text': clean_text,
                        'start': start_time,
                        'end': end_time,
                        'confidence': 0.95,  # Supadata typically has high accuracy
                        'duration': duration_ms / 1000.0,  # Add duration in seconds
                        'source': 'supadata_api',
                        'language': segment.get('lang', 'en'),
                        'is_generated': False  # Supadata provides high-quality transcripts
                    })
            
            print(f"📝 Parsed {len(transcript_segments)} transcript segments", file=sys.stderr)
            
            # Sort segments by start time to ensure proper order
            transcript_segments.sort(key=lambda x: x['start'])
            
            return transcript_segments
            
        except Exception as e:
            print(f"Error parsing Supadata transcript: {e}", file=sys.stderr)
            import traceback
            print(f"Traceback: {traceback.format_exc()}", file=sys.stderr)
            return []

    async def create_smart_chapters(self, transcript: List[Dict], metadata: Dict) -> List[Dict[str, Any]]:
        """Create chapters using multiple smart strategies"""
        
        # Strategy 1: Parse from description timestamps
        description_chapters = self.parse_description_chapters(metadata.get('description', ''))
        if description_chapters:
            print(f"📚 Found {len(description_chapters)} chapters in description", file=sys.stderr)
            return description_chapters
        
        # Strategy 2: Advanced LLM-based chapter creation using transcript content
        if transcript and len(transcript) > 10:
            content_chapters = await self.create_content_based_chapters(transcript, metadata)
            if content_chapters:
                print(f"🤖 Created {len(content_chapters)} content-based chapters", file=sys.stderr)
                return content_chapters
        
        # Strategy 3: Simple time-based chapters
        time_chapters = self.create_time_chapters(metadata.get('duration', 0))
        print(f"⏰ Created {len(time_chapters)} time-based chapters", file=sys.stderr)
        return time_chapters

    def parse_description_chapters(self, description: str) -> List[Dict[str, Any]]:
        """Parse chapter timestamps from video description"""
        chapters = []
        
        # Look for patterns like "0:00 Introduction" or "1:23 - Chapter Title"
        patterns = [
            r'(\d{1,2}:\d{2}(?::\d{2})?)\s*[-–]?\s*(.+?)(?=\n|$|\d{1,2}:\d{2})',
            r'(\d{1,2}:\d{2}(?::\d{2})?)\s+(.+?)(?=\n|$|\d{1,2}:\d{2})',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, description, re.MULTILINE | re.IGNORECASE)
            if len(matches) >= 2:  # Need at least 2 chapters to be valid
                for i, (timestamp, title) in enumerate(matches):
                    start_seconds = self.timestamp_to_seconds(timestamp)
                    # Calculate end time (next chapter start or end of video)
                    end_seconds = (
                        self.timestamp_to_seconds(matches[i + 1][0]) 
                        if i + 1 < len(matches) 
                        else None
                    )
                    
                    clean_title = self.clean_text_for_json(title.strip())
                    if clean_title and len(clean_title) > 3:  # Valid title
                        chapters.append({
                            'id': f'desc_chapter_{i}',
                            'title': clean_title[:80],
                            'start_time': start_seconds,
                            'end_time': end_seconds,
                            'summary': f'Chapter covering: {clean_title}'[:200],
                            'key_topics': self.extract_keywords(clean_title),
                            'word_count': len(clean_title.split()),
                            'main_topic': clean_title[:100],
                            'source': 'description_timestamps'
                        })
                
                if chapters:
                    return chapters
        
        return []

    async def create_content_based_chapters(self, transcript: List[Dict], metadata: Dict) -> List[Dict[str, Any]]:
        """Create chapters using advanced content analysis"""
        try:
            # Group transcript into larger chunks for better analysis
            chunk_size = max(20, len(transcript) // 8)  # Aim for 6-8 chapters max
            chunks = []
            
            for i in range(0, len(transcript), chunk_size):
                chunk = transcript[i:i + chunk_size]
                chunk_text = ' '.join([seg['text'] for seg in chunk])
                chunk_start = chunk[0]['start']
                chunk_end = chunk[-1]['end']
                
                chunks.append({
                    'text': chunk_text,
                    'start': chunk_start,
                    'end': chunk_end,
                    'segment_count': len(chunk)
                })
            
            # Create prompt for LLM analysis
            duration_minutes = metadata.get('duration', 0) / 60
            
            prompt = f"""Analyze this {duration_minutes:.1f}-minute video transcript and create logical chapters based on natural topic changes and content flow.

Video: "{metadata.get('title', 'Unknown')}" by {metadata.get('author', 'Unknown')}

Transcript chunks with timestamps:
"""
            
            for i, chunk in enumerate(chunks[:6]):  # Limit to first 6 chunks
                start_min = int(chunk['start'] // 60)
                start_sec = int(chunk['start'] % 60)
                prompt += f"\n[{start_min}:{start_sec:02d}] {chunk['text'][:300]}..."
            
            prompt += """

Create chapters in this exact JSON format:
Use the following as just an example of the format you should try to follow. It doesn't have to be exactly this timestamp but based on the duration of the particular chapter.
{
  "chapters": [
    {
      "title": "Engaging Chapter Title",
      "start_seconds": 0,
      "end_seconds": 180,
      "summary": "Brief description of this section's content",
      "main_topic": "Key theme or subject"
    }
  ]
}

Make chapter titles engaging and descriptive. Ensure chapters flow logically and cover the full video duration."""

            response = await self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert at analyzing video content and creating logical chapter divisions. Always respond with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                temperature=0.3
            )
            
            content = response.choices[0].message.content.strip()
            
            # Clean JSON from markdown
            if content.startswith('```'):
                content = content.split('```')[1]
                if content.startswith('json'):
                    content = content[4:]
            
            result = json.loads(content)
            chapters_data = result.get('chapters', [])
            
            # Convert to our format with content-based word counts
            formatted_chapters = []
            for i, chapter in enumerate(chapters_data):
                start_time = float(chapter.get('start_seconds', i * 60))
                end_time = float(chapter.get('end_seconds', (i + 1) * 60))
                
                # Calculate actual word count from transcript
                chapter_text = self.get_transcript_text_for_timerange(transcript, start_time, end_time)
                word_count = len(chapter_text.split()) if chapter_text else 0
                
                formatted_chapters.append({
                    'id': f'content_chapter_{i}',
                    'title': self.clean_text_for_json(chapter.get('title', f'Chapter {i+1}'))[:80],
                    'start_time': start_time,
                    'end_time': end_time,
                    'summary': self.clean_text_for_json(chapter.get('summary', ''))[:200],
                    'key_topics': self.extract_keywords(chapter.get('main_topic', '') + ' ' + chapter.get('title', '')),
                    'word_count': word_count,
                    'main_topic': self.clean_text_for_json(chapter.get('main_topic', ''))[:100],
                    'source': 'content_analysis'
                })
            
            return formatted_chapters
            
        except Exception as e:
            print(f"Content-based chapter creation error: {e}", file=sys.stderr)
            return []

    def get_transcript_text_for_timerange(self, transcript: List[Dict], start_time: float, end_time: float) -> str:
        """Extract transcript text for a specific time range"""
        text_parts = []
        for segment in transcript:
            if segment['start'] >= start_time and segment['end'] <= end_time:
                text_parts.append(segment['text'])
        return ' '.join(text_parts)

    def timestamp_to_seconds(self, timestamp: str) -> float:
        """Convert timestamp string to seconds"""
        try:
            parts = timestamp.split(':')
            if len(parts) == 2:  # mm:ss
                return int(parts[0]) * 60 + int(parts[1])
            elif len(parts) == 3:  # hh:mm:ss
                return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
            return 0.0
        except:
            return 0.0

    def extract_keywords(self, text: str) -> List[str]:
        """Extract simple keywords from text"""
        # Simple keyword extraction
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        # Remove common words
        common_words = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 'how', 'its', 'may', 'new', 'now', 'old', 'see', 'two', 'who', 'boy', 'did', 'man', 'men', 'say', 'she', 'too', 'use'}
        unique_words = [word for word in set(words) if word not in common_words]
        return unique_words[:5]  # Return up to 5 unique keywords

    def create_time_chapters(self, duration: float) -> List[Dict[str, Any]]:
        """Create simple time-based chapters as fallback"""
        if duration <= 0:
            return []
        
        # Determine number of chapters based on duration
        if duration < 180:  # < 3 minutes
            num_chapters = 2
        elif duration < 600:  # < 10 minutes
            num_chapters = 3
        elif duration < 1800:  # < 30 minutes
            num_chapters = 5
        else:
            num_chapters = 6
        
        chapter_length = duration / num_chapters
        chapters = []
        
        for i in range(num_chapters):
            start_time = i * chapter_length
            end_time = min((i + 1) * chapter_length, duration)
            
            chapters.append({
                'id': f'time_chapter_{i}',
                'title': f'Part {i + 1}',
                'start_time': start_time,
                'end_time': end_time,
                'summary': f'Content from {self.seconds_to_timestamp(start_time)} to {self.seconds_to_timestamp(end_time)}',
                'key_topics': [],
                'word_count': 0,
                'main_topic': f'Time segment {i + 1}',
                'source': 'time_based'
            })
        
        return chapters

    def seconds_to_timestamp(self, seconds: float) -> str:
        """Convert seconds to MM:SS format"""
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}:{secs:02d}"

async def main():
    """Main entry point"""
    if len(sys.argv) != 2:
        print(json.dumps({
            'success': False,
            'error': 'Usage: python enhanced_metadata_analysis.py <youtube_url>'
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
        analyzer = EnhancedMetadataAnalyzer()
        result = await analyzer.analyze_video_enhanced(youtube_url)
        
        # Output clean JSON
        json_str = json.dumps(result, ensure_ascii=True, separators=(',', ':'))
        print(json_str)
        
        sys.exit(0 if result.get('success') else 1)
        
    except Exception as e:
        print(json.dumps({
            'success': False,
            'error': f'Analysis error: {str(e)}'
        }, ensure_ascii=True))
        sys.exit(1)

if __name__ == '__main__':
    asyncio.run(main()) 