#!/usr/bin/env ts-node
/**
 * Enhanced Metadata-Only Video Analysis with Supadata Transcript API
 * Uses external API for better transcript quality
 */

import { spawn } from 'child_process';
import { config } from 'dotenv';
import OpenAI from 'openai';
import ytdl from 'ytdl-core';
import axios from 'axios';

// Load environment variables
config();

interface TranscriptSegment {
  text: string;
  start: number;
  end: number;
  confidence: number;
  duration: number;
  source: string;
  language: string;
  is_generated: boolean;
}

interface Chapter {
  id: string;
  title: string;
  start_time: number;
  end_time: number;
  summary: string;
  key_topics: string[];
  word_count: number;
  main_topic: string;
  source: string;
}

interface AnalysisResult {
  success: boolean;
  video_id: string;
  metadata: any;
  transcript: TranscriptSegment[];
  chapters: Chapter[];
  keyFrames: any[];
  processing_time: number;
  analysis_method: string;
  stats: {
    transcript_segments: number;
    chapters_generated: number;
    key_frames_extracted: number;
    transcript_source: string;
    chapter_method: string;
    video_downloaded: boolean;
  };
}

class EnhancedMetadataAnalyzer {
  private openai: OpenAI;
  private supadataApiKey: string;
  private supadataBaseUrl: string;

  constructor() {
    // Initialize OpenAI client
    this.openai = new OpenAI({
      apiKey: process.env.OPENAI_API_KEY
    });

    // Supadata API configuration
    this.supadataApiKey = process.env.SUPADATA_API_KEY || '';
    if (!this.supadataApiKey) {
      throw new Error('SUPADATA_API_KEY environment variable is not set');
    }
    this.supadataBaseUrl = 'https://api.supadata.ai/v1/youtube';
  }

  private cleanTextForJson(text: string): string {
    if (!text) return '';

    // Remove problematic characters
    text = text.replace(/[\u200b\u200c\u200d\u200e\u200f]/g, '')
               .replace(/…/g, '...');

    // Remove emojis and keep basic characters
    text = text.replace(/[^\x00-\x7F\u00A0-\u024F\u1E00-\u1EFF\u2000-\u206F\u2070-\u209F\u20A0-\u20CF\u2100-\u214F]/g, '');

    return text.trim();
  }

  private extractVideoId(url: string): string | null {
    const patterns = [
      /(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/)([^&\n?#]+)/,
      /youtube\.com\/watch\?.*v=([^&\n?#]+)/,
    ];

    for (const pattern of patterns) {
      const match = url.match(pattern);
      if (match) return match[1];
    }
    return null;
  }

  private async getMetadataOnly(url: string): Promise<any> {
    try {
      console.error('🔍 Extracting video metadata...');
      const info = await ytdl.getInfo(url);

      return {
        id: info.videoDetails.videoId,
        title: this.cleanTextForJson(info.videoDetails.title),
        author: this.cleanTextForJson(info.videoDetails.author.name),
        channel: this.cleanTextForJson(info.videoDetails.author.id),
        duration: parseInt(info.videoDetails.lengthSeconds),
        view_count: parseInt(info.videoDetails.viewCount),
        like_count: parseInt(String(info.videoDetails.likes || '0')),
        upload_date: info.videoDetails.uploadDate,
        description: this.cleanTextForJson(info.videoDetails.description || ''),
        thumbnail: info.videoDetails.thumbnails[0]?.url,
        tags: info.videoDetails.keywords?.slice(0, 15).map(tag => this.cleanTextForJson(tag)) || [],
        category: info.videoDetails.category,
        webpage_url: info.videoDetails.video_url,
      };
    } catch (error) {
      console.error('Metadata extraction error:', error);
      return {
        id: this.extractVideoId(url) || '',
        title: 'Video Analysis',
        author: 'Unknown',
        channel: '',
        duration: 0,
        view_count: 0,
        like_count: 0,
        upload_date: '',
        description: '',
        thumbnail: '',
        tags: [],
        category: '',
        webpage_url: url,
      };
    }
  }

  private async getSupadataTranscript(youtubeUrl: string): Promise<TranscriptSegment[]> {
    try {
      console.error('📜 Fetching transcript via Supadata API...');

      const encodedUrl = encodeURIComponent(youtubeUrl);
      const apiUrl = `${this.supadataBaseUrl}/transcript?url=${encodedUrl}`;

      const response = await axios.get(apiUrl, {
        headers: {
          'x-api-key': this.supadataApiKey,
          'Content-Type': 'application/json'
        }
      });

      if (response.status === 200) {
        console.error('Successfully got the transcript');
        return this.parseSupadataTranscript(response.data);
      } else {
        console.error(`Supadata API error ${response.status}: ${response.data}`);
        return [];
      }
    } catch (error) {
      console.error('Supadata transcript error:', error);
      return [];
    }
  }

  private parseSupadataTranscript(apiResponse: any): TranscriptSegment[] {
    const transcriptSegments: TranscriptSegment[] = [];

    try {
      const content = apiResponse.content || [];

      for (const segment of content) {
        const text = segment.text?.trim();
        if (!text) continue;

        const startTime = segment.offset / 1000.0;
        const durationMs = segment.duration;
        const endTime = startTime + (durationMs / 1000.0);

        const cleanText = this.cleanTextForJson(text);

        if (cleanText) {
          transcriptSegments.push({
            text: cleanText,
            start: startTime,
            end: endTime,
            confidence: 0.95,
            duration: durationMs / 1000.0,
            source: 'supadata_api',
            language: segment.lang || 'en',
            is_generated: false
          });
        }
      }

      console.error(`📝 Parsed ${transcriptSegments.length} transcript segments`);

      // Sort segments by start time
      transcriptSegments.sort((a, b) => a.start - b.start);

      return transcriptSegments;
    } catch (error) {
      console.error('Error parsing Supadata transcript:', error);
      return [];
    }
  }

  public async analyzeVideo(youtubeUrl: string): Promise<AnalysisResult> {
    const startTime = Date.now();

    try {
      console.error('🚀 Starting enhanced metadata analysis...');

      const videoId = this.extractVideoId(youtubeUrl);
      if (!videoId) {
        throw new Error('Invalid YouTube URL');
      }

      console.error(`📹 Video ID: ${videoId}`);

      // Run metadata and transcript extraction in parallel
      console.error('⚡ Running parallel extraction...');
      const [metadata, transcript] = await Promise.all([
        this.getMetadataOnly(youtubeUrl),
        this.getSupadataTranscript(youtubeUrl)
      ]);

      console.error(`✅ Metadata: ${metadata.title.substring(0, 50)}...`);
      console.error(`✅ Transcript: ${transcript.length} segments`);

      // Create intelligent chapters
      console.error('🧠 Creating intelligent chapters based on content...');
      const chapters = await this.createSmartChapters(transcript, metadata);
      console.error(`✅ Intelligent chapters: ${chapters.length}`);

      const result: AnalysisResult = {
        success: true,
        video_id: videoId,
        metadata,
        transcript,
        chapters,
        keyFrames: [],
        processing_time: (Date.now() - startTime) / 1000,
        analysis_method: 'enhanced_metadata_supadata',
        stats: {
          transcript_segments: transcript.length,
          chapters_generated: chapters.length,
          key_frames_extracted: 0,
          transcript_source: 'supadata_api',
          chapter_method: 'smart_content_analysis',
          video_downloaded: false
        }
      };

      console.error(`🎉 Analysis complete in ${result.processing_time.toFixed(1)}s`);
      return result;

    } catch (error) {
      console.error('❌ Analysis failed:', error);
      return {
        success: false,
        video_id: this.extractVideoId(youtubeUrl) || '',
        metadata: {},
        transcript: [],
        chapters: [],
        keyFrames: [],
        processing_time: (Date.now() - startTime) / 1000,
        analysis_method: 'enhanced_metadata_supadata',
        stats: {
          transcript_segments: 0,
          chapters_generated: 0,
          key_frames_extracted: 0,
          transcript_source: 'none',
          chapter_method: 'none',
          video_downloaded: false
        }
      };
    }
  }

  private async createSmartChapters(transcript: TranscriptSegment[], metadata: any): Promise<Chapter[]> {
    try {
      // Strategy 1: Parse from description timestamps
      const descriptionChapters = this.parseDescriptionChapters(metadata.description);
      if (descriptionChapters.length > 0) {
        console.error(`📚 Found ${descriptionChapters.length} chapters in description`);
        return descriptionChapters;
      }

      // Strategy 2: Advanced LLM-based chapter creation using transcript content
      if (transcript.length > 10) {
        const contentChapters = await this.createContentBasedChapters(transcript, metadata);
        if (contentChapters.length > 0) {
          console.error(`🤖 Created ${contentChapters.length} content-based chapters`);
          return contentChapters;
        }
      }

      // Strategy 3: Simple time-based chapters
      const timeChapters = this.createTimeChapters(metadata.duration);
      console.error(`⏰ Created ${timeChapters.length} time-based chapters`);
      return timeChapters;
    } catch (error) {
      console.error('Chapter creation error:', error);
      return this.createTimeChapters(metadata.duration);
    }
  }

  private parseDescriptionChapters(description: string): Chapter[] {
    const chapters: Chapter[] = [];
    
    // Look for patterns like "0:00 Introduction" or "1:23 - Chapter Title"
    const patterns = [
      /(\d{1,2}:\d{2}(?::\d{2})?)\s*[-–]?\s*(.+?)(?=\n|$|\d{1,2}:\d{2})/g,
      /(\d{1,2}:\d{2}(?::\d{2})?)\s+(.+?)(?=\n|$|\d{1,2}:\d{2})/g,
    ];

    for (const pattern of patterns) {
      const matches = Array.from(description.matchAll(pattern));
      if (matches.length >= 2) {  // Need at least 2 chapters to be valid
        for (let i = 0; i < matches.length; i++) {
          const [_, timestamp, title] = matches[i];
          const startSeconds = this.timestampToSeconds(timestamp);
          const endSeconds = i + 1 < matches.length 
            ? this.timestampToSeconds(matches[i + 1][1])
            : undefined;

          const cleanTitle = this.cleanTextForJson(title.trim());
          if (cleanTitle && cleanTitle.length > 3) {  // Valid title
            chapters.push({
              id: `desc_chapter_${i}`,
              title: cleanTitle.slice(0, 80),
              start_time: startSeconds,
              end_time: endSeconds || 0,
              summary: `Chapter covering: ${cleanTitle}`.slice(0, 200),
              key_topics: this.extractKeywords(cleanTitle),
              word_count: cleanTitle.split(/\s+/).length,
              main_topic: cleanTitle.slice(0, 100),
              source: 'description_timestamps'
            });
          }
        }
        if (chapters.length > 0) {
          return chapters;
        }
      }
    }
    return [];
  }

  private async createContentBasedChapters(transcript: TranscriptSegment[], metadata: any): Promise<Chapter[]> {
    try {
      // Group transcript into larger chunks for better analysis
      const chunkSize = Math.max(20, Math.floor(transcript.length / 8));  // Aim for 6-8 chapters max
      const chunks = [];

      for (let i = 0; i < transcript.length; i += chunkSize) {
        const chunk = transcript.slice(i, i + chunkSize);
        const chunkText = chunk.map(seg => seg.text).join(' ');
        const chunkStart = chunk[0].start;
        const chunkEnd = chunk[chunk.length - 1].end;

        chunks.push({
          text: chunkText,
          start: chunkStart,
          end: chunkEnd,
          segment_count: chunk.length
        });
      }

      // Create prompt for LLM analysis
      const durationMinutes = metadata.duration / 60;
      let prompt = `Analyze this ${durationMinutes.toFixed(1)}-minute video transcript and create logical chapters based on natural topic changes and content flow.

Video: "${metadata.title}" by ${metadata.author}

Transcript chunks with timestamps:
`;

      for (let i = 0; i < Math.min(chunks.length, 6); i++) {
        const chunk = chunks[i];
        const startMin = Math.floor(chunk.start / 60);
        const startSec = Math.floor(chunk.start % 60);
        prompt += `\n[${startMin}:${startSec.toString().padStart(2, '0')}] ${chunk.text.slice(0, 300)}...`;
      }

      prompt += `\n\nCreate chapters in this exact JSON format:
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

Make chapter titles engaging and descriptive. Ensure chapters flow logically and cover the full video duration.`;

      const response = await this.openai.chat.completions.create({
        model: "gpt-4",
        messages: [
          {
            role: "system",
            content: "You are an expert at analyzing video content and creating logical chapter divisions. Always respond with valid JSON."
          },
          { role: "user", content: prompt }
        ],
        max_tokens: 1000,
        temperature: 0.3
      });

      const content = response.choices[0].message.content?.trim() || '';
      let jsonContent = content;

      // Clean JSON from markdown
      if (content.startsWith('```')) {
        jsonContent = content.split('```')[1];
        if (jsonContent.startsWith('json')) {
          jsonContent = jsonContent.slice(4);
        }
      }

      const result = JSON.parse(jsonContent);
      const chaptersData = result.chapters || [];

      // Convert to our format with content-based word counts
      const formattedChapters: Chapter[] = [];
      for (let i = 0; i < chaptersData.length; i++) {
        const chapter = chaptersData[i];
        const startTime = parseFloat(chapter.start_seconds);
        const endTime = parseFloat(chapter.end_seconds);

        // Calculate actual word count from transcript
        const chapterText = this.getTranscriptTextForTimerange(transcript, startTime, endTime);
        const wordCount = chapterText.split(/\s+/).length;

        formattedChapters.push({
          id: `content_chapter_${i}`,
          title: this.cleanTextForJson(chapter.title).slice(0, 80),
          start_time: startTime,
          end_time: endTime,
          summary: this.cleanTextForJson(chapter.summary).slice(0, 200),
          key_topics: this.extractKeywords(chapter.main_topic + ' ' + chapter.title),
          word_count: wordCount,
          main_topic: this.cleanTextForJson(chapter.main_topic).slice(0, 100),
          source: 'content_analysis'
        });
      }

      return formattedChapters;
    } catch (error) {
      console.error('Content-based chapter creation error:', error);
      return [];
    }
  }

  private getTranscriptTextForTimerange(transcript: TranscriptSegment[], startTime: number, endTime: number): string {
    return transcript
      .filter(segment => segment.start >= startTime && segment.end <= endTime)
      .map(segment => segment.text)
      .join(' ');
  }

  private timestampToSeconds(timestamp: string): number {
    try {
      const parts = timestamp.split(':');
      if (parts.length === 2) {  // mm:ss
        return parseInt(parts[0]) * 60 + parseInt(parts[1]);
      } else if (parts.length === 3) {  // hh:mm:ss
        return parseInt(parts[0]) * 3600 + parseInt(parts[1]) * 60 + parseInt(parts[2]);
      }
      return 0;
    } catch {
      return 0;
    }
  }

  private extractKeywords(text: string): string[] {
    // Simple keyword extraction
    const words = text.toLowerCase().match(/\b[a-zA-Z]{3,}\b/g) || [];
    // Remove common words
    const commonWords = new Set([
      'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had',
      'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his',
      'how', 'its', 'may', 'new', 'now', 'old', 'see', 'two', 'who', 'boy',
      'did', 'man', 'men', 'say', 'she', 'too', 'use'
    ]);
    const uniqueWords = [...new Set(words)].filter(word => !commonWords.has(word));
    return uniqueWords.slice(0, 5);  // Return up to 5 unique keywords
  }

  private createTimeChapters(duration: number): Chapter[] {
    if (duration <= 0) return [];

    // Determine number of chapters based on duration
    let numChapters: number;
    if (duration < 180) {  // < 3 minutes
      numChapters = 2;
    } else if (duration < 600) {  // < 10 minutes
      numChapters = 3;
    } else if (duration < 1800) {  // < 30 minutes
      numChapters = 5;
    } else {
      numChapters = 6;
    }

    const chapterLength = duration / numChapters;
    const chapters: Chapter[] = [];

    for (let i = 0; i < numChapters; i++) {
      const startTime = i * chapterLength;
      const endTime = Math.min((i + 1) * chapterLength, duration);

      chapters.push({
        id: `time_chapter_${i}`,
        title: `Part ${i + 1}`,
        start_time: startTime,
        end_time: endTime,
        summary: `Content from ${this.secondsToTimestamp(startTime)} to ${this.secondsToTimestamp(endTime)}`,
        key_topics: [],
        word_count: 0,
        main_topic: `Time segment ${i + 1}`,
        source: 'time_based'
      });
    }

    return chapters;
  }

  private secondsToTimestamp(seconds: number): string {
    const minutes = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${minutes}:${secs.toString().padStart(2, '0')}`;
  }
}

// Main execution
async function main() {
  if (process.argv.length !== 3) {
    console.error(JSON.stringify({
      success: false,
      error: 'Usage: ts-node metadata_analysis.ts <youtube_url>'
    }));
    process.exit(1);
  }

  const youtubeUrl = process.argv[2];

  if (!youtubeUrl.includes('youtube.com') && !youtubeUrl.includes('youtu.be')) {
    console.error(JSON.stringify({
      success: false,
      error: 'Invalid YouTube URL'
    }));
    process.exit(1);
  }

  try {
    const analyzer = new EnhancedMetadataAnalyzer();
    const result = await analyzer.analyzeVideo(youtubeUrl);
    console.log(JSON.stringify(result));
    process.exit(result.success ? 0 : 1);
  } catch (error) {
    console.error(JSON.stringify({
      success: false,
      error: `Analysis error: ${error}`
    }));
    process.exit(1);
  }
}

if (require.main === module) {
  main();
} 