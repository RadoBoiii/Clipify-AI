import { NextResponse } from 'next/server';
//import ytdl from 'yt-dlp-wrap';
import { spawn } from 'child_process';
import path from 'path';
import fs from 'fs';

const TEMP_DIR = path.join(process.cwd(), 'public', 'temp');

function cleanOutput(output: string): string {
  // Remove download progress messages
  output = output.replace(/\[download\].*\n/g, '');
  
  // Remove emoji and other console output
  output = output.replace(/[🚀📹⚡✅🔄🖼️📚🎉❌].*\n/g, '');
  
  // Find the last JSON object in the output
  const jsonMatch = output.match(/\{[\s\S]*\}/);
  if (jsonMatch) {
    return jsonMatch[0];
  }
  
  return output;
}

export async function POST(req: Request): Promise<NextResponse> {
  try {
    const { url } = await req.json();

    // Create temp directory if not in production
    if (process.env.NODE_ENV !== 'production' && !fs.existsSync(TEMP_DIR)) {
      fs.mkdirSync(TEMP_DIR, { recursive: true });
    }

    // Choose script based on environment
    const isProduction = process.env.NODE_ENV === 'production';
    const scriptPath = isProduction 
      ? 'scripts/metadata_analysis.py'  // Production: metadata-only analysis
      : 'scripts/fast_video_analysis.py'; // Development: full analysis

    console.log(`Using ${isProduction ? 'metadata-only' : 'full'} analysis script`);
    console.log('Environment variables:', {
      NODE_ENV: process.env.NODE_ENV,
      SUPADATA_API_KEY: process.env.SUPADATA_API_KEY ? 'present' : 'missing'
    });

    // Run Python script for analysis
    const pythonProcess = spawn('python', [
      scriptPath,
      url
    ]);

    let output = '';
    let error = '';

    pythonProcess.stdout.on('data', (data) => {
      const chunk = data.toString();
      output += chunk;
      console.log('Python stdout:', chunk);
    });

    pythonProcess.stderr.on('data', (data) => {
      const chunk = data.toString();
      error += chunk;
      console.error('Python stderr:', chunk);
    });

    return new Promise<NextResponse>((resolve) => {
      pythonProcess.on('close', (code) => {
        if (code !== 0) {
          console.error('Python script error:', error);
          resolve(NextResponse.json(
            { error: `Failed to analyze video: ${error}` },
            { status: 500 }
          ));
          return;
        }

        try {
          // Clean the output before parsing
          const cleanedOutput = cleanOutput(output);
          console.log('Cleaned output:', cleanedOutput);
          
          const analysisResult = JSON.parse(cleanedOutput);
          
          if (!analysisResult.success) {
            resolve(NextResponse.json(
              { error: analysisResult.error || 'Analysis failed' },
              { status: 500 }
            ));
            return;
          }
          
          // Log transcript data
          console.log('📝 Transcript Data:', {
            totalSegments: analysisResult.transcript?.length || 0,
            firstSegment: analysisResult.transcript?.[0],
            lastSegment: analysisResult.transcript?.[analysisResult.transcript.length - 1],
            sampleSegments: analysisResult.transcript?.slice(0, 3)
          });
          
          resolve(NextResponse.json(analysisResult));
        } catch (e) {
          console.error('Failed to parse analysis result:', e);
          console.error('Raw output:', output);
          resolve(NextResponse.json(
            { error: 'Failed to parse analysis result' },
            { status: 500 }
          ));
        }
      });
    });
  } catch (error) {
    console.error('Error in analyze-video route:', error);
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
} 