import { NextResponse } from 'next/server';
import { OpenAI } from 'openai';

const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

export async function POST(req: Request) {
  try {
    const { question, video } = await req.json();

    // Compose a prompt with explicit title, description, and more transcript
    const prompt = `
You are an expert video assistant.

Title:
${video.metadata?.title || ''}

Description:
${video.metadata?.description || ''}

Metadata:
${JSON.stringify(video.metadata, null, 2)}

Video Uploader/ Channel: 
${video.metadata?.author || ''}
${video.metadata?.channel || ''}

Transcript (partial):
${video.transcript.slice(0, 100).map((t: any) => t.text).join(' ')}

Chapters:
${video.chapters.map((c: any) => c.title + ': ' + c.summary).join('\n')}

User question: ${question}
User question: ${question}
Answer as helpfully as possible in as much detail as possible. You may use internet source to try and understand the video/ scene/ topics that are involved in the video.
Avoid phrases like -"based on the transcript" or "based on the title" or "based on the description" or "based on the metadata"
if the user asks about the owner check information from the internet. Also for a lot of the answers reference the timestamps/ scenes from the video.
`;

    const completion = await openai.chat.completions.create({
      model: 'gpt-4', // or 'gpt-4' if you have access
      messages: [{ role: 'user', content: prompt }],
      max_tokens: 500,
    });

    const answer = completion.choices[0]?.message?.content || 'No answer from agent.';
    return NextResponse.json({ answer });
  } catch (error) {
    return NextResponse.json({ error: 'Failed to process chat request.' }, { status: 500 });
  }
} 