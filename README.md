# Clipify AI – Multimodal Video Analyzer

Clipify AI is a full-stack, AI-powered video analysis tool that extracts transcripts, chapters, and keyframes from YouTube videos. It combines a modern [Next.js](https://nextjs.org) frontend with a Python backend leveraging LLMs, Whisper, and advanced video processing.

---

## ✨ Features
- **YouTube Video Analysis**: Download and analyze YouTube videos with a single link.
- **Automatic Transcription**: Uses YouTube API and Whisper for accurate transcripts.
- **Intelligent Chapters**: LLM-powered chapter generation based on content.
- **Key Frame Extraction**: Visual summary of important video moments.
- **Modern UI**: Built with Next.js, Tailwind CSS, and React.
- **Chat About Video**: Ask questions and get answers about the video content.

---

## 🛠️ Project Structure

- `src/app/` – Next.js frontend (UI, API routes)
- `scripts/fast_video_analysis.py` – Python backend for video analysis
- `requirements.txt` – Python dependencies
- `tailwind.config.js` – Tailwind CSS configuration
- `.gitignore` – Ignores `venv/`, `node_modules/`, and other system files

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/RadoBoiii/Clipify-AI.git
cd Clipify-AI/multimodal-video-analyzer
```

### 2. Install Node.js dependencies
```bash
npm install
# or
yarn install
```

### 3. Set up the Python backend

**Create a virtual environment inside the project folder:**
```bash
python -m venv venv
source venv/bin/activate
```

**Install Python dependencies:**
```bash
pip install -r requirements.txt
```

### 4. Environment Variables
- Set your OpenAI API key in your environment:
  ```bash
  export OPENAI_API_KEY=your-key-here
  ```
- (Optional) Add other API keys as needed for your use case.

### 5. Run the Next.js development server
```bash
npm run dev
```
Open [http://localhost:3000](http://localhost:3000) in your browser.

---

## 🧑‍💻 Python Video Analysis Script

The main script is at `scripts/fast_video_analysis.py`.

**Usage:**
```bash
python scripts/fast_video_analysis.py <youtube_url>
```
- All logs and errors are printed to stderr.
- Only the final JSON result is printed to stdout.

**Example:**
```bash
python scripts/fast_video_analysis.py https://www.youtube.com/watch?v=YOUR_VIDEO_ID
```

**What it does:**
- Downloads the video (using yt-dlp)
- Extracts metadata, transcript, and keyframes
- Generates intelligent chapters using LLMs
- Outputs a single JSON result

---

## 🐞 Troubleshooting

- **Video not downloading?**
  - Make sure `yt-dlp` is up to date (`pip install --upgrade yt-dlp`)
  - Check your internet connection
  - Some videos may be region-locked or age-restricted
  - See stderr for detailed error logs

- **Dependency conflicts?**
  - If you see errors about `pydantic` or other packages, try updating or creating a fresh virtual environment

- **Transcript not found?**
  - The script will fall back to Whisper transcription if YouTube transcript is unavailable

- **venv not ignored?**
  - The `.gitignore` is set up to ignore `venv/` in the project root. Make sure your virtual environment is inside the project folder.

---

## 📝 .gitignore
- Ignores `venv/`, `node_modules/`, and other system/IDE files
- Do **not** commit your virtual environment or local environment files

---

## 🛠️ Customization
- Edit `src/app/page.tsx` for the main UI
- Edit `scripts/fast_video_analysis.py` for backend logic
- Tweak `tailwind.config.js` for custom styles

---

## 📦 Deployment
- Deploy the Next.js frontend to Vercel or any Node.js host
- The Python backend can be run as a local service or deployed as a serverless function

---

## 🤝 Contributing
Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.

---

## 📄 License
MIT
