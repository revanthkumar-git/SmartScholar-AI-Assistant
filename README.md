# SmartScholar AI Assistant

AI-Powered Personal Study Assistant Chatbot.

## Features
- Study chatbot with Beginner, Detailed and Interview teaching modes
- Document Q&A using RAG, Sentence Transformers and FAISS
- Interactive multiple-choice quizzes with scoring
- Notes summarization
- AI study planner
- Interactive flashcards
- Quiz progress tracking with SQLite

## Run locally
1. Install Python 3.10 or 3.11.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set your Groq API key as an environment variable:
   ```bash
   # macOS/Linux
   export GROQ_API_KEY="your-key"

   # Windows PowerShell
   $env:GROQ_API_KEY="your-key"
   ```
4. Start:
   ```bash
   python app.py
   ```

## Render deployment
- Build command: `pip install -r requirements.txt`
- Start command: `python app.py`
- Add `GROQ_API_KEY` under the Render service's Environment settings.
- Optional: set `GROQ_MODEL` to a model available to your Groq account.

The app reads Render's `PORT` environment variable and binds to `0.0.0.0`.

## Important storage notes
The uploaded documents and FAISS index are currently held in application memory. They are lost when the service restarts or redeploys, and each running instance has its own memory.

Quiz history is stored in SQLite at `study_assistant.db`. Render Free services have an ephemeral filesystem, so this database is not durable across restarts/redeploys. For persistent production storage, connect a managed PostgreSQL database and object storage (or another durable storage service).

## Supported document types
This deployment version supports text-extractable PDF, PPTX and DOCX files. Older `.ppt` and `.doc` files were supported in the Colab notebook using LibreOffice; this Render-ready version does not include LibreOffice conversion. Scanned PDFs need OCR, which is not included.
