import os
import re
import json
import sqlite3
import tempfile
import pandas as pd
import time
import threading
from pathlib import Path

import gradio as gr
import numpy as np
import faiss
from groq import Groq, RateLimitError, APIConnectionError, APIStatusError
from pypdf import PdfReader
from pptx import Presentation
from docx import Document

# -------------------- Configuration --------------------
APP_NAME = "SmartScholar AI Assistant"
MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
API_KEY = os.getenv("GROQ_API_KEY")

if not API_KEY:
    raise RuntimeError(
        "GROQ_API_KEY is not configured. Add it in Render → Environment."
    )

client = Groq(api_key=API_KEY)

SYSTEM_PROMPT = """
You are SmartScholar AI Assistant, an educational personal study assistant.
Explain difficult topics clearly, use practical examples, guide students step by step,
help with exam preparation, and correct mistakes politely.
"""

TEACHING_MODES = {
    "Beginner": "Explain in simple language, step by step, with everyday examples.",
    "Detailed": "Give an in-depth explanation with definitions, examples, applications, advantages and disadvantages.",
    "Interview": "Explain from an AIML interview perspective, with definitions, technical points and practical examples."
}

# In-memory document index. Uploaded files are processed for this running instance.
document_texts = {}
all_chunks = []
# Load the embedding model only when Document Q&A is first used. This lets
# Render start the Gradio web server before downloading/loading the model.
embedding_model = None
_embedding_model_lock = threading.Lock()

def get_embedding_model():
    global embedding_model
    if embedding_model is None:
        with _embedding_model_lock:
            if embedding_model is None:
                print("Loading Sentence Transformers library for Document Q&A...", flush=True)
                from sentence_transformers import SentenceTransformer
                print("Loading embedding model for Document Q&A...", flush=True)
                embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
                print("Sentence Transformer model loaded.", flush=True)
    return embedding_model

index = None

# Render's local filesystem is ephemeral on free instances. SQLite is useful for
# a demo, but use managed external storage for durable production history.
DB_PATH = os.getenv("DATABASE_PATH", "study_assistant.db")


def db_connect():
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS quiz_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            subject TEXT,
            difficulty TEXT,
            total_questions INTEGER,
            correct_answers INTEGER,
            score REAL,
            date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    return conn


def safe_groq_response(messages, temperature=0.4):
    try:
        response = client.chat.completions.create(
            model=MODEL, messages=messages, temperature=temperature
        )
        return response.choices[0].message.content or "The AI returned an empty response."
    except RateLimitError:
        return "The AI service rate limit was reached. Please wait and try again."
    except APIConnectionError:
        return "Could not connect to the AI service. Please try again."
    except APIStatusError as exc:
        return f"The AI service returned HTTP {exc.status_code}. Please try again later."
    except Exception as exc:
        print("Groq error:", repr(exc))
        return "Something went wrong while contacting the AI service."


def clean_and_parse_json(response_text):
    text = response_text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("The AI response did not contain a JSON object.")
    return json.loads(text[start:end + 1])


# -------------------- Document extraction and RAG --------------------
def extract_text(file_path):
    path = Path(file_path)
    ext = path.suffix.lower()
    output = []

    if ext == ".pdf":
        reader = PdfReader(str(path))
        for page_no, page in enumerate(reader.pages, 1):
            text = page.extract_text() or ""
            if text.strip():
                output.append(f"Page {page_no}\n{text}")

    elif ext == ".pptx":
        presentation = Presentation(str(path))
        for slide_no, slide in enumerate(presentation.slides, 1):
            output.append(f"Slide {slide_no}")
            for shape in slide.shapes:
                if getattr(shape, "has_text_frame", False):
                    output.append(shape.text)
                if getattr(shape, "has_table", False):
                    for row in shape.table.rows:
                        output.append(" | ".join(cell.text for cell in row.cells))

    elif ext == ".docx":
        document = Document(str(path))
        output.extend(p.text for p in document.paragraphs if p.text.strip())
        for table in document.tables:
            for row in table.rows:
                output.append(" | ".join(cell.text for cell in row.cells))

    else:
        raise ValueError("Supported formats for this Render version: PDF, PPTX and DOCX.")

    text = "\n".join(output).strip()
    if not text:
        raise ValueError("No readable text found. Scanned/image-only documents need OCR.")
    return text


def split_text(text, chunk_size=500, overlap=100):
    chunks = []
    start = 0
    step = chunk_size - overlap
    while start < len(text):
        piece = text[start:start + chunk_size].strip()
        if piece:
            chunks.append(piece)
        start += step
    return chunks


def process_documents(files):
    global document_texts, all_chunks, index

    if not files:
        return "Please upload at least one document."

    new_texts, new_chunks, errors = {}, [], []

    for file_path in files:
        name = Path(file_path).name
        try:
            text = extract_text(file_path)
            new_texts[name] = text
            for chunk in split_text(text):
                new_chunks.append({"filename": name, "text": chunk})
        except Exception as exc:
            errors.append(f"{name}: {exc}")

    if not new_chunks:
        details = "\n".join(errors)
        return "No documents could be processed.\n" + details

    try:
        vectors = get_embedding_model().encode(
            [item["text"] for item in new_chunks],
            convert_to_numpy=True,
            normalize_embeddings=True
        ).astype("float32")
        new_index = faiss.IndexFlatIP(vectors.shape[1])
        new_index.add(vectors)
    except Exception as exc:
        return f"Could not build the document search index: {exc}"

    document_texts = new_texts
    all_chunks = new_chunks
    index = new_index

    report = (
        f"Processed {len(new_texts)} document(s) and indexed "
        f"{len(new_chunks)} text chunks for this running session."
    )
    if errors:
        report += "\n\nFiles with errors:\n" + "\n".join(errors)
    report += "\n\nNote: uploaded documents are held in this running app's memory; "
    report += "they will need to be uploaded again after a service restart."
    return report


def search_documents(question, top_k=4):
    if index is None or index.ntotal == 0:
        return []
    vector = get_embedding_model().encode(
        [question], convert_to_numpy=True, normalize_embeddings=True
    ).astype("float32")
    scores, ids = index.search(vector, min(top_k, index.ntotal))
    results = []
    for score, idx in zip(scores[0], ids[0]):
        if idx >= 0:
            item = all_chunks[int(idx)]
            results.append({
                "filename": item["filename"],
                "text": item["text"],
                "similarity": float(score)
            })
    return results


def document_chat(message, history, mode):
    if not message or not message.strip():
        return "Please enter a question."
    results = search_documents(message, top_k=4)
    if not results:
        return "Please upload and process study materials first."

    context = "\n\n".join(
        f"Source: {r['filename']}\n{r['text']}" for r in results
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT + "\n" + TEACHING_MODES[mode] +
         "\nAnswer from the supplied document context as your primary source. "
         "Do not invent information. If it is absent, say so and name the source."}
    ]
    for item in (history or [])[-10:]:
        if isinstance(item, dict) and item.get("role") in ("user", "assistant"):
            content = item.get("content")
            if isinstance(content, str):
                messages.append({"role": item["role"], "content": content})
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            if item[0]:
                messages.append({"role": "user", "content": str(item[0])})
            if item[1]:
                messages.append({"role": "assistant", "content": str(item[1])})
    messages.append({
        "role": "user",
        "content": f"Document context:\n{context}\n\nStudent question:\n{message}"
    })
    return safe_groq_response(messages, temperature=0.3)


# -------------------- Chatbot --------------------
def respond(message, history, mode):
    if not message or not message.strip():
        return "Please enter a question."
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT + "\n" + TEACHING_MODES[mode]}
    ]
    for item in (history or [])[-10:]:
        if isinstance(item, dict) and item.get("role") in ("user", "assistant"):
            if isinstance(item.get("content"), str):
                messages.append({"role": item["role"], "content": item["content"]})
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            if item[0]:
                messages.append({"role": "user", "content": str(item[0])})
            if item[1]:
                messages.append({"role": "assistant", "content": str(item[1])})
    messages.append({"role": "user", "content": message})
    return safe_groq_response(messages, temperature=0.5)


# -------------------- Quiz --------------------
def generate_structured_quiz(num_questions=5, difficulty="Medium"):
    if not document_texts:
        raise ValueError("Please upload and process study materials first.")
    context = "\n\n".join(
        f"Document: {name}\n{text[:8000]}" for name, text in document_texts.items()
    )
    prompt = f"""
Generate exactly {int(num_questions)} multiple-choice questions at {difficulty} difficulty.
Use only the supplied study material. Return only JSON:
{{"questions":[{{"question":"...","options":["A. ...","B. ...","C. ...","D. ..."],"answer":"A","explanation":"..."}}]}}
Rules: exactly four options per question; answer must be A, B, C or D; no duplicates;
include a clear explanation.
Study material:
{context}
"""
    for attempt in range(3):
        raw = safe_groq_response([{"role": "user", "content": prompt}], temperature=0.3)
        try:
            data = clean_and_parse_json(raw)
            questions = data["questions"]
            if len(questions) != int(num_questions):
                raise ValueError("Incorrect number of questions.")
            seen = set()
            for q in questions:
                if not q.get("question") or len(q.get("options", [])) != 4:
                    raise ValueError("Question/options structure is invalid.")
                if q.get("answer") not in ("A", "B", "C", "D"):
                    raise ValueError("Correct answer must be A, B, C or D.")
                if not q.get("explanation"):
                    raise ValueError("Question explanation is missing.")
                key = q["question"].strip().lower()
                if key in seen:
                    raise ValueError("Duplicate questions.")
                seen.add(key)
            return questions
        except Exception as exc:
            print(f"Quiz JSON validation attempt {attempt + 1}: {exc}")
    raise ValueError("Could not generate a valid quiz after three attempts.")


def generate_quiz_ui(count, difficulty):
    try:
        questions = generate_structured_quiz(int(count), difficulty)
        return questions, f"Generated {len(questions)} questions. Select an answer for each."
    except Exception as exc:
        return [], f"Quiz generation failed: {exc}"


def submit_quiz(questions, difficulty, *answers):
    if not questions:
        return "Generate a quiz first."
    if any(a not in ("A", "B", "C", "D") for a in answers[:len(questions)]):
        return "Please answer every question before submitting."
    score, feedback = 0, []
    for i, q in enumerate(questions):
        correct = q["answer"]
        selected = answers[i]
        is_correct = selected == correct
        score += int(is_correct)
        feedback.append(
            f"**Question {i+1}: {'Correct' if is_correct else 'Incorrect'}**\n\n"
            f"Your answer: {selected} · Correct answer: {correct}\n\n"
            f"{q['explanation']}"
        )
    total = len(questions)
    percent = score / total * 100
    try:
        conn = db_connect()
        conn.execute(
            """INSERT INTO quiz_results
            (subject, difficulty, total_questions, correct_answers, score)
            VALUES (?, ?, ?, ?, ?)""",
            ("General Study", difficulty, total, score, percent)
        )
        conn.commit()
        conn.close()
        saved = "Quiz result saved in the app database."
    except Exception as exc:
        print("Database save error:", repr(exc))
        saved = "Score calculated, but database saving failed."
    return (
        f"# Score: {score}/{total} ({percent:.1f}%)\n\n"
        + "\n\n---\n\n".join(feedback)
        + f"\n\n{saved}"
    )


# -------------------- Notes and planner --------------------
def summarize_notes(summary_type):
    if not document_texts:
        return "Please upload and process study materials first."
    instructions = {
        "Short": "Create concise revision notes with key concepts, definitions and formulas.",
        "Detailed": "Create detailed topic-wise notes with definitions, examples, formulas and applications.",
        "Exam Preparation": "Create exam notes with definitions, key concepts, formulas, likely testable topics, short and long answer questions."
    }
    outputs = []
    for filename, text in document_texts.items():
        prompt = (
            f"Create {summary_type} study notes from this material. "
            f"{instructions[summary_type]} Preserve technical terms. "
            f"Do not invent information. Organize with headings.\n\n{text[:12000]}"
        )
        summary = safe_groq_response([{"role": "user", "content": prompt}], 0.3)
        outputs.append(f"# Notes: {filename}\n\n{summary}")
    return "\n\n---\n\n".join(outputs)


def generate_study_plan(subject, topics, days, hours_per_day, current_level):
    if not subject or not subject.strip() or not topics or not topics.strip():
        return "Please enter the subject and topics."
    if not days or not hours_per_day or days <= 0 or hours_per_day <= 0:
        return "Days and study hours must be greater than zero."
    prompt = f"""
Create a personalized day-wise study timetable.
Subject: {subject}
Topics: {topics}
Total days: {days}
Available study hours per day: {hours_per_day}
Current level: {current_level}
Cover all topics, allocate realistic time, include breaks, revision, practice,
daily goals and a mock test. Do not exceed the daily hours. Present a clear table
and finish with preparation tips and a revision strategy.
"""
    return safe_groq_response([{"role": "user", "content": prompt}], 0.5)


# -------------------- Flashcards --------------------
def generate_structured_flashcards(num_cards=10, difficulty="Medium"):
    if not document_texts:
        raise ValueError("Please upload and process study materials first.")
    context = "\n\n".join(
        f"Document: {name}\n{text[:8000]}" for name, text in document_texts.items()
    )
    prompt = f"""
Generate exactly {int(num_cards)} distinct {difficulty} flashcards using only
this study material. Return only JSON:
{{"flashcards":[{{"question":"...","answer":"..."}}]}}
Every card must have a non-empty question and answer. Avoid duplicates.
Study material:
{context}
"""
    for attempt in range(3):
        raw = safe_groq_response([{"role": "user", "content": prompt}], 0.4)
        try:
            cards = clean_and_parse_json(raw)["flashcards"]
            if len(cards) != int(num_cards):
                raise ValueError("Incorrect card count.")
            seen = set()
            for card in cards:
                if not card.get("question", "").strip() or not card.get("answer", "").strip():
                    raise ValueError("A card is missing a question or answer.")
                key = card["question"].strip().lower()
                if key in seen:
                    raise ValueError("Duplicate cards.")
                seen.add(key)
            return cards
        except Exception as exc:
            print(f"Flashcard validation attempt {attempt + 1}: {exc}")
    raise ValueError("Could not generate valid flashcards after three attempts.")


def start_flashcards(num_cards, difficulty):
    try:
        cards = generate_structured_flashcards(int(num_cards), difficulty)
        return cards, 0, f"### Card 1 of {len(cards)}\n\n{cards[0]['question']}", ""
    except Exception as exc:
        return [], 0, f"Flashcard generation failed: {exc}", ""


def reveal_flashcard(cards, current_index):
    if not cards:
        return "Generate flashcards first."
    return cards[int(current_index)]["answer"]


def next_flashcard(cards, current_index):
    if not cards:
        return 0, "Generate flashcards first.", ""
    new_index = int(current_index) + 1
    if new_index >= len(cards):
        return current_index, "You have completed all flashcards!", ""
    return new_index, f"### Card {new_index + 1} of {len(cards)}\n\n{cards[new_index]['question']}", ""


# -------------------- Progress dashboard --------------------
def get_progress_dashboard():
    conn = db_connect()
    stats = conn.execute("""
        SELECT COUNT(*), ROUND(AVG(score), 2), ROUND(MAX(score), 2)
        FROM quiz_results
    """).fetchone()
    history = conn.execute("""
        SELECT subject, difficulty, total_questions, correct_answers, ROUND(score, 2), date
        FROM quiz_results ORDER BY date DESC
    """).fetchall()
    conn.close()
    stats_md = (
        f"**Total quizzes:** {stats[0] or 0}  \n"
        f"**Average score:** {stats[1] if stats[1] is not None else 0}%  \n"
        f"**Best score:** {stats[2] if stats[2] is not None else 0}%"
    )
    rows = [
        [r[0], r[1], r[2], r[3], r[4], r[5]] for r in history
    ]
    return stats_md, rows


# -------------------- Gradio UI --------------------
with gr.Blocks(title=APP_NAME, theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        "# 🎓 SmartScholar AI Assistant\n"
        "### AI-Powered Personal Study Assistant Chatbot\n"
        "Learn, revise, practise and track your progress."
    )

    with gr.Tab("Study Chatbot"):
        mode = gr.Radio(
            ["Beginner", "Detailed", "Interview"],
            value="Beginner", label="Teaching Mode"
        )
        gr.ChatInterface(fn=respond, additional_inputs=[mode])

    with gr.Tab("Document Q&A"):
        file_input = gr.File(
            label="Upload study materials",
            file_count="multiple",
            file_types=[".pdf", ".pptx", ".docx"],
            type="filepath"
        )
        process_button = gr.Button("Process Documents", variant="primary")
        upload_status = gr.Markdown()
        process_button.click(process_documents, inputs=file_input, outputs=upload_status)
        document_mode = gr.Radio(
            ["Beginner", "Detailed", "Interview"],
            value="Beginner", label="Teaching Mode"
        )
        gr.ChatInterface(fn=document_chat, additional_inputs=[document_mode])

    with gr.Tab("Interactive Quiz"):
        gr.Markdown("Generate a quiz from your processed study materials.")
        question_count = gr.Dropdown([5, 10, 12, 15, 20], value=5, label="Number of questions")
        quiz_difficulty = gr.Dropdown(["Easy", "Medium", "Hard"], value="Medium", label="Difficulty")
        generate_quiz_button = gr.Button("Generate Quiz", variant="primary")
        quiz_status = gr.Markdown()
        quiz_state = gr.State([])
        quiz_answer_components = []
        for i in range(20):
            with gr.Group(visible=False) as question_group:
                question_text = gr.Markdown()
                answer_radio = gr.Radio(["A", "B", "C", "D"], label="Your answer")
            quiz_answer_components.append((question_group, question_text, answer_radio))

        submit_quiz_button = gr.Button("Submit Answers")
        quiz_result = gr.Markdown()
        def generate_and_display(count, level):
            questions, message = generate_quiz_ui(count, level)
            outputs = [questions, message]
            for i in range(20):
                if i < len(questions):
                    q = questions[i]
                    outputs.extend([
                        gr.update(visible=True),
                        f"**Question {i+1}. {q['question']}**\n\n" + "\n".join(q["options"]),
                        gr.update(choices=["A", "B", "C", "D"], value=None)
                    ])
                else:
                    outputs.extend([gr.update(visible=False), "", gr.update(choices=[], value=None)])
            return outputs
        generate_quiz_button.click(
            generate_and_display,
            inputs=[question_count, quiz_difficulty],
            outputs=[quiz_state, quiz_status] + [
                component for group in quiz_answer_components for component in group
            ]
        )
        submit_quiz_button.click(
            submit_quiz,
            inputs=[quiz_state, quiz_difficulty] + [item[2] for item in quiz_answer_components],
            outputs=quiz_result
        )

    with gr.Tab("Notes Summarizer"):
        summary_type = gr.Dropdown(
            ["Short", "Detailed", "Exam Preparation"],
            value="Short", label="Summary style"
        )
        summary_button = gr.Button("Summarize Notes")
        summary_output = gr.Markdown()
        summary_button.click(summarize_notes, inputs=summary_type, outputs=summary_output)

    with gr.Tab("Study Planner"):
        subject = gr.Textbox(label="Subject")
        topics = gr.Textbox(label="Topics", lines=4)
        days = gr.Number(label="Available days", value=7)
        hours = gr.Number(label="Study hours per day", value=3)
        level = gr.Dropdown(["Beginner", "Intermediate", "Advanced"], value="Beginner", label="Current level")
        planner_button = gr.Button("Generate Study Plan")
        planner_output = gr.Markdown()
        planner_button.click(generate_study_plan, inputs=[subject, topics, days, hours, level], outputs=planner_output)

    with gr.Tab("Interactive Flashcards"):
        card_count = gr.Dropdown([5, 10, 15, 20], value=10, label="Number of cards")
        card_difficulty = gr.Dropdown(["Easy", "Medium", "Hard"], value="Medium", label="Difficulty")
        generate_cards_button = gr.Button("Generate Flashcards", variant="primary")
        flashcard_state = gr.State([])
        current_index = gr.State(0)
        question_display = gr.Markdown()
        show_answer_button = gr.Button("Show Answer")
        answer_display = gr.Markdown()
        next_button = gr.Button("Next Card")
        generate_cards_button.click(
            start_flashcards,
            inputs=[card_count, card_difficulty],
            outputs=[flashcard_state, current_index, question_display, answer_display]
        )
        show_answer_button.click(
            reveal_flashcard,
            inputs=[flashcard_state, current_index],
            outputs=answer_display
        )
        next_button.click(
            next_flashcard,
            inputs=[flashcard_state, current_index],
            outputs=[current_index, question_display, answer_display]
        )

    with gr.Tab("Progress Tracking"):
        refresh_button = gr.Button("Refresh Progress")
        statistics = gr.Markdown()
        history_table = gr.Dataframe(
            headers=["Subject", "Difficulty", "Total", "Correct", "Score (%)", "Date"],
            datatype=["str", "str", "number", "number", "number", "str"],
            interactive=False,
            label="Quiz history"
        )
        refresh_button.click(get_progress_dashboard, outputs=[statistics, history_table])


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "7860"))
    print("SmartScholar app initialized. Starting Gradio server...", flush=True)
    demo.queue().launch(server_name="0.0.0.0", server_port=port)
