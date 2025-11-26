import os
import json
import threading
import re
from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from groq import Groq
from dotenv import load_dotenv
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

load_dotenv()

app = FastAPI()

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Groq client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Email credentials
EMAIL_HOST = os.getenv("EMAIL_HOST")
EMAIL_PORT = int(os.getenv("EMAIL_PORT"))
EMAIL_USER = os.getenv("EMAIL_USER")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
EMAIL_TO = os.getenv("EMAIL_TO")


def send_summary_email(ai_output):
    """Send the AI meeting summary via email (background)."""
    msg = MIMEMultipart()
    msg['From'] = EMAIL_USER
    msg['To'] = EMAIL_TO
    msg['Subject'] = "Meeting Summary"

    # Attach AI output directly
    body = f"Meeting Summary and Analysis:\n\n{ai_output}"
    msg.attach(MIMEText(body, 'plain'))

    try:
        server = smtplib.SMTP(EMAIL_HOST, EMAIL_PORT)
        server.starttls()
        server.login(EMAIL_USER, EMAIL_PASSWORD)
        server.send_message(msg)
        server.quit()
        print("Email sent successfully")
    except Exception as e:
        print("Error sending email:", e)


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Load homepage."""
    with open("templates/index.html", "r", encoding="utf-8") as f:
        html = f.read()
    return HTMLResponse(html)


@app.post("/process")
async def process_audio(file: UploadFile = File(...)):
    """Process uploaded audio: transcribe, extract tasks/decisions, send email."""
    file_path = f"temp_{file.filename}"

    # Save uploaded file
    with open(file_path, "wb") as f:
        f.write(await file.read())

    try:
        # 🔹 Transcribe using Groq Whisper
        with open(file_path, "rb") as f:
            transcript = client.audio.transcriptions.create(
                file=f,
                model="whisper-large-v3"
            )

        text = transcript.text  # Transcribed text

        # 🔹 Extract tasks and decisions using Llama
        prompt = f"""
You are an AI meeting assistant. Extract tasks and decisions from the transcript below.
Respond ONLY with valid JSON in this exact format:

{{
    "summary": "...",
    "tasks": ["...", "..."],
    "decisions": ["...", "..."]
}}

Transcript:
{text}
"""

        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )

        ai_output = completion.choices[0].message.content
        print("AI Output:", ai_output)  # For debugging

        # 🔹 Robust JSON parsing
        parsed = {"summary": "", "tasks": [], "decisions": []}
        try:
            match = re.search(r"\{.*\}", ai_output, re.DOTALL)
            if match:
                parsed = json.loads(match.group(0))
            else:
                print("No JSON found in AI output.")
        except json.JSONDecodeError as e:
            print("JSON decode error:", e)
            print("AI output:", ai_output)

        # Send summary email in background with entire AI output
        threading.Thread(target=send_summary_email, args=(ai_output,)).start()

        # Return transcription + analysis immediately
        return JSONResponse({
            "transcript": text,
            "analysis": parsed
        })

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

    finally:
        if os.path.exists(file_path):
            os.remove(file_path)