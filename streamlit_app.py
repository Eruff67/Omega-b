# ==============================================================
# Jack.AI — Super Offline Edition
# Personal Property (© You)
# Single-file, offline Streamlit app
# ==============================================================


# IMPORT GITHUB REPO


# END OF GIRHUB IMPORTS

# RETRIEVE SKLEARN MODULE



# END OF SKLEARN RETRIEVAL


import os, sys, json, time, random, threading
from datetime import datetime
from io import BytesIO
from typing import List, Dict, Any, Optional
import streamlit as st

# --- Optional imports (handled gracefully if missing) ---
OCR_AVAILABLE = PDF_AVAILABLE = TTS_AVAILABLE = STT_AVAILABLE = SKLEARN_AVAILABLE = False
try:
    from PIL import Image; import pytesseract; OCR_AVAILABLE = True
except Exception: pass
try:
    import fitz; PDF_AVAILABLE = True  # PyMuPDF
except Exception: pass
try:
    import pyttsx3; TTS_AVAILABLE = True
except Exception: pass
try:
    import speech_recognition as sr; STT_AVAILABLE = True
except Exception: pass
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LinearRegression
    import numpy as np; SKLEARN_AVAILABLE = True
except Exception: pass

# -------------------------------
# Directories and files
# -------------------------------
DATA_DIR = os.path.join(os.getcwd(), "jack_data")
os.makedirs(DATA_DIR, exist_ok=True)
FILES = {
    "chat": "chat.json",
    "mem": "memories.json",
    "persona": "persona.json",
    "tiny": "tiny_model.json",
    "gallery": "gallery.json",
}

def file_path(key): return os.path.join(DATA_DIR, FILES[key])

def load_json(key, fallback):
    p = file_path(key)
    if os.path.exists(p):
        try:
            with open(p, "r", encoding="utf-8") as f: return json.load(f)
        except Exception: return fallback
    return fallback

def save_json(key, data):
    with open(file_path(key), "w", encoding="utf-8") as f: json.dump(data, f, indent=2)

# -------------------------------
# Initialize state
# -------------------------------
if "chat" not in st.session_state: st.session_state.chat = load_json("chat", [])
if "persona" not in st.session_state: st.session_state.persona = load_json("persona", {"name":"Jack","style":"default","tone":"neutral"})
if "memories" not in st.session_state: st.session_state.memories = load_json("mem", [])
if "gallery" not in st.session_state: st.session_state.gallery = load_json("gallery", [])
if "tiny_model" not in st.session_state: st.session_state.tiny_model = load_json("tiny", {"trained":False})
if "training" not in st.session_state: st.session_state.training = False

# -------------------------------
# Helper functions
# -------------------------------
def speak(text):
    if not TTS_AVAILABLE: return "TTS not available."
    engine = pyttsx3.init(); engine.say(text); engine.runAndWait()

def listen():
    if not STT_AVAILABLE: return "STT not available."
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Listening... speak now")
        audio = r.listen(source)
    try: return r.recognize_google(audio)
    except Exception as e: return f"Error: {e}"

def extract_text_from_pdf(uploaded):
    if not PDF_AVAILABLE: return "PDF extraction unavailable."
    text = ""
    with fitz.open(stream=uploaded.read(), filetype="pdf") as doc:
        for page in doc: text += page.get_text()
    return text

def extract_text_from_image(uploaded):
    if not OCR_AVAILABLE: return "OCR unavailable."
    img = Image.open(uploaded)
    return pytesseract.image_to_string(img)

def summarize_text(text: str, max_len: int = 200) -> str:
    words = text.split()
    return " ".join(words[:max_len]) + ("..." if len(words) > max_len else "")

# -------------------------------
# Tiny offline ML brain
# -------------------------------
def train_tiny_model(examples: List[Dict[str,str]]):
    if not SKLEARN_AVAILABLE: return {"trained": False, "reason": "sklearn missing"}
    if not examples: return {"trained": False, "reason": "no examples"}
    vec = TfidfVectorizer()
    X = vec.fit_transform([e["prompt"] for e in examples])
    y = np.array([len(e["response"]) for e in examples])
    reg = LinearRegression().fit(X, y)
    save_json("tiny", {"trained": True, "features": len(vec.get_feature_names_out())})
    return {"trained": True, "features": len(vec.get_feature_names_out())}

def tiny_predict(text: str) -> str:
    if not st.session_state.tiny_model.get("trained"): return "🤔 Model not trained yet."
    return random.choice([
        "That's an interesting take.",
        "I remember something similar.",
        "Let's connect that to what we learned.",
        "I’d say it depends on perspective.",
        "That reminds me of a prior example."
    ])

# -------------------------------
# Chat and command handling
# -------------------------------
def add_message(role, content):
    st.session_state.chat.append({"role": role, "content": content, "time": datetime.now().isoformat()})
    save_json("chat", st.session_state.chat)

def handle_command(cmd):
    parts = cmd.strip().split(" ", 1)
    main = parts[0].lower()
    arg = parts[1] if len(parts) > 1 else ""

    if main == "/clear":
        st.session_state.chat = []; save_json("chat", [])
        return "🧹 Chat cleared."

    elif main == "/persona":
        if arg:
            st.session_state.persona["style"] = arg
            save_json("persona", st.session_state.persona)
            return f"🧠 Persona switched to '{arg}'."
        else:
            return f"Current persona: {st.session_state.persona}"

    elif main == "/learn":
        st.session_state.memories.append({"memory": arg, "time": datetime.now().isoformat()})
        save_json("mem", st.session_state.memories)
        return "💾 Learned new fact."

    elif main == "/train":
        if st.session_state.training: return "Training already in progress."
        st.session_state.training = True
        bar = st.progress(0)
        for i in range(100):
            time.sleep(0.02); bar.progress(i+1)
        result = train_tiny_model(st.session_state.chat)
        st.session_state.training = False
        return f"✅ Training complete: {result}"

    elif main == "/summarize":
        convo = " ".join([m["content"] for m in st.session_state.chat if m["role"]=="user"])
        return summarize_text(convo)

    elif main == "/speak":
        speak(arg or "Hello")
        return "🔊 Spoke out loud."

    else:
        return f"Unknown command: {cmd}"

# -------------------------------
# UI
# -------------------------------
st.set_page_config(page_title="Jack.AI — Offline", layout="wide")

st.sidebar.title("Omega-B Control Panel")
choice = st.sidebar.radio("Mode", ["💬 Chat", "🧠 Memory", "🖼️ Gallery", "📂 Import/Export", "⚙️ Debug"])

st.sidebar.markdown("---")
st.sidebar.write(f"Persona: **{st.session_state.persona['style']}**")
st.sidebar.write("Offline mode ✅")

# -------------------------------
# Chat Tab
# -------------------------------
if choice == "💬 Chat":
    st.title("💬 Jack.AI — Offline Assistant")
    for msg in st.session_state.chat:
        align = "end" if msg["role"] == "user" else "start"
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    prompt = st.chat_input("Message or /command...")
    if prompt:
        add_message("user", prompt)
        if prompt.startswith("/"):
            response = handle_command(prompt)
        else:
            # Simple persona-aware response
            mood = st.session_state.persona.get("style", "neutral")
            if st.session_state.tiny_model.get("trained"):
                ai = tiny_predict(prompt)
            else:
                ai = random.choice([
                    f"As {mood} Jack, I'd say that’s interesting.",
                    "I’ll note that down for next time.",
                    "Let's explore that thought further."
                ])
            response = ai
        add_message("assistant", response)
        st.rerun()

# -------------------------------
# Memory Tab
# -------------------------------
elif choice == "🧠 Memory":
    st.title("🧠 Memory")
    if st.session_state.memories:
        for m in st.session_state.memories:
            st.info(f"{m['time']}: {m['memory']}")
    else:
        st.warning("No memories yet.")

# -------------------------------
# Gallery Tab
# -------------------------------
elif choice == "🖼️ Gallery":
    st.title("🖼️ Local Gallery")
    up = st.file_uploader("Upload image or PDF", type=["png","jpg","jpeg","pdf"])
    if up:
        if up.type == "application/pdf" and PDF_AVAILABLE:
            text = extract_text_from_pdf(up)
            st.text_area("Extracted PDF Text", text, height=200)
        elif up.type.startswith("image/") and OCR_AVAILABLE:
            text = extract_text_from_image(up)
            st.text_area("Extracted Image Text", text, height=200)
        st.success("File processed.")
    if st.session_state.gallery:
        for item in st.session_state.gallery:
            st.image(item["data"], caption=item["caption"])

# -------------------------------
# Import / Export
# -------------------------------
elif choice == "📂 Import/Export":
    st.title("📂 Import / Export Data")
    exp = st.button("Export all data")
    if exp:
        bundle = {k: load_json(k, []) for k in FILES.keys()}
        st.download_button("Download JSON", json.dumps(bundle, indent=2), "jack_backup.json")
    imp = st.file_uploader("Import JSON bundle", type=["json"])
    if imp:
        data = json.load(imp)
        for k in FILES.keys():
            if k in data: save_json(k, data[k])
        st.success("Imported data.")

# -------------------------------
# Debug Tab
# -------------------------------
elif choice == "⚙️ Debug":
    st.title("⚙️ Debug Console")
    st.json({
        "persona": st.session_state.persona,
        "chat_len": len(st.session_state.chat),
        "memories_len": len(st.session_state.memories),
        "tiny_model": st.session_state.tiny_model,
        "OCR_AVAILABLE": OCR_AVAILABLE,
        "PDF_AVAILABLE": PDF_AVAILABLE,
        "TTS_AVAILABLE": TTS_AVAILABLE,
        "STT_AVAILABLE": STT_AVAILABLE,
        "SKLEARN_AVAILABLE": SKLEARN_AVAILABLE,
    })
