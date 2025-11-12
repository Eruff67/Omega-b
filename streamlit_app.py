

# jack_ai_super.py
# Jack.AI — Super Full (Python Streamlit version)
# Full local version: chat, OCR, PDF extraction, persona memory, mini model

import os, json, time, random
from datetime import datetime
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
import pyttsx3
import pytesseract
from PIL import Image
import fitz  # PyMuPDF
import speech_recognition as sr

# -------------------------------
# Persistent storage
# -------------------------------
DATA_DIR = "jack_data"
os.makedirs(DATA_DIR, exist_ok=True)

def load_json(name, fallback):
    path = os.path.join(DATA_DIR, f"{name}.json")
    if os.path.exists(path):
        try:
            return json.load(open(path))
        except:
            return fallback
    return fallback

def save_json(name, data):
    path = os.path.join(DATA_DIR, f"{name}.json")
    json.dump(data, open(path, "w"), indent=2)

chat = load_json("chat", [])
memories = load_json("memories", [])
examples = load_json("examples", [])
persona = load_json("persona", {"name": "Jack", "system_prompt": "You are Jack, a witty assistant."})

# -------------------------------
# Utilities
# -------------------------------
def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def debug_log(msg):
    st.session_state.setdefault("debug", "")
    st.session_state["debug"] = f"{now()} - {msg}\n" + st.session_state["debug"]

def summarize_text(text, max_sent=3):
    sentences = text.split(". ")
    if len(sentences) <= max_sent:
        return text
    vectorizer = TfidfVectorizer().fit(sentences)
    tfidf = vectorizer.transform(sentences)
    scores = tfidf.sum(axis=1).A1
    ranked = [s for _, s in sorted(zip(scores, sentences), reverse=True)]
    return ". ".join(ranked[:max_sent])

def local_reply(user_input):
    txt = user_input.lower().strip()
    if txt.startswith("/persona"):
        arg = user_input.split(" ", 1)[1:] or ["Jack"]
        persona["name"] = arg[0]
        save_json("persona", persona)
        return f"Persona updated to {arg[0]}"
    if txt == "/clear":
        chat.clear()
        save_json("chat", chat)
        return "Chat cleared."
    if txt.startswith("/learn"):
        content = user_input.split(" ", 1)[1] if " " in user_input else ""
        if content:
            memories.append({"key": f"learned_{now()}", "value": content})
            save_json("memories", memories)
            return "Learned and saved to memory."
        return "Usage: /learn <text>"
    if "time" in txt:
        return f"Current time: {now()}"
    if "joke" in txt:
        return "Why do programmers prefer dark mode? Because light attracts bugs!"
    if "summary" in txt:
        joined = " ".join([m["value"] for m in memories])
        return summarize_text(joined)
    return random.choice([
        "Probably fix X and Y. Or set it on fire.",
        "Looks messy. Might be intentional.",
        "Check your code, then breathe."
    ]) + "\n\nContext: " + user_input[:200]

# -------------------------------
# Tiny model (TF-IDF + LinearRegression)
# -------------------------------
def train_tiny_model():
    if not examples:
        return None
    X = [ex["user"] for ex in examples]
    y = [ex["assistant"] for ex in examples]
    vec = TfidfVectorizer()
    Xv = vec.fit_transform(X)
    yv = vec.transform(y)
    model = LinearRegression()
    model.fit(Xv.toarray(), yv.toarray())
    debug_log(f"Trained on {len(X)} examples.")
    return (vec, model)

tiny_model = train_tiny_model()

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config("Jack.AI — Python Edition", layout="wide")
st.title("🤖 Jack.AI — Super Full (Python Streamlit Edition)")
st.caption("Local-first • OCR • PDF • TTS/STT • Memory • TF-IDF tiny model")

# Sidebar
with st.sidebar:
    st.subheader("Persona / Memory")
    persona["name"] = st.text_input("Assistant name", persona.get("name", "Jack"))
    persona["system_prompt"] = st.text_area("System prompt", persona.get("system_prompt", "You are Jack, a witty assistant."))
    if st.button("💾 Save Persona"):
        save_json("persona", persona)
        st.success("Saved persona.")
    if st.button("🧠 Summarize Memories"):
        if memories:
            st.write(summarize_text(" ".join([m["value"] for m in memories])))
        else:
            st.info("No memories yet.")
    if st.button("🗑 Wipe All Data"):
        for f in os.listdir(DATA_DIR):
            os.remove(os.path.join(DATA_DIR, f))
        st.warning("All local data wiped.")

# Main chat
user_input = st.text_area("Ask Jack something...", "", height=120)
col1, col2, col3 = st.columns([1, 1, 3])
if col1.button("Send"):
    reply = local_reply(user_input)
    chat.append({"role": "user", "text": user_input, "time": now()})
    chat.append({"role": "assistant", "text": reply, "time": now()})
    save_json("chat", chat)
    st.session_state["last_reply"] = reply

if col2.button("Speak (TTS)"):
    if "last_reply" in st.session_state:
        tts = pyttsx3.init()
        tts.say(st.session_state["last_reply"])
        tts.runAndWait()

# Display chat
st.markdown("---")
for msg in chat[-20:]:
    role = "🧍 You" if msg["role"] == "user" else f"🤖 {persona['name']}"
    st.markdown(f"**{role}** — *{msg['time']}*  \n{msg['text']}")

# OCR / PDF Tools
st.markdown("### 📂 File Tools")
uploaded = st.file_uploader("Upload image or PDF", type=["png", "jpg", "jpeg", "pdf"])
if uploaded:
    if uploaded.type == "application/pdf":
        pdf = fitz.open(stream=uploaded.read(), filetype="pdf")
        text = ""
        for page in pdf:
            text += page.get_text()
        st.text_area("Extracted PDF Text", text, height=200)
    else:
        img = Image.open(uploaded)
        text = pytesseract.image_to_string(img)
        st.text_area("Extracted OCR Text", text, height=200)

# Debug
st.markdown("### 🪵 Debug Console")
st.text_area("Debug output", st.session_state.get("debug", ""), height=180)
