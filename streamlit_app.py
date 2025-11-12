# jack_offline_ui.py
# Jack — Offline AI with Web Interface
# ------------------------------------
# Run: streamlit run jack_offline_ui.py

import json, os, re, math, random
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
import streamlit as st

# -------------------------
# File names
# -------------------------
STATE_FILE = "ai_state.json"
DICT_FILE = "dictionary.json"

# -------------------------
# Helpers
# -------------------------
def load_json(path, default):
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return default

def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def normalize_key(s: str) -> str:
    return re.sub(r"[^a-z0-9\s]", "", s.lower()).strip()

# -------------------------
# Persistent AI state
# -------------------------
ai_state = load_json(STATE_FILE, {"conversations": [], "learned": {}, "settings": {}, "model_meta": {}})

# -------------------------
# Minimal dictionary
# -------------------------
DICTIONARY = {
    "george washington": {
        "definition": "The first President of the United States (1789–1797).",
        "type": "proper_noun",
        "examples": ["George Washington led the Continental Army to victory."]
    },
    "abraham lincoln": {
        "definition": "The 16th President of the United States who led the nation through the Civil War.",
        "type": "proper_noun",
        "examples": ["Abraham Lincoln issued the Emancipation Proclamation."]
    },
    "paris": {"definition": "Capital of France.", "type": "noun", "examples": ["Paris is famous for the Eiffel Tower."]},
    "python": {"definition": "A high-level programming language.", "type": "noun", "examples": ["Python is popular for AI."]},
}

# -------------------------
# Knowledge base
# -------------------------
KB = {
    "who was the first president of the united states": "George Washington",
    "capital of france": "Paris",
    "what is pi": "Pi is approximately 3.14159",
    "who wrote hamlet": "William Shakespeare",
}

# -------------------------
# Tokenization
# -------------------------
WORD_RE = re.compile(r"[a-zA-Z']+")
def tokenize(t): return WORD_RE.findall(str(t).lower())

# -------------------------
# Markov generator
# -------------------------
class Markov:
    def __init__(self): self.map, self.starts = {}, []
    def train(self, text):
        toks = tokenize(text)
        if len(toks) < 3: return
        self.starts.append((toks[0], toks[1]))
        for i in range(len(toks)-2):
            key = (toks[i], toks[i+1])
            nxt = toks[i+2]
            self.map.setdefault(key, {})
            self.map[key][nxt] = self.map[key].get(nxt, 0)+1
    def generate(self, seed=None, max_words=40):
        if not self.map: return ""
        key = random.choice(self.starts)
        out = [key[0], key[1]]
        for _ in range(max_words-2):
            nxts = self.map.get((out[-2], out[-1]))
            if not nxts: break
            total = sum(nxts.values())
            r, acc = random.randint(1, total), 0
            for w, c in nxts.items():
                acc += c
                if r <= acc:
                    out.append(w); break
        return " ".join(out)

MARKOV = Markov()
for k, v in DICTIONARY.items():
    MARKOV.train(k + " " + v["definition"])
    for ex in v["examples"]: MARKOV.train(ex)
for c in ai_state.get("conversations", []):
    MARKOV.train(c.get("text", ""))

# -------------------------
# Math and Q/A
# -------------------------
def safe_eval_math(expr):
    try:
        filtered = re.sub(r"[^0-9\.\+\-\*\/\%\(\)\s\^]", "", expr)
        if not re.search(r"\d", filtered): return None
        filtered = filtered.replace("^", "**")
        return eval(filtered, {"__builtins__": None}, {"math": math})
    except Exception:
        return None

def lookup_fact(q):
    qn = normalize_key(q.strip("? "))
    if qn in KB: return KB[qn]
    for k, v in KB.items():
        if k in qn: return v
    return None

# -------------------------
# Learn new definitions
# -------------------------
def try_learn(text):
    m = re.match(r"^\s*([A-Za-z' \-]+)\s+means\s+(.+)$", text.strip(), re.I)
    if not m: return None
    word, meaning = normalize_key(m.group(1)), m.group(2).strip()
    ai_state["learned"][word] = {"definition": meaning, "type": "learned", "examples": []}
    save_json(STATE_FILE, ai_state)
    return f"Learned: '{word}' = {meaning}"

# -------------------------
# Core reply
# -------------------------
def compose_reply(user_text):
    t = user_text.strip()
    low = t.lower()

    # math
    r = safe_eval_math(t)
    if r is not None: return f"Math result: {r}"

    # time/date
    if "time" in low: return f"The current time is {datetime.now().strftime('%H:%M:%S')}."
    if "date" in low: return f"Today's date is {datetime.now().strftime('%Y-%m-%d')}."

    # teaching
    taught = try_learn(t)
    if taught: return taught

    # definition
    if low.startswith("define "):
        key = normalize_key(t[7:])
        if key in DICTIONARY: return DICTIONARY[key]["definition"]
        if key in ai_state["learned"]: return ai_state["learned"][key]["definition"]
        return f"I don't know '{key}' yet."

    # facts
    fact = lookup_fact(low)
    if fact: return fact

    # learned
    for k, v in ai_state["learned"].items():
        if k in low: return f"{k}: {v['definition']}"

    # markov
    return MARKOV.generate(seed=low) or "I don't know that yet."

# -------------------------
# Streamlit Interface
# -------------------------
st.set_page_config(page_title="Jack Offline AI", page_icon="🤖", layout="centered")
st.title("🤖 Jack — Offline AI")

# Sidebar controls
with st.sidebar:
    st.subheader("🧠 Memory & Settings")
    if st.button("Clear Chat"):
        ai_state["conversations"].clear()
        save_json(STATE_FILE, ai_state)
        st.success("Chat cleared.")
    if st.button("Forget Learned"):
        ai_state["learned"].clear()
        save_json(STATE_FILE, ai_state)
        st.success("Forgot all learned definitions.")
    st.markdown("---")
    st.markdown("**Learned Words:**")
    for k,v in ai_state.get("learned", {}).items():
        st.write(f"• {k}: {v.get('definition','')}")

# Chat area
if "chat_log" not in st.session_state:
    st.session_state.chat_log = ai_state.get("conversations", [])

for msg in st.session_state.chat_log:
    role, text = msg["role"], msg["text"]
    if role == "user":
        st.chat_message("user").write(text)
    else:
        st.chat_message("assistant").write(text)

user_input = st.chat_input("Ask something or teach me...")

if user_input:
    st.chat_message("user").write(user_input)
    reply = compose_reply(user_input)
    st.chat_message("assistant").write(reply)
    # save conversation
    ai_state.setdefault("conversations", []).append({"role": "user", "text": user_input})
    ai_state.setdefault("conversations", []).append({"role": "assistant", "text": reply})
    save_json(STATE_FILE, ai_state)
    st.session_state.chat_log.append({"role": "user", "text": user_input})
    st.session_state.chat_log.append({"role": "assistant", "text": reply})

