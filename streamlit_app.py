# jack_offline_ai.py
"""
Jack — Offline conversational AI (single file)
- Requires only Streamlit (pip install streamlit)
- Conversational replies, math, time/date
- Learns definitions from conversation and /define commands
- Persistent storage: ai_memory.json (conversations + learned_definitions)
- Can load an external large dictionary: dictionary.json (optional)
"""

import streamlit as st
import json
import os
import re
import math
from datetime import datetime
from typing import Dict, Any, List, Tuple

# ---------- Files ----------
MEMORY_FILE = "ai_memory.json"
DICT_FILE = "dictionary.json"  # optional large dictionary to drop into same folder

# ---------- Load / Init ----------
def load_json(path: str, default):
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return default

ai_state = load_json(MEMORY_FILE, {"conversations": [], "learned": {}})

# seed dictionary (modest seed; you may replace/add a large dictionary by saving dictionary.json)
SEED_DICTIONARY: Dict[str, Dict[str, Any]] = {
    "apple": {"definition": "A round fruit with red or green skin and a whitish interior.", "type": "noun",
              "examples": ["I ate an apple for lunch.", "Apple pie is delicious."]},
    "run": {"definition": "To move swiftly on foot.", "type": "verb",
            "examples": ["I run every morning.", "She runs faster than me."]},
    "hello": {"definition": "A greeting used when meeting someone.", "type": "interjection",
              "examples": ["Hello! How are you?", "She said hello and waved."]},
    "time": {"definition": "The ongoing progression of events measured in seconds, minutes, hours.", "type": "noun",
             "examples": ["What time is it?", "Time flies."]},
    "learn": {"definition": "To gain knowledge or skill through study or experience.", "type": "verb",
              "examples": ["I want to learn Python.", "She learned quickly."]},
    # ...seed entries; app supports loading a much larger dictionary from dictionary.json
}

# merge: priority to external dictionary if present
external_dict = load_json(DICT_FILE, None)
if external_dict and isinstance(external_dict, dict):
    DICTIONARY = {**SEED_DICTIONARY, **{k.lower():v for k,v in external_dict.items()}}
else:
    DICTIONARY = {k.lower(): v for k, v in SEED_DICTIONARY.items()}

# internal convenience: make learned definitions available in runtime dict
def merged_definitions() -> Dict[str, Dict[str, Any]]:
    d = {**DICTIONARY}
    learned = ai_state.get("learned", {})
    for k, v in learned.items():
        d[k.lower()] = {"definition": v.get("definition",""), "type": v.get("type","learned"), "examples": v.get("examples",[])}
    return d

# ---------- Utilities ----------
def save_state():
    try:
        with open(MEMORY_FILE, "w", encoding="utf-8") as f:
            json.dump(ai_state, f, ensure_ascii=False, indent=2)
    except Exception as e:
        st.error(f"Failed to save state: {e}")

def normalize_word(w: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_\-']", "", w.strip().lower())

def simple_tokenize(text: str) -> List[str]:
    return re.findall(r"\b[a-zA-Z']+\b", text.lower())

# ---------- Patterns for learning definitions from conversation ----------
# Matches like:
#   "X means Y."
#   "X is Y."
#   "X = Y"
#   "Define X: Y"
LEARN_PATTERNS = [
    re.compile(r'^\s*define\s+([^\:]+)\s*[:\-]\s*(.+)$', re.I),
    re.compile(r'^\s*([A-Za-z\'\-\s]+)\s+means\s+(.+)$', re.I),
    re.compile(r'^\s*([A-Za-z\'\-\s]+)\s+is\s+(.+)$', re.I),
    re.compile(r'^\s*([^\s=]+)\s*=\s*(.+)$', re.I),
]

def try_extract_definition(text: str) -> Tuple[str,str]:
    """Return (word, definition_text) if matched, else (None, None)."""
    s = text.strip()
    for pat in LEARN_PATTERNS:
        m = pat.match(s)
        if m:
            left = m.group(1).strip()
            right = m.group(2).strip().rstrip(".")
            # if left contains multiple words, choose first token as head word
            left_token = left.split()[0]
            return normalize_word(left_token), right
    return None, None

# ---------- Core reply generation (rule-based + retrieval) ----------
def answer_query(user_text: str) -> Dict[str, Any]:
    """
    Returns a dict with keys:
      - 'reply': str (assistant reply)
      - 'source': 'definition'|'memory'|'math'|'time'|'date'|'fallback'
      - 'confidence': 0.0..1.0
    """
    user = user_text.strip()
    lower = user.lower()

    # 1) Commands
    if lower in ("/clear", "clear memory", "wipe memory"):
        ai_state["conversations"].clear()
        ai_state["learned"].clear()
        save_state()
        return {"reply":"Memory cleared.", "source":"memory", "confidence":1.0}

    if lower.startswith("/delete "):
        # delete learned definition or conversation index
        arg = lower[len("/delete "):].strip()
        # try as int conversation index
        if arg.isdigit():
            idx = int(arg)-1
            if 0 <= idx < len(ai_state["conversations"]):
                removed = ai_state["conversations"].pop(idx)
                save_state()
                return {"reply": f"Deleted conversation #{idx+1}: {removed.get('user')}", "source":"memory", "confidence":0.9}
            else:
                return {"reply":"Invalid conversation index.", "source":"memory", "confidence":0.2}
        else:
            # delete learned definition
            key = normalize_word(arg)
            if key in ai_state.get("learned", {}):
                ai_state["learned"].pop(key)
                save_state()
                return {"reply": f"Removed learned definition for '{key}'.", "source":"memory", "confidence":0.9}
            else:
                return {"reply": f"No learned definition found for '{key}'.", "source":"memory", "confidence":0.2}

    # 2) math expression detection: allow digits and math symbols; extra safety
    math_expr = re.sub(r"[^0-9\.\+\-\*\/\%\(\)\s\^]", "", user)
    # require at least one operator and a digit
    if any(op in math_expr for op in "+-*/%") and re.search(r"\d", math_expr):
        # safe evaluate: replace ^ with **, block builtins
        try:
            expr = math_expr.replace("^", "**")
            # eval with restricted globals
            result = eval(expr, {"__builtins__": None}, {"math": math, **{k: getattr(math,k) for k in dir(math) if not k.startswith("_")}})
            return {"reply": f"Math result: {result}", "source":"math", "confidence":1.0}
        except Exception:
            # fall through to general processing
            pass

    # 3) time/date
    if re.search(r"\bwhat(?:'s| is)? the time\b|\bcurrent time\b|\btime now\b", lower):
        return {"reply": f"The current time is {datetime.now().strftime('%H:%M:%S')}", "source":"time", "confidence":1.0}
    if re.search(r"\bwhat(?:'s| is)? the date\b|\bcurrent date\b|\bdate today\b", lower):
        return {"reply": f"Today's date is {datetime.now().strftime('%Y-%m-%d')}", "source":"date", "confidence":1.0}

    # 4) explicit define command
    if lower.startswith("/define ") or lower.startswith("define "):
        # parse form: /define word: definition  OR define word: def
        rest = user.split(None,1)[1] if len(user.split(None,1))>1 else ""
        m = re.match(r'\s*([^\:]+)\s*[:\-]\s*(.+)', rest)
        if m:
            w = normalize_word(m.group(1))
            d = m.group(2).strip()
            # save learned
            ai_state.setdefault("learned", {})[w] = {"definition": d, "type":"learned", "examples": []}
            save_state()
            return {"reply": f"Learned definition for '{w}'.", "source":"learning", "confidence":1.0}
        else:
            return {"reply": "Usage: /define word: definition", "source":"learning", "confidence":0.3}

    # 5) learn from natural sentences: "X means Y", "X is Y", etc.
    w, d = try_extract_definition(user)
    if w and d:
        ai_state.setdefault("learned", {})[w] = {"definition": d, "type":"learned", "examples": []}
        save_state()
        return {"reply": f"Understood — saved definition: '{w}' = {d}", "source":"learning", "confidence":0.98}

    # 6) dictionary lookup question patterns
    # detect "what is/define/meaning of <word>"
    m = re.search(r'\bwhat(?:\'s| is)? the meaning of ([a-zA-Z\'\-]+)\b', lower) or \
        re.search(r'\bwhat(?:\'s| is)? ([a-zA-Z\'\-]+)\b', lower)  # fallback
    if m:
        key = normalize_word(m.group(1))
        defs = merged_definitions()
        if key in defs:
            entry = defs[key]
            reply = format_definition_reply(key, entry)
            return {"reply": reply, "source":"definition", "confidence":0.95}
        # else fall through

    # look for "define <word>"
    m2 = re.search(r'\bdefine\s+([a-zA-Z\'\-]+)\b', lower)
    if m2:
        key = normalize_word(m2.group(1))
        defs = merged_definitions()
        if key in defs:
            entry = defs[key]
            reply = format_definition_reply(key, entry)
            return {"reply": reply, "source":"definition", "confidence":0.95}

    # 7) direct single-word question: user types a word only -> define it
    if re.fullmatch(r"[A-Za-z'\-]+", user.strip()):
        key = normalize_word(user.strip())
        defs = merged_definitions()
        if key in defs:
            entry = defs[key]
            reply = format_definition_reply(key, entry)
            return {"reply": reply, "source":"definition", "confidence":0.9}

    # 8) retrieval from memory: try to find similar earlier user messages or learned text
    sim_reply = retrieve_from_memory_or_learned(user)
    if sim_reply:
        return {"reply": sim_reply, "source":"memory", "confidence":0.75}

    # 9) fallback conversational reply (compose an assistant-style answer)
    fallback = compose_fallback_reply(user)
    return {"reply": fallback, "source":"fallback", "confidence":0.4}

# helper to pretty format definition reply
def format_definition_reply(key: str, entry: Dict[str,Any]) -> str:
    typ = entry.get("type","")
    definition = entry.get("definition","")
    examples = entry.get("examples",[])
    ex_text = ("\nExamples:\n - " + "\n - ".join(examples)) if examples else ""
    return f"**{key}** ({typ}): {definition}{ex_text}"

# simple retrieval from memory/learned definitions using token overlap
def retrieve_from_memory_or_learned(query: str) -> str:
    qtokens = set(simple_tokenize(query))
    best_score = 0
    best_text = None
    # search past conversations (assistant replies)
    for conv in ai_state.get("conversations", []):
        text = conv.get("ai","") + " " + conv.get("user","")
        tokens = set(simple_tokenize(text))
        score = len(qtokens & tokens)
        if score > best_score:
            best_score = score
            best_text = conv.get("ai") or conv.get("user")
    # search learned definitions
    for k,v in ai_state.get("learned", {}).items():
        tokens = set(simple_tokenize(k + " " + v.get("definition","")))
        score = len(qtokens & tokens)
        if score > best_score:
            best_score = score
            best_text = f"{k}: {v.get('definition','')}"
    if best_score >= 1:
        return best_text
    return None

# compose fallback assistant-style reply (rudimentary but conversational)
def compose_fallback_reply(user: str) -> str:
    # Greedy heuristics:
    # - If question-like, answer politely asking for clarification plus possible suggestions
    if user.strip().endswith("?") or user.lower().startswith(("how","what","why","when","where","who")):
        return ("That's a good question. I don't have a full answer stored, but I can:\n"
                "- look up or learn a definition if you provide one (e.g. 'X means Y' or '/define X: Y'),\n"
                "- analyze a text file you drop in the folder, or\n"
                "- try a quick guess: " + simple_guess(user))
    # otherwise short conversational reply
    return "I hear you. Tell me more or ask me to /define a word or 'X means Y' to teach me."

def simple_guess(user: str) -> str:
    # very naive heuristics: echo back topic words
    toks = simple_tokenize(user)
    if not toks:
        return "I need more words."
    top = toks[:6]
    return " ".join(top) + " ... (I can learn more if you define terms in conversation.)"

# ---------- Sentence completion & grammar helpers ----------
COMMON_NEXT = {
    "i": ["am","think","want","will","have"],
    "you": ["are","have","should","can"],
    "we": ["are","will","should"],
    "they": ["are","have","will"],
    "the": ["same","best","next","first"]
}

def complete_sentence(text: str, max_words:int=15) -> str:
    tokens = simple_tokenize(text)
    out = tokens[:]
    for _ in range(max_words):
        last = out[-1] if out else ""
        cand = COMMON_NEXT.get(last.lower())
        if not cand:
            break
        out.append(cand[0])
    return " ".join(out)  # crude

# ---------- Chat UI (Streamlit) ----------
st.set_page_config(page_title="Jack — Offline AI", layout="wide")
st.title("Jack — Offline Conversational AI (Learns definitions)")

col1, col2 = st.columns([3,1])

with col2:
    st.subheader("Controls")
    if st.button("Clear memory"):
        ai_state["conversations"].clear()
        ai_state["learned"].clear()
        save_state()
        st.success("Cleared memory & learned definitions.")
    if st.button("Export state"):
        st.download_button("Download ai_state.json", data=json.dumps(ai_state, ensure_ascii=False, indent=2), file_name="ai_state.json")
    uploaded = st.file_uploader("Upload dictionary.json (optional)", type=["json"])
    if uploaded:
        try:
            ext = json.load(uploaded)
            if isinstance(ext, dict):
                # merge into DICTIONARY
                for k,v in ext.items():
                    DICTIONARY[k.lower()] = v
                st.success("Loaded uploaded dictionary (merged).")
            else:
                st.error("dictionary.json must be a JSON object mapping words to entries.")
        except Exception as e:
            st.error(f"Failed to load dictionary: {e}")

with col1:
    st.subheader("Chat")
    # display last N messages
    history = ai_state.get("conversations", [])
    start = max(0, len(history)-200)
    for i, msg in enumerate(history[start:], start+1):
        who = "You" if msg.get("role","user")=="user" else "Jack"
        st.markdown(f"**{who}** — {msg.get('time','')}")
        st.write(msg.get("text",""))

    user_input = st.text_input("Message (Shift+Enter for newline):", key="user_input")

    tcol1, tcol2, tcol3 = st.columns([1,1,1])
    if tcol1.button("Send"):
        ui = user_input.strip()
        if ui:
            # process
            result = answer_query(ui)
            reply = result["reply"]
            # store conversation
            ai_state.setdefault("conversations", []).append({"role":"user","text":ui,"time":datetime.now().isoformat()})
            ai_state.setdefault("conversations", []).append({"role":"assistant","text":reply,"time":datetime.now().isoformat()})
            save_state()
            st.experimental_rerun()
    if tcol2.button("Complete sentence"):
        ui = user_input.strip()
        if ui:
            comp = complete_sentence(ui)
            ai_state.setdefault("conversations", []).append({"role":"user","text":ui,"time":datetime.now().isoformat()})
            ai_state.setdefault("conversations", []).append({"role":"assistant","text":comp,"time":datetime.now().isoformat()})
            save_state()
            st.experimental_rerun()
    if tcol3.button("Teach (define)"):
        ui = user_input.strip()
        # attempt to parse "word: definition" style
        m = re.match(r'\s*([^\:]+)\s*[:\-]\s*(.+)', ui)
        if m:
            w = normalize_word(m.group(1))
            d = m.group(2).strip()
            ai_state.setdefault("learned", {})[w] = {"definition": d, "type":"learned", "examples":[]}
            save_state()
            st.success(f"Learned '{w}'.")
            st.experimental_rerun()
        else:
            st.warning("To teach, enter: word: definition  (e.g. gravity: a force that pulls)")

# bottom area: quick tools and last actions
st.markdown("---")
st.subheader("Quick Tools")
st.write("You can ask things like: `What is apple?`, `Define gravity: ...`, `X means Y`, `What time is it?`, `12 * 7`, or just a normal chat message.")
st.write("Commands: `/define word: definition`, `/delete 3` (delete conv #3), `/delete word` (remove learned def), `/clear`")
