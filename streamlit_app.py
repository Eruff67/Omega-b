# jack_offline_saved_memory.py
# Jack — Offline AI with persistent memory, file ingest, and Streamlit UI
# Requires only Streamlit.
# Run:
#   pip install streamlit
#   streamlit run jack_offline_saved_memory.py

import streamlit as st
import json
import os
import re
import math
import random
from datetime import datetime
from typing import List, Dict, Tuple, Any, Optional

# -------------------------
# Files & Persistence
# -------------------------
STATE_FILE = "ai_state.json"   # persistent state (conversations, learned, settings)
DICT_FILE = "dictionary.json"  # optional external dictionary to merge

def load_json(path: str, default):
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return default

def save_json(path: str, data):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print("Failed saving", path, e)

# load or init state
ai_state = load_json(STATE_FILE, {"conversations": [], "learned": {}, "settings": {}, "model_meta": {}})

# -------------------------
# Small embedded dictionary (templated core set)
# -------------------------
# You can later upload/merge a richer dictionary.json via UI
BASE_DICT = {
    "george washington": {"definition":"The first President of the United States (1789–1797).","type":"proper_noun","examples":["George Washington led the Continental Army."]},
    "abraham lincoln": {"definition":"16th U.S. President, led during the Civil War.","type":"proper_noun","examples":["Abraham Lincoln issued the Emancipation Proclamation."]},
    "paris": {"definition":"Capital of France.","type":"proper_noun","examples":["Paris is known for the Eiffel Tower."]},
    "python": {"definition":"High-level programming language popular for scripting and data science.","type":"noun","examples":["I used Python to write that script."]},
    "pi": {"definition":"Mathematical constant π ≈ 3.14159.","type":"number","examples":["Pi is used to compute circle properties."]}
}

# merge external dictionary file if present at startup
if os.path.exists(DICT_FILE):
    ext = load_json(DICT_FILE, {})
    if isinstance(ext, dict):
        for k,v in ext.items():
            BASE_DICT[k.lower()] = v

def merged_dictionary() -> Dict[str, Dict[str,Any]]:
    """Combined base dictionary + learned items."""
    d = {k.lower(): dict(v) for k,v in BASE_DICT.items()}
    for k,v in ai_state.get("learned", {}).items():
        d[k.lower()] = {"definition": v.get("definition",""), "type": v.get("type","learned"), "examples": v.get("examples",[])}
    return d

# -------------------------
# Knowledge base for quick facts
# -------------------------
KB: Dict[str,str] = {
    "who was the first president of the united states": "George Washington",
    "who was the first president": "George Washington",
    "capital of france": "Paris",
    "largest planet": "Jupiter",
    "what is pi": "Pi is approximately 3.14159",
    "who wrote hamlet": "William Shakespeare",
}

# -------------------------
# Tokenization & vocab
# -------------------------
WORD_RE = re.compile(r"[a-zA-Z']+")

def tokenize(text: str) -> List[str]:
    return WORD_RE.findall((text or "").lower())

def build_vocab() -> List[str]:
    vocab = set()
    md = merged_dictionary()
    for k,v in md.items():
        vocab.update(tokenize(k))
        vocab.update(tokenize(v.get("definition","")))
        for ex in v.get("examples",[]):
            vocab.update(tokenize(ex))
    for c in ai_state.get("conversations", [])[-500:]:
        vocab.update(tokenize(c.get("text","")))
    vocab.update(["what","who","when","where","why","how","define","means","calculate","time","date"])
    return sorted(vocab)

def text_to_vector(text: str, vocab_list: List[str]) -> List[float]:
    toks = tokenize(text)
    vec = [0.0]*len(vocab_list)
    idx = {w:i for i,w in enumerate(vocab_list)}
    for t in toks:
        if t in idx:
            vec[idx[t]] += 1.0
    norm = math.sqrt(sum(x*x for x in vec)) or 1.0
    return [x/norm for x in vec]

# -------------------------
# TinyNN (from-scratch) for intent classification
# -------------------------
def random_matrix(rows, cols, scale=0.1):
    return [[(random.random()*2-1)*scale for _ in range(cols)] for _ in range(rows)]

def matvec(M, v):
    return [sum(M[i][j]*v[j] for j in range(len(v))) for i in range(len(M))]

def add_vec(a,b):
    return [a[i]+b[i] for i in range(len(a))]

def tanh_vec(v):
    return [math.tanh(x) for x in v]

def softmax(v):
    mx = max(v)
    exps = [math.exp(x-mx) for x in v]
    s = sum(exps) or 1.0
    return [e/s for e in exps]

class TinyNN:
    def __init__(self, input_dim:int, hidden_dim:int, output_dim:int):
        self.in_dim = input_dim
        self.h_dim = hidden_dim
        self.out_dim = output_dim
        self.W1 = random_matrix(hidden_dim, input_dim, scale=0.25)
        self.b1 = [0.0]*hidden_dim
        self.W2 = random_matrix(output_dim, hidden_dim, scale=0.25)
        self.b2 = [0.0]*output_dim

    def forward(self, x: List[float]) -> Tuple[List[float], List[float]]:
        h_in = add_vec(matvec(self.W1, x), self.b1)
        h = tanh_vec(h_in)
        o_in = add_vec(matvec(self.W2, h), self.b2)
        out = softmax(o_in)
        return h, out

    def predict(self, x: List[float]) -> int:
        _, out = self.forward(x)
        return max(range(len(out)), key=lambda i: out[i])

    def train(self, dataset: List[Tuple[List[float], int]], epochs:int=40, lr:float=0.05):
        if not dataset: return
        for _ in range(epochs):
            random.shuffle(dataset)
            for x_vec, label in dataset:
                h_in = add_vec(matvec(self.W1, x_vec), self.b1)
                h = tanh_vec(h_in)
                o_in = add_vec(matvec(self.W2, h), self.b2)
                out = softmax(o_in)
                # one-hot target
                y = [0.0]*len(out); y[label] = 1.0
                err_out = [out[i] - y[i] for i in range(len(out))]
                # W2, b2 update
                for i in range(len(self.W2)):
                    for j in range(len(self.W2[0])):
                        self.W2[i][j] -= lr * err_out[i] * h[j]
                    self.b2[i] -= lr * err_out[i]
                # hidden error
                err_hidden = [0.0]*len(h)
                for j in range(len(h)):
                    s = 0.0
                    for i in range(len(err_out)):
                        s += self.W2[i][j] * err_out[i]
                    err_hidden[j] = s * (1.0 - h[j]*h[j])
                # W1, b1 update
                for j in range(len(self.W1)):
                    for k in range(len(self.W1[0])):
                        self.W1[j][k] -= lr * err_hidden[j] * x_vec[k]
                    self.b1[j] -= lr * err_hidden[j]

# -------------------------
# Intents / seed examples
# -------------------------
INTENTS = ["define","fact","math","time","date","teach","chat"]
SEED_EXAMPLES = [
    ("what is gravity", "fact"),
    ("who was the first president of the united states", "fact"),
    ("define gravity", "define"),
    ("what is the meaning of gravity", "define"),
    ("calculate 12 * 7", "math"),
    ("what time is it", "time"),
    ("what is today's date", "date"),
    ("x means y", "teach"),
    ("gravity means a force", "teach"),
    ("hello how are you", "chat"),
    ("tell me a story", "chat"),
    ("who was abraham lincoln", "fact"),
    ("define python", "define"),
]

def build_training(vocab: List[str]) -> List[Tuple[List[float], int]]:
    data = []
    for text,intent in SEED_EXAMPLES:
        data.append((text_to_vector(text, vocab), INTENTS.index(intent)))
    for k,v in ai_state.get("learned", {}).items():
        phrase = f"{k} means {v.get('definition','')}"
        data.append((text_to_vector(phrase, vocab), INTENTS.index("teach")))
    return data

# -------------------------
# Markov fallback generator (greedy next-word continuation)
# -------------------------
class Markov:
    def __init__(self):
        # map: (w1,w2) -> { next_word: count, ... }
        self.map: Dict[Tuple[str,str], Dict[str,int]] = {}
        # list of starting bigrams observed
        self.starts: List[Tuple[str,str]] = []

    def train(self, text: str):
        toks = tokenize(text)
        if len(toks) < 3:
            return
        self.starts.append((toks[0].lower(), toks[1].lower()))
        for i in range(len(toks)-2):
            key = (toks[i].lower(), toks[i+1].lower())
            nxt = toks[i+2].lower()
            self.map.setdefault(key, {})
            self.map[key][nxt] = self.map[key].get(nxt, 0) + 1

    def _best_choice(self, choices: Dict[str,int]) -> Optional[str]:
        """Return the most likely next token (argmax)."""
        if not choices:
            return None
        # pick token with max count; tie-break deterministically by sorted order
        best = max(sorted(choices.items()), key=lambda kv: kv[1])
        return best[0]

    def _best_unigram_after(self, token: str) -> Optional[str]:
        """Backoff: look for bigrams that start with token and choose most common follower overall."""
        candidates: Dict[str,int] = {}
        for (a,b), nxts in self.map.items():
            if a == token:
                for w,cnt in nxts.items():
                    candidates[w] = candidates.get(w,0) + cnt
        if not candidates:
            return None
        return self._best_choice(candidates)

    def generate(self, seed: str=None, max_words:int=40) -> str:
        """
        Deterministic greedy generation:
         - If seed has >=2 tokens and we have a matching bigram, greedily pick the most likely next word
           and repeat using the last two words each step, returning only the NEW words (continuation).
         - Back off to a unigram-after-last-token heuristic if exact bigram missing.
         - If no seed or nothing found, generate a full sentence starting from a random observed start.
        """
        if seed:
            toks = tokenize(seed)
            if len(toks) >= 2:
                # start from the last two tokens of the seed (lowercased)
                key = (toks[-2].lower(), toks[-1].lower())
                continuation: List[str] = []
                # greedy loop
                for _ in range(max_words):
                    # try exact bigram
                    if key in self.map:
                        nxt = self._best_choice(self.map[key])
                    else:
                        # backoff: try best unigram after the last token
                        nxt = self._best_unigram_after(key[1])
                    if not nxt:
                        break
                    continuation.append(nxt)
                    # shift window
                    key = (key[1], nxt)
                    # small safety: avoid infinite loops when model repeats the same word forever
                    if len(continuation) >= 2 and continuation[-1] == continuation[-2]:
                        break
                # return only new words (don't repeat the seed)
                if continuation:
                    return " ".join(continuation)
                # else fall back to trying to generate from any starting bigram below
        # No seed or couldn't continue: generate full sentence using greedy choices from a start bigram
        if not self.starts:
            return ""
        key = random.choice(self.starts)
        out = [key[0], key[1]]
        for _ in range(max_words-2):
            choices = self.map.get((out[-2], out[-1]))
            if not choices:
                break
            nxt = self._best_choice(choices)
            if not nxt:
                break
            out.append(nxt)
            # stop if repetitive
            if len(out) >= 3 and out[-1] == out[-2] == out[-3]:
                break
        return " ".join(out)

MARKOV = Markov()

def train_markov():
    MARKOV.map.clear(); MARKOV.starts.clear()
    md = merged_dictionary()
    for k,v in md.items():
        for ex in v.get("examples", []): MARKOV.train(ex)
        MARKOV.train(k + " " + v.get("definition",""))
    for c in ai_state.get("conversations", []):
        MARKOV.train(c.get("text",""))

# initial markov content
train_markov()

# -------------------------
# Retrieval and helpers
# -------------------------
LEARN_PATTERNS = [
    re.compile(r'^\s*define\s+([^\:]+)\s*[:\-]\s*(.+)$', re.I),
    re.compile(r'^\s*([A-Za-z\'\-\s]+)\s+means\s+(.+)$', re.I),
    re.compile(r'^\s*([A-Za-z\'\-\s]+)\s+is\s+(.+)$', re.I),
    re.compile(r'^\s*([^\s=]+)\s*=\s*(.+)$', re.I),
]

def normalize_key(s: str) -> str:
    return re.sub(r"[^a-z0-9\s]", "", s.lower()).strip()

def try_extract_definition(text: str) -> Tuple[Optional[str], Optional[str]]:
    s = text.strip()
    for pat in LEARN_PATTERNS:
        m = pat.match(s)
        if m:
            left = m.group(1).strip(); right = m.group(2).strip().rstrip(".")
            left_token = left.split()[0]
            return normalize_key(left_token), right
    return None, None

def retrieve_from_memory_or_learned(query: str) -> Optional[str]:
    qtokens = set(tokenize(query))
    best_score = 0; best_text = None
    for conv in ai_state.get("conversations", []):
        t = conv.get("text","")
        sc = len(qtokens & set(tokenize(t)))
        if sc > best_score:
            best_score = sc; best_text = t
    for k,v in ai_state.get("learned", {}).items():
        sc = len(qtokens & set(tokenize(k + " " + v.get("definition",""))))
        if sc > best_score:
            best_score = sc; best_text = f"{k}: {v.get('definition','')}"
    if best_score >= 1:
        return best_text
    return None

def lookup_kb(query: str) -> Tuple[Optional[str], float]:
    q = normalize_key(query.strip("? "))
    if q in KB: return KB[q], 0.95
    qtokens = set(tokenize(q))
    best = None; best_score = 0
    for k,v in KB.items():
        sc = len(qtokens & set(tokenize(k)))
        if sc > best_score:
            best_score = sc; best = v
    if best_score >= 1: return best, 0.7
    for k,v in ai_state.get("learned", {}).items():
        if normalize_key(k) in q or normalize_key(q) in k:
            return v.get("definition",""), 0.85
    return None, 0.0

# -------------------------
# Build & train TinyNN model (initial heavy train)
# -------------------------
def build_and_train_model():
    global VOCAB, NN_MODEL
    VOCAB = build_vocab()
    NN_MODEL = TinyNN(len(VOCAB), max(32, len(VOCAB)//8 or 16), len(INTENTS))
    dataset = build_training(VOCAB)
    if dataset:
        NN_MODEL.train(dataset, epochs=100, lr=0.06)

VOCAB: List[str] = []
NN_MODEL: Optional[TinyNN] = None
build_and_train_model()

def incremental_retrain():
    build_and_train_model()

# -------------------------
# Compose reply core
# -------------------------
def format_definition(key: str, entry: Dict[str,Any]) -> str:
    ex = entry.get("examples", [])
    ex_text = ("\nExamples:\n - " + "\n - ".join(ex)) if ex else ""
    return f"{key} ({entry.get('type','')}): {entry.get('definition','')}{ex_text}"

def safe_eval_math(expr: str):
    try:
        filtered = re.sub(r"[^0-9\.\+\-\*\/\%\(\)\s\^]", "", expr)
        if not re.search(r"\d", filtered): return None
        filtered = filtered.replace("^", "**")
        result = eval(filtered, {"__builtins__": None}, {"math": math, **{k:getattr(math,k) for k in dir(math) if not k.startswith("_")}})
        return result
    except Exception:
        return None

def compose_reply(user_text: str) -> Dict[str,Any]:
    user = user_text.strip()
    lower = user.lower()

    # commands
    if lower in ("/clear", "clear chat"):
        ai_state["conversations"].clear(); save_json(STATE_FILE, ai_state); train_markov(); return {"reply":"Chat cleared.","meta":{"intent":"memory"}}
    if lower in ("/forget", "forget"):
        ai_state["learned"].clear(); save_json(STATE_FILE, ai_state); incremental_retrain(); train_markov(); return {"reply":"Learned memory cleared.","meta":{"intent":"memory"}}
    if lower.startswith("/delete "):
        arg = lower[len("/delete "):].strip()
        if arg.isdigit():
            idx = int(arg)-1
            if 0 <= idx < len(ai_state.get("conversations", [])):
                removed = ai_state["conversations"].pop(idx)
                save_json(STATE_FILE, ai_state)
                return {"reply": f"Deleted conversation #{idx+1}: {removed.get('text')}", "meta":{"intent":"memory"}}
            else:
                return {"reply":"Invalid conversation index.","meta":{"intent":"error"}}
        else:
            key = normalize_key(arg)
            if key in ai_state.get("learned", {}):
                ai_state["learned"].pop(key); save_json(STATE_FILE, ai_state); incremental_retrain(); train_markov(); return {"reply": f"Removed learned definition for '{key}'.", "meta":{"intent":"memory"}}
            else:
                return {"reply": f"No learned definition for '{key}'.", "meta":{"intent":"error"}}

    # safe math
    math_res = safe_eval_math(user)
    if math_res is not None:
        return {"reply": f"Math result: {math_res}", "meta":{"intent":"math"}}

    # time/date
    if re.search(r"\bwhat(?:'s| is)? the time\b|\btime now\b|\bcurrent time\b", lower):
        return {"reply": f"The current time is {datetime.now().strftime('%H:%M:%S')}", "meta":{"intent":"time"}}
    if re.search(r"\bwhat(?:'s| is)? the date\b|\bcurrent date\b|\bdate today\b", lower):
        return {"reply": f"Today's date is {datetime.now().strftime('%Y-%m-%d')}", "meta":{"intent":"date"}}

    # explicit define command
    if lower.startswith("/define ") or lower.startswith("define "):
        rest = user.split(None,1)[1] if len(user.split(None,1))>1 else ""
        m = re.match(r'\s*([^\:]+)\s*[:\-]\s*(.+)', rest)
        if m:
            w = normalize_key(m.group(1)); d = m.group(2).strip()
            ai_state.setdefault("learned", {})[w] = {"definition": d, "type":"learned", "examples": []}
            save_json(STATE_FILE, ai_state)
            incremental_retrain(); train_markov()
            return {"reply": f"Learned definition for '{w}'.", "meta":{"intent":"learning"}}
        m2 = re.match(r'\s*([A-Za-z\'\- ]+)\s*$', rest)
        if m2:
            key = normalize_key(m2.group(1))
            defs = merged_dictionary()
            if key in defs:
                return {"reply": format_definition(key, defs[key]), "meta":{"intent":"definition"}}
            else:
                return {"reply": f"No definition for '{key}'. Use '/define {key}: <meaning>' to teach me.", "meta":{"intent":"definition"}}
        return {"reply":"Usage: /define word: definition", "meta":{"intent":"define"}}

    # natural teaching patterns like "X means Y"
    w,d = try_extract_definition(user)
    if w and d:
        ai_state.setdefault("learned", {})[w] = {"definition": d, "type":"learned", "examples": []}
        save_json(STATE_FILE, ai_state)
        incremental_retrain(); train_markov()
        return {"reply": f"Saved learned definition: '{w}' = {d}", "meta":{"intent":"learning"}}

    # classification via TinyNN (if available)
    if VOCAB and NN_MODEL:
        xvec = text_to_vector(user, VOCAB)
        intent_idx = NN_MODEL.predict(xvec)
        intent = INTENTS[intent_idx]
    else:
        intent = "chat"

    # intent-driven replies
    if intent == "fact":
        ans, conf = lookup_kb(user)
        if ans:
            return {"reply": str(ans), "meta":{"intent":"fact","confidence":conf}}
    if intent == "define":
        key = normalize_key(user)
        defs = merged_dictionary()
        if key in defs:
            return {"reply": format_definition(key, defs[key]), "meta":{"intent":"definition"}}
        m = re.search(r'\bmeaning of ([a-zA-Z\'\- ]+)\b', lower)
        if m:
            k = normalize_key(m.group(1))
            if k in defs:
                return {"reply": format_definition(k, defs[k]), "meta":{"intent":"definition"}}
        return {"reply": "I don't have that definition yet. Teach me with '/define word: definition' or 'X means Y'.", "meta":{"intent":"definition"}}
    if intent == "time":
        return {"reply": f"The current time is {datetime.now().strftime('%H:%M:%S')}", "meta":{"intent":"time"}}
    if intent == "date":
        return {"reply": f"Today's date is {datetime.now().strftime('%Y-%m-%d')}", "meta":{"intent":"date"}}
    if intent == "math":
        if math_res is not None:
            return {"reply": f"Math result: {math_res}", "meta":{"intent":"math"}}

    # retrieval from memories or learned
    mem = retrieve_from_memory_or_learned(user)
    if mem:
        return {"reply": mem, "meta":{"intent":"memory"}}

    # Markov generative fallback (greedy continuation)
    gen = MARKOV.generate(seed=user, max_words=50)
    if gen:
        if gen.strip():
            if user and user.strip()[-1] in ".!?":
                reply_text = gen.capitalize() + "."
            else:
                reply_text = (user.rstrip() + " " + gen).strip()
            return {"reply": reply_text, "meta":{"intent":"gen"}}

    return {"reply": "I don't know that yet. Teach me with 'X means Y' or '/define X: Y'.", "meta":{"intent":"unknown"}}

# -------------------------
# File ingestion (txt/json)
# -------------------------
def ingest_text_content(name: str, text: str, save_as_memory: bool=True):
    """Add uploaded text into learned memory or conversations. If save_as_memory True, store under learned."""
    if not text:
        return "No content."
    # simple split into paragraphs and add each as a memory entry
    parts = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    added = 0
    for p in parts:
        key = f"ingest_{name}_{added}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        if save_as_memory:
            ai_state.setdefault("learned", {})[key] = {"definition": p[:200], "type":"ingest", "examples":[p[:200]]}
        else:
            ai_state.setdefault("conversations", []).append({"role":"assistant","text":p,"time":datetime.now().isoformat()})
        added += 1
    save_json(STATE_FILE, ai_state)
    incremental_retrain(); train_markov()
    return f"Ingested {added} blocks from {name}."

# -------------------------
# UI: Streamlit Chat & Controls
# -------------------------
st.set_page_config(page_title="Jack — Offline AI (Saved Memory)", layout="wide")
st.title("Jack — Offline AI (Persistent Memory)")

left, right = st.columns([3,1])

with right:
    st.header("Memory Controls")
    if st.button("Clear Conversation"):
        ai_state["conversations"].clear()
        save_json(STATE_FILE, ai_state)
        st.success("Conversation cleared.")
        st.rerun()
    if st.button("Forget Learned Memories"):
        ai_state["learned"].clear()
        save_json(STATE_FILE, ai_state)
        incremental_retrain(); train_markov()
        st.success("All learned memories forgotten.")
        st.rerun()
    st.markdown("---")
    st.markdown("**Manage Learned**")
    # list learned items with delete buttons
    learned_keys = list(ai_state.get("learned", {}).keys())
    if learned_keys:
        for k in learned_keys:
            colk1, colk2 = st.columns([5,1])
            with colk1:
                st.write(f"• **{k}** — {ai_state['learned'][k].get('definition','')[:180]}")
            with colk2:
                if st.button(f"Delete {k}", key=f"del_{k}"):
                    ai_state["learned"].pop(k, None)
                    save_json(STATE_FILE, ai_state)
                    incremental_retrain(); train_markov()
                    st.rerun()
    else:
        st.write("_No learned items yet._")

    st.markdown("---")
    st.write("Upload text or JSON to ingest (encyclopedia articles, notes).")
    uploaded = st.file_uploader("Upload .txt or .json", type=["txt","json"])
    if uploaded:
        try:
            raw = uploaded.read().decode("utf-8")
            if uploaded.name.lower().endswith(".json"):
                data = json.loads(raw)
                # if it's an object with text fields, flatten: else store raw
                if isinstance(data, dict):
                    text = "\n\n".join(str(v) for v in data.values())
                elif isinstance(data, list):
                    text = "\n\n".join(str(i) for i in data)
                else:
                    text = str(data)
            else:
                text = raw
            save_as_memory = st.checkbox("Save as learned memory (recommended)", value=True)
            if st.button("Ingest file"):
                msg = ingest_text_content(uploaded.name, text, save_as_memory=save_as_memory)
                st.success(msg)
                #st.rerun()
        except Exception as e:
            st.error(f"Failed to read uploaded file: {e}")

    st.markdown("---")
    st.write("State export / import")
    if st.button("Export ai_state.json"):
        st.download_button("Download ai_state.json", data=json.dumps(ai_state, ensure_ascii=False, indent=2), file_name="ai_state.json")
    import_file = st.file_uploader("Import ai_state.json (merge)", type=["json"], key="import_state")
    if import_file:
        try:
            payload = json.loads(import_file.read().decode("utf-8"))
            if isinstance(payload, dict):
                # merge learned and conversations
                for k,v in payload.get("learned", {}).items():
                    ai_state.setdefault("learned", {})[k] = v
                for c in payload.get("conversations", []):
                    ai_state.setdefault("conversations", []).append(c)
                save_json(STATE_FILE, ai_state)
                incremental_retrain(); train_markov()
                st.success("Merged imported state.")
               # st.rerun()
            else:
                st.error("Imported file not in expected format.")
        except Exception as e:
            st.error(f"Import failed: {e}")

with left:
    st.subheader("Conversation")
    # show last messages
    history = ai_state.get("conversations", [])[-500:]
    if "chat_index" not in st.session_state:
        st.session_state.chat_index = len(history)
    for m in history:
        who = "You" if m.get("role","user")=="user" else "Jack"
        t = m.get("time","")
        st.markdown(f"**{who}**  <span style='color:gray;font-size:12px'>{t}</span>", unsafe_allow_html=True)
        st.write(m.get("text",""))

    st.markdown("---")
    user_input = st.text_area("Message (Shift+Enter = newline)", height=120)
    c1,c2,c3 = st.columns([1,1,1])
    if c1.button("Send"):
        ui = user_input.strip()
        if ui:
            out = compose_reply(ui)
            reply = out.get("reply","")
            ai_state.setdefault("conversations", []).append({"role":"user","text":ui,"time":datetime.now().isoformat()})
            ai_state.setdefault("conversations", []).append({"role":"assistant","text":reply,"time":datetime.now().isoformat()})
            save_json(STATE_FILE, ai_state)
            incremental_retrain(); train_markov()
            st.rerun()
    if c2.button("Complete"):
        ui = user_input.rstrip()
        if ui:
            # improved deterministic completion: greedy next-word continuation (chooses best-fitting next word each step)
            cont = MARKOV.generate(seed=ui, max_words=40)
            if cont:
                if ui and ui.strip()[-1] in ".!?":
                    final = ui.rstrip() + " " + cont.capitalize()
                else:
                    final = (ui + " " + cont).strip()
                ai_state.setdefault("conversations", []).append({"role":"user","text":ui,"time":datetime.now().isoformat()})
                ai_state.setdefault("conversations", []).append({"role":"assistant","text":final,"time":datetime.now().isoformat()})
            else:
                gen = MARKOV.generate(max_words=40)
                ai_state.setdefault("conversations", []).append({"role":"user","text":ui,"time":datetime.now().isoformat()})
                ai_state.setdefault("conversations", []).append({"role":"assistant","text":gen,"time":datetime.now().isoformat()})
            save_json(STATE_FILE, ai_state)
            st.rerun()
    if c3.button("Teach (word: definition)"):
        ui = user_input.strip()
        m = re.match(r'\s*([^\:]+)\s*[:\-]\s*(.+)', ui)
        if m:
            w = normalize_key(m.group(1)); d = m.group(2).strip()
            ai_state.setdefault("learned", {})[w] = {"definition": d, "type":"learned", "examples": []}
            save_json(STATE_FILE, ai_state)
            incremental_retrain(); train_markov()
            st.success(f"Learned '{w}'.")
            st.rerun()
        else:
            st.warning("To teach: enter `word: definition` (e.g. gravity: a force that pulls)")

st.markdown("---")
st.markdown("**Examples / Commands**")
st.markdown("""
- Ask a fact: `Who was the first president of the U.S.?`  
- Teach: `gravity means a force that pulls` or `/define gravity: a force that pulls`  
- Math: `12 * (3 + 4)`  
- Time/Date: `what is the time?` or `what is the date?`  
- Commands: `/clear` (clear conversation), `/forget` (clear learned memories), `/delete N` (delete conversation #N)
""")
