# jack_offline_ai.py
# Jack — Offline generative AI with a simple from-scratch ML intent classifier
# Requires only: streamlit
# Run: pip install streamlit
#      streamlit run jack_offline_ai.py

import streamlit as st
import json
import os
import re
import math
import random
from datetime import datetime
from typing import List, Dict, Tuple, Any

# -------------------------
# Files & persistence
# -------------------------
STATE_FILE = "ai_state.json"
DICT_FILE = "dictionary.json"  # optional large dictionary to drop into app folder

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

# load state
ai_state = load_json(STATE_FILE, {"conversations": [], "learned": {}, "settings": {}})

# -------------------------
# Seed dictionary (expandable)
# -------------------------
SEED_DICT = {
    "george washington": {"definition":"The first President of the United States (1789–1797).", "type":"proper_noun", "examples":["George Washington led the Continental Army."]},
    "abraham lincoln": {"definition":"16th President of the United States who led during the Civil War.", "type":"proper_noun", "examples":["Lincoln issued the Emancipation Proclamation."]},
    "apple": {"definition":"A round fruit with red or green skin.", "type":"noun", "examples":["I ate an apple."]},
    "python": {"definition":"A high-level programming language.", "type":"noun", "examples":["I wrote a script in Python."]},
    "time": {"definition":"The ongoing sequence of events.", "type":"noun", "examples":["What is the time?"],},
    "date": {"definition":"A particular day of a month or year.", "type":"noun", "examples":["What is the date?"],},
    "run": {"definition":"To move swiftly on foot.", "type":"verb", "examples":["I run every morning."]},
    "learn": {"definition":"To gain knowledge or skill.", "type":"verb", "examples":["I want to learn."]},
    # many more can be merged by uploading dictionary.json
}

# merge external dictionary if present
external = load_json(DICT_FILE, None)
if external and isinstance(external, dict):
    DICTIONARY = {**{k.lower():v for k,v in SEED_DICT.items()}, **{k.lower():v for k,v in external.items()}}
else:
    DICTIONARY = {k.lower():v for k,v in SEED_DICT.items()}

# function to get merged dict including learned
def merged_dictionary():
    d = {**DICTIONARY}
    learned = ai_state.get("learned", {})
    for k,v in learned.items():
        d[k.lower()] = {"definition": v.get("definition",""), "type": v.get("type","learned"), "examples": v.get("examples",[])}
    return d

# -------------------------
# Simple tokenizer & vectorizer (bag-of-words)
# -------------------------
WORD_RE = re.compile(r"[a-zA-Z']+")

def tokenize(text: str) -> List[str]:
    return WORD_RE.findall(text.lower())

def build_vocab() -> List[str]:
    # Build vocabulary from dictionary keys, examples, and recent conversation
    vocab = set()
    # include dictionary words and their tokens
    for k,v in merged_dictionary().items():
        vocab.update(tokenize(k))
        for ex in v.get("examples",[]):
            vocab.update(tokenize(ex))
        vocab.update(tokenize(v.get("definition","")))
    # include conversation tokens
    for conv in ai_state.get("conversations",[])[-500:]:
        vocab.update(tokenize(conv.get("text","")))
    # ensure some common words exist
    common = ["what","is","who","when","where","why","how","define","means","calculate","time","date"]
    vocab.update(common)
    return sorted(list(vocab))

# Bag-of-words vector
def text_to_vector(text: str, vocab_list: List[str]) -> List[float]:
    toks = tokenize(text)
    vec = [0.0]*len(vocab_list)
    # simple term frequency
    for t in toks:
        try:
            idx = vocab_list.index(t)
            vec[idx] += 1.0
        except ValueError:
            pass
    # normalize by length
    n = math.sqrt(sum(x*x for x in vec)) or 1.0
    return [x/n for x in vec]

# -------------------------
# Simple feedforward neural network (from scratch)
# - single hidden layer
# - uses lists for vectors/matrices
# -------------------------
def zeros_matrix(rows, cols):
    return [[0.0]*cols for _ in range(rows)]

def random_matrix(rows, cols, scale=0.1):
    return [[(random.random()*2-1)*scale for _ in range(cols)] for _ in range(rows)]

def matvec_mul(M, v):
    return [sum(M[i][j]*v[j] for j in range(len(v))) for i in range(len(M))]

def vec_add(a,b):
    return [a[i]+b[i] for i in range(len(a))]

def tanh_vec(v):
    return [math.tanh(x) for x in v]

def tanh_derivative_vec(v):
    return [1.0 - math.tanh(x)**2 for x in v]

def softmax(v):
    maxv = max(v)
    exps = [math.exp(x - maxv) for x in v]
    s = sum(exps) or 1.0
    return [e/s for e in exps]

class SimpleNN:
    def __init__(self, input_dim:int, hidden_dim:int, output_dim:int):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        # weights
        self.W1 = random_matrix(hidden_dim, input_dim, scale=0.2)  # hidden x input
        self.b1 = [0.0]*hidden_dim
        self.W2 = random_matrix(output_dim, hidden_dim, scale=0.2) # out x hidden
        self.b2 = [0.0]*output_dim

    def forward(self, x: List[float]) -> Tuple[List[float], List[float]]:
        # hidden = tanh(W1 * x + b1)
        h_in = vec_add(matvec_mul(self.W1, x), self.b1)
        h = tanh_vec(h_in)
        o_in = vec_add(matvec_mul(self.W2, h), self.b2)
        out = softmax(o_in)
        return h, out

    def predict(self, x: List[float]) -> int:
        _, out = self.forward(x)
        return max(range(len(out)), key=lambda i: out[i])

    def train(self, examples: List[Tuple[List[float], int]], epochs:int=30, lr:float=0.1):
        for epoch in range(epochs):
            random.shuffle(examples)
            for x_vec, label in examples:
                # forward
                h_in = vec_add(matvec_mul(self.W1, x_vec), self.b1)
                h = tanh_vec(h_in)
                o_in = vec_add(matvec_mul(self.W2, h), self.b2)
                out = softmax(o_in)
                # one-hot label
                y = [0.0]*self.output_dim
                y[label] = 1.0
                # compute output error (cross-entropy derivative)
                err_out = [out[i] - y[i] for i in range(self.output_dim)]
                # grad W2 = err_out * h^T
                for i in range(self.output_dim):
                    for j in range(self.hidden_dim):
                        self.W2[i][j] -= lr * err_out[i] * h[j]
                    self.b2[i] -= lr * err_out[i]
                # backprop to hidden
                # err_hidden = (W2^T * err_out) * tanh'(h_in)
                err_hidden = [0.0]*self.hidden_dim
                for j in range(self.hidden_dim):
                    s = 0.0
                    for i in range(self.output_dim):
                        s += self.W2[i][j]*err_out[i]
                    err_hidden[j] = s * (1.0 - h[j]*h[j])  # derivative using tanh'(x)=1-tanh^2(x)
                # grad W1 = err_hidden * x^T
                for j in range(self.hidden_dim):
                    for k in range(self.input_dim):
                        self.W1[j][k] -= lr * err_hidden[j] * x_vec[k]
                    self.b1[j] -= lr * err_hidden[j]

# -------------------------
# Intents & training data (seed)
# -------------------------
INTENTS = ["define","fact","math","time","date","teach","chat"]
# seed example phrases mapping to intents
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

# Build vocab and training examples (vectorized)
def build_training(vocab) -> List[Tuple[List[float], int]]:
    examples = []
    for text, intent in SEED_EXAMPLES:
        vec = text_to_vector(text, vocab)
        label = INTENTS.index(intent)
        examples.append((vec, label))
    # Also include some auto-generated negative examples (random)
    return examples

# We'll create the NN later after building vocab
NN_MODEL = None
VOCAB = build_vocab()
NN_MODEL = SimpleNN(len(VOCAB), max(16, len(VOCAB)//10), len(INTENTS))
TRAIN_EXAMPLES = build_training(VOCAB)
NN_MODEL.train(TRAIN_EXAMPLES, epochs=80, lr=0.08)

# Allow retraining with more data (e.g., learned patterns)
def retrain_model():
    global VOCAB, NN_MODEL, TRAIN_EXAMPLES
    VOCAB = build_vocab()
    TRAIN_EXAMPLES = build_training(VOCAB)
    # incorporate learned definitions as teach examples
    for k,v in ai_state.get("learned", {}).items():
        phrase = f"{k} means {v.get('definition','')}"
        TRAIN_EXAMPLES.append((text_to_vector(phrase, VOCAB), INTENTS.index("teach")))
    NN_MODEL = SimpleNN(len(VOCAB), max(16, len(VOCAB)//10), len(INTENTS))
    NN_MODEL.train(TRAIN_EXAMPLES, epochs=80, lr=0.08)

# -------------------------
# Simple generative model (Markov) and helpers
# -------------------------
class Markov:
    def __init__(self):
        self.map = {}  # (a,b) -> {c:count}
        self.starts = []

    def train(self, text):
        toks = tokenize(text)
        if len(toks) < 3:
            return
        self.starts.append((toks[0].lower(), toks[1].lower()))
        for i in range(len(toks)-2):
            key = (toks[i].lower(), toks[i+1].lower())
            nxt = toks[i+2].lower()
            self.map.setdefault(key, {})
            self.map[key][nxt] = self.map[key].get(nxt, 0) + 1

    def generate(self, seed=None, max_words=50):
        if seed:
            toks = tokenize(seed)
            if len(toks) >= 2:
                key = (toks[-2].lower(), toks[-1].lower())
            elif self.starts:
                key = random.choice(self.starts)
            else:
                return ""
        else:
            key = random.choice(self.starts) if self.starts else None
        if not key:
            return ""
        out = [key[0], key[1]]
        for _ in range(max_words-2):
            choices = self.map.get((out[-2], out[-1]))
            if not choices:
                break
            total = sum(choices.values())
            r = random.randint(1, total)
            acc = 0
            for w,cnt in choices.items():
                acc += cnt
                if r <= acc:
                    out.append(w)
                    break
        return " ".join(out)

MARKOV = Markov()
# initial train from dictionary examples
for k,v in merged_dictionary().items():
    for ex in v.get("examples", []):
        MARKOV.train(ex)
# also from seeded facts
for k,v in merged_dictionary().items():
    MARKOV.train(k + " " + v.get("definition",""))

def train_markov_from_history():
    MARKOV.map.clear()
    MARKOV.starts.clear()
    for k,v in merged_dictionary().items():
        for ex in v.get("examples", []):
            MARKOV.train(ex)
    for c in ai_state.get("conversations",[]):
        MARKOV.train(c.get("text",""))

# -------------------------
# Knowledge base (facts)
# -------------------------
KB = {
    "first president of the united states": "George Washington",
    "who was the first president of the united states": "George Washington",
    "who was the first president": "George Washington",
    "capital of france": "Paris",
    "who discovered penicillin": "Alexander Fleming",
    "largest planet": "Jupiter",
    "what is pi": "Pi is approximately 3.14159",
    "who wrote hamlet": "William Shakespeare",
    # add more facts here; user can teach new facts via "X means Y" (we store in learned)
}

# incorporate learned facts into KB retrieval via merged_dictionary/learned storage

# -------------------------
# Retrieval helpers
# -------------------------
def lookup_fact(query: str) -> Tuple[str,float]:
    """
    Try exact/partial matches in KB and learned definitions.
    Returns (answer, confidence)
    """
    q = query.lower().strip("? ")
    # direct KB lookup
    if q in KB:
        return KB[q], 0.95
    # try token overlap with KB keys
    qtokens = set(tokenize(q))
    best = None
    best_score = 0
    for k,v in KB.items():
        ktoks = set(tokenize(k))
        score = len(qtokens & ktoks)
        if score > best_score:
            best_score = score
            best = v
    if best_score >= 1:
        return best, 0.7
    # check learned definitions
    for k,v in ai_state.get("learned", {}).items():
        if normalize_key(k) in q or normalize_key(q) in k:
            return v.get("definition",""), 0.9
        ktoks = set(tokenize(k))
        sc = len(qtokens & ktoks)
        if sc > best_score:
            best_score = sc
            best = v.get("definition","")
    if best_score >= 1:
        return best, 0.6
    return None, 0.0

def normalize_key(s: str) -> str:
    return re.sub(r"[^a-z0-9\s]","", s.lower())

# -------------------------
# Compose assistant reply
# -------------------------
def format_definition(word:str, entry:Dict[str,Any]) -> str:
    ex = entry.get("examples",[])
    extext = ("\nExamples:\n - " + "\n - ".join(ex)) if ex else ""
    return f"**{word}** ({entry.get('type','')}): {entry.get('definition','')}{extext}"

def compose_reply(user_text: str) -> Dict[str,Any]:
    # vectorize and get intent via NN
    vocab = VOCAB  # defined earlier
    x = text_to_vector(user_text, vocab)
    intent_idx = NN_MODEL.predict(x)
    intent = INTENTS[intent_idx]

    # override checks: math, time/date, define commands, teach patterns
    lower = user_text.lower().strip()

    # direct math detection
    math_expr = re.sub(r"[^0-9\.\+\-\*\/\%\(\)\s\^]", "", user_text)
    if any(op in math_expr for op in "+-*/%") and re.search(r"\d", math_expr):
        try:
            expr = math_expr.replace("^", "**")
            res = eval(expr, {"__builtins__": None}, {"math": math, **{k: getattr(math,k) for k in dir(math) if not k.startswith("_")}})
            return {"reply": f"Math result: {res}", "meta": {"intent":"math"}}
        except Exception:
            pass

    # check time/date phrases
    if re.search(r"\bwhat(?:'s| is)? the time\b|\btime now\b|\bcurrent time\b", lower):
        return {"reply": f"The current time is {datetime.now().strftime('%H:%M:%S')}", "meta": {"intent":"time"}}
    if re.search(r"\bwhat(?:'s| is)? the date\b|\bcurrent date\b|\bdate today\b", lower):
        return {"reply": f"Today's date is {datetime.now().strftime('%Y-%m-%d')}", "meta": {"intent":"date"}}

    # explicit define or teach commands
    if lower.startswith("/define ") or lower.startswith("define "):
        # try parse
        rest = user_text.split(None,1)[1] if len(user_text.split(None,1))>1 else ""
        m = re.match(r'\s*([^\:]+)\s*[:\-]\s*(.+)', rest)
        if m:
            w = normalize_key(m.group(1))
            d = m.group(2).strip()
            ai_state.setdefault("learned", {})[w] = {"definition": d, "type":"learned", "examples": []}
            save_json(STATE_FILE, ai_state)
            retrain_model(); train_markov_from_history()
            return {"reply": f"Learned definition for '{w}'.", "meta":{"intent":"learning"}}
        else:
            # check if single-word define
            m2 = re.match(r'\s*([A-Za-z\'\-]+)\s*$', rest)
            if m2:
                key = normalize_key(m2.group(1))
                defs = merged_dictionary()
                if key in defs:
                    return {"reply": format_definition(key, defs[key]), "meta":{"intent":"define"}}
                else:
                    return {"reply": f"No definition found for '{key}'. To teach, type: /define {key}: your definition", "meta": {"intent":"define"}}
            return {"reply":"Usage: /define word: definition", "meta":{"intent":"define"}}

    # natural teaching patterns
    w,d = try_extract_definition(user_text)
    if w and d:
        ai_state.setdefault("learned", {})[w] = {"definition": d, "type":"learned", "examples": []}
        save_json(STATE_FILE, ai_state)
        retrain_model(); train_markov_from_history()
        return {"reply": f"Understood — saved '{w}' = {d}", "meta":{"intent":"learning"}}

    # If NN predicted 'define' or single-word user, try dictionary lookup
    if intent == "define" or re.fullmatch(r"[A-Za-z'\- ]+", user_text.strip()):
        key = normalize_key(user_text.strip())
        defs = merged_dictionary()
        if key in defs:
            return {"reply": format_definition(key, defs[key]), "meta":{"intent":"definition"}}
    # If NN predicted fact, try knowledge base lookup
    if intent == "fact":
        ans, conf = lookup_fact(user_text)
        if ans:
            return {"reply": str(ans), "meta":{"intent":"fact", "confidence":conf}}
    # fallback retrieval
    mem = retrieve_from_memory_or_learned(user_text)
    if mem:
        return {"reply": mem, "meta":{"intent":"memory"}}
    # else generate using markov
    gen = MARKOV.generate(seed=user_text, max_words=40)
    if gen:
        return {"reply": gen.capitalize() + ".", "meta":{"intent":"gen"}}
    # final fallback
    return {"reply": "I don't know that yet. You can teach me with 'X means Y' or '/define X: Y'.", "meta":{"intent":"unknown"}}

# -------------------------
# UI
# -------------------------
st.set_page_config(page_title="Jack — Offline ML Generative AI", layout="wide")
st.title("Jack — Offline ML Generative AI (from-scratch classifier)")

left, right = st.columns([3,1])

with right:
    st.header("Controls")
    if st.button("Clear memory & learned"):
        ai_state["conversations"].clear(); ai_state["learned"].clear()
        save_json(STATE_FILE, ai_state)
        retrain_model(); train_markov_from_history()
        st.success("Cleared.")
    if st.button("Export state JSON"):
        st.download_button("Download ai_state.json", data=json.dumps(ai_state, ensure_ascii=False, indent=2), file_name="ai_state.json")
    uploaded = st.file_uploader("Upload dictionary.json (merge)", type=["json"])
    if uploaded:
        try:
            ext = json.load(uploaded)
            if isinstance(ext, dict):
                for k,v in ext.items():
                    DICTIONARY[k.lower()] = v
                st.success("Merged dictionary.")
                retrain_model(); train_markov_from_history()
            else:
                st.error("dictionary.json must be an object")
        except Exception as e:
            st.error(f"Load failed: {e}")

with left:
    st.subheader("Conversation")
    # show last messages
    for msg in ai_state.get("conversations", [])[-300:]:
        who = "You" if msg.get("role","user")=="user" else "Jack"
        t = msg.get("time","")
        st.markdown(f"**{who}** <span style='color:gray;font-size:12px'>{t}</span>", unsafe_allow_html=True)
        st.write(msg.get("text",""))

    message = st.text_area("Message", height=120)
    c1, c2, c3 = st.columns([1,1,1])
    if c1.button("Send"):
        ui = message.strip()
        if ui:
            out = compose_reply(ui)
            reply = out["reply"]
            # save conversation
            ai_state.setdefault("conversations", []).append({"role":"user","text":ui,"time":datetime.now().isoformat()})
            ai_state.setdefault("conversations", []).append({"role":"assistant","text":reply,"time":datetime.now().isoformat()})
            save_json(STATE_FILE, ai_state)
            retrain_model(); train_markov_from_history()
            st.experimental_rerun()
    if c2.button("Complete"):
        ui = message.strip()
        if ui:
            comp = MARKOV.generate(seed=ui, max_words=40) or (ui + " ...")
            ai_state.setdefault("conversations", []).append({"role":"user","text":ui,"time":datetime.now().isoformat()})
            ai_state.setdefault("conversations", []).append({"role":"assistant","text":comp,"time":datetime.now().isoformat()})
            save_json(STATE_FILE, ai_state)
            st.experimental_rerun()
    if c3.button("Teach (define)"):
        ui = message.strip()
        m = re.match(r'\s*([^\:]+)\s*[:\-]\s*(.+)', ui)
        if m:
            w = normalize_key(m.group(1))
            d = m.group(2).strip()
            ai_state.setdefault("learned", {})[w] = {"definition": d, "type": "learned", "examples": []}
            save_json(STATE_FILE, ai_state)
            retrain_model(); train_markov_from_history()
            st.success(f"Learned {w}")
            st.experimental_rerun()
        else:
            st.warning("To teach: word: definition")

st.markdown("---")
st.markdown("**Usage examples:**")
st.markdown(
"""
- Ask facts: `Who was the first president of the U.S.?`  
- Define or teach: `gravity means the force that attracts` or `/define gravity: a force that attracts`  
- Math: `12 * (3 + 4)`  
- Time / date: `what is the time?` or `what is the date?`  
- Commands: `/clear`, `/delete 3` (delete conv #3), `/delete word` (remove learned def)
"""
)

