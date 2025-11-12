# jack_offline_ai.py
# Jack — Offline hybrid GPT-lite + evolving ML intent classifier
# Single-file Streamlit app. Requires only Streamlit.
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
global VOCAB, NN_MODEL
# ----------------- Files & persistence -----------------
STATE_FILE = "ai_state.json"
DICT_FILE = "dictionary.json"  # optional external dictionary to drop in folder

def load_json(path: str, default):
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return default

def save_json(path: str, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

# initialize state
ai_state = load_json(STATE_FILE, {"conversations": [], "learned": {}, "settings": {}})

# ----------------- Seed dictionary & knowledge -----------------
# Small seed dictionary; you can upload/merge a larger dictionary.json later
SEED_DICT: Dict[str, Dict[str, Any]] = {
    "george washington": {"definition":"The first President of the United States (1789–1797).", "type":"proper_noun", "examples":["George Washington was commander of the Continental Army."]},
    "abraham lincoln": {"definition":"16th President of the United States who led the country during the Civil War.", "type":"proper_noun", "examples":["Abraham Lincoln delivered the Gettysburg Address."]},
    "apple": {"definition":"A round fruit with red or green skin.", "type":"noun", "examples":["I ate an apple."]},
    "python": {"definition":"A high-level programming language.", "type":"noun", "examples":["I wrote the script in Python."]},
    "time": {"definition":"The ongoing sequence of events; measured in seconds, minutes, and hours.", "type":"noun", "examples":["What time is it?"]},
    "date": {"definition":"A particular day of a month or year.", "type":"noun", "examples":["What is the date today?"]},
    "run": {"definition":"To move swiftly on foot.", "type":"verb", "examples":["I run every morning."]},
    "learn": {"definition":"To gain knowledge or skill.", "type":"verb", "examples":["She learned the poem."]},
}

# Knowledge base for FAQ-style facts
KB: Dict[str, str] = {
    "who was the first president of the united states": "George Washington",
    "who was the first president of the u.s.": "George Washington",
    "who was the first president": "George Washington",
    "capital of france": "Paris",
    "largest planet": "Jupiter",
    "who wrote hamlet": "William Shakespeare",
}

# Merge external dictionary (if present at startup)
external = load_json(DICT_FILE, None)
if external and isinstance(external, dict):
    DICTIONARY = {**{k.lower():v for k,v in SEED_DICT.items()}, **{k.lower():v for k,v in external.items()}}
else:
    DICTIONARY = {k.lower():v for k,v in SEED_DICT.items()}

def merged_dictionary() -> Dict[str, Dict[str, Any]]:
    """Return combined dictionary (seed/external + learned)."""
    d = {**DICTIONARY}
    for k,v in ai_state.get("learned", {}).items():
        d[k.lower()] = {"definition": v.get("definition",""), "type": v.get("type","learned"), "examples": v.get("examples", [])}
    return d

# ----------------- Tokenization / vocab -----------------
WORD_RE = re.compile(r"[a-zA-Z']+")

def tokenize(text: str) -> List[str]:
    return WORD_RE.findall(text.lower())

def build_vocab() -> List[str]:
    vocab = set()
    # dictionary keys, defs, examples
    for k,v in merged_dictionary().items():
        vocab.update(tokenize(k))
        vocab.update(tokenize(v.get("definition","")))
        for ex in v.get("examples", []):
            vocab.update(tokenize(ex))
    # recent conversation
    for conv in ai_state.get("conversations", [])[-500:]:
        vocab.update(tokenize(conv.get("text","")))
    # common tokens
    common = ["what","who","when","where","why","how","define","means","calculate","time","date"]
    vocab.update(common)
    return sorted(vocab)

def text_to_vector(text: str, vocab_list: List[str]) -> List[float]:
    toks = tokenize(text)
    vec = [0.0]*len(vocab_list)
    idx_map = {w:i for i,w in enumerate(vocab_list)}
    for t in toks:
        if t in idx_map:
            vec[idx_map[t]] += 1.0
    # normalize (L2)
    norm = math.sqrt(sum(x*x for x in vec)) or 1.0
    return [x/norm for x in vec]

# ----------------- Small neural network (from-scratch) -----------------
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

def softmax(v):
    mx = max(v)
    exps = [math.exp(x-mx) for x in v]
    s = sum(exps) or 1.0
    return [e/s for e in exps]

class SimpleNN:
    def __init__(self, input_dim:int, hidden_dim:int, output_dim:int):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.W1 = random_matrix(hidden_dim, input_dim, scale=0.2)
        self.b1 = [0.0]*hidden_dim
        self.W2 = random_matrix(output_dim, hidden_dim, scale=0.2)
        self.b2 = [0.0]*output_dim

    def forward(self, x: List[float]) -> Tuple[List[float], List[float]]:
        h_in = vec_add(matvec_mul(self.W1, x), self.b1)
        h = tanh_vec(h_in)
        o_in = vec_add(matvec_mul(self.W2, h), self.b2)
        out = softmax(o_in)
        return h, out

    def predict(self, x: List[float]) -> int:
        _, out = self.forward(x)
        return max(range(len(out)), key=lambda i: out[i])

    def train(self, examples: List[Tuple[List[float], int]], epochs:int=30, lr:float=0.05):
        for epoch in range(epochs):
            random.shuffle(examples)
            for x_vec, label in examples:
                # forward
                h_in = vec_add(matvec_mul(self.W1, x_vec), self.b1)
                h = tanh_vec(h_in)
                o_in = vec_add(matvec_mul(self.W2, h), self.b2)
                out = softmax(o_in)
                # target
                y = [0.0]*self.output_dim
                y[label] = 1.0
                # output error
                err_out = [out[i] - y[i] for i in range(self.output_dim)]
                # update W2, b2
                for i in range(self.output_dim):
                    for j in range(self.hidden_dim):
                        self.W2[i][j] -= lr * err_out[i] * h[j]
                    self.b2[i] -= lr * err_out[i]
                # backprop to hidden
                err_hidden = [0.0]*self.hidden_dim
                for j in range(self.hidden_dim):
                    s = 0.0
                    for i in range(self.output_dim):
                        s += self.W2[i][j]*err_out[i]
                    err_hidden[j] = s * (1.0 - h[j]*h[j])
                # update W1, b1
                for j in range(self.hidden_dim):
                    for k in range(self.input_dim):
                        self.W1[j][k] -= lr * err_hidden[j] * x_vec[k]
                    self.b1[j] -= lr * err_hidden[j]

# ----------------- Intents + seed training -----------------
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

def build_training_examples(vocab: List[str]) -> List[Tuple[List[float], int]]:
    examples = []
    for text, intent in SEED_EXAMPLES:
        vec = text_to_vector(text, vocab)
        examples.append((vec, INTENTS.index(intent)))
    # include learned as "teach" examples
    for k,v in ai_state.get("learned", {}).items():
        phrase = f"{k} means {v.get('definition','')}"
        vec = text_to_vector(phrase, vocab)
        examples.append((vec, INTENTS.index("teach")))
    return examples

# ----------------- Markov generator -----------------
class Markov:
    def __init__(self):
        self.map = {}
        self.starts = []

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

    def generate(self, seed: str=None, max_words:int=40) -> str:
        if seed:
            toks = tokenize(seed)
            if len(toks) >= 2:
                key = (toks[-2].lower(), toks[-1].lower())
            else:
                key = random.choice(self.starts) if self.starts else None
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

def train_markov_from_data():
    MARKOV.map.clear(); MARKOV.starts.clear()
    for k,v in merged_dictionary().items():
        for ex in v.get("examples", []):
            MARKOV.train(ex)
        MARKOV.train(k + " " + v.get("definition",""))
    for c in ai_state.get("conversations", []):
        MARKOV.train(c.get("text",""))

# ----------------- Build initial vocab and model -----------------
VOCAB = build_vocab()
NN_MODEL = SimpleNN(len(VOCAB), max(16, len(VOCAB)//8), len(INTENTS))
TRAIN_EXAMPLES = build_training_examples(VOCAB)
if TRAIN_EXAMPLES:
    NN_MODEL.train(TRAIN_EXAMPLES, epochs=80, lr=0.06)
train_markov_from_data()

# ----------------- Helpers: normalize, extract definitions, retrieval -----------------
def normalize_key(s: str) -> str:
    return re.sub(r"[^a-z0-9\s]", "", s.lower())

LEARN_PATTERNS = [
    re.compile(r'^\s*define\s+([^\:]+)\s*[:\-]\s*(.+)$', re.I),
    re.compile(r'^\s*([A-Za-z\'\-\s]+)\s+means\s+(.+)$', re.I),
    re.compile(r'^\s*([A-Za-z\'\-\s]+)\s+is\s+(.+)$', re.I),
    re.compile(r'^\s*([^\s=]+)\s*=\s*(.+)$', re.I),
]

def try_extract_definition(text: str) -> Tuple[str,str]:
    s = text.strip()
    for pat in LEARN_PATTERNS:
        m = pat.match(s)
        if m:
            left = m.group(1).strip()
            right = m.group(2).strip().rstrip(".")
            left_token = left.split()[0]
            return normalize_key(left_token), right
    return None, None

def retrieve_from_memory_or_learned(query: str) -> str:
    qtokens = set(tokenize(query))
    best_score = 0
    best_text = None
    for conv in ai_state.get("conversations", []):
        t = conv.get("text","")
        score = len(qtokens & set(tokenize(t)))
        if score > best_score:
            best_score = score
            best_text = t
    # learned definitions
    for k,v in ai_state.get("learned", {}).items():
        tokens = set(tokenize(k + " " + v.get("definition","")))
        score = len(qtokens & tokens)
        if score > best_score:
            best_score = score
            best_text = f"{k}: {v.get('definition','')}"
    if best_score >= 1:
        return best_text
    return None

def lookup_fact(query: str) -> Tuple[Any, float]:
    q = normalize_key(query.strip("? "))
    if q in KB:
        return KB[q], 0.95
    # token-overlap heuristic
    qtokens = set(tokenize(q))
    best = None; best_score = 0
    for k,v in KB.items():
        score = len(qtokens & set(tokenize(k)))
        if score > best_score:
            best_score = score
            best = v
    if best_score >= 1:
        return best, 0.7
    # learned definitions as facts
    for k,v in ai_state.get("learned", {}).items():
        if normalize_key(k) in q or normalize_key(q) in k:
            return v.get("definition",""), 0.85
    return None, 0.0

# ----------------- Compose reply -----------------
def compose_reply(user_text: str) -> Dict[str, Any]:
    user = user_text.strip()
    lower = user.lower()

    # Commands
    if lower in ("/clear", "clear memory", "wipe memory"):
        ai_state["conversations"].clear(); ai_state["learned"].clear()
        save_json(STATE_FILE, ai_state)
        # rebuild models
        global VOCAB, NN_MODEL
        VOCAB = build_vocab()
        NN_MODEL = SimpleNN(len(VOCAB), max(16, len(VOCAB)//8), len(INTENTS))
        train_examples = build_training_examples(VOCAB)
        if train_examples:
            NN_MODEL.train(train_examples, epochs=50, lr=0.05)
        train_markov_from_data()
        return {"reply":"Memory and learned definitions cleared.", "meta":{"intent":"memory"}}

    if lower.startswith("/delete "):
        arg = lower[len("/delete "):].strip()
        if arg.isdigit():
            idx = int(arg)-1
            if 0 <= idx < len(ai_state.get("conversations", [])):
                removed = ai_state["conversations"].pop(idx)
                save_json(STATE_FILE, ai_state)
                return {"reply": f"Deleted conversation #{idx+1}: {removed.get('text')}", "meta":{"intent":"memory"}}
            else:
                return {"reply":"Invalid conversation index.", "meta":{"intent":"error"}}
        else:
            key = normalize_key(arg)
            if key in ai_state.get("learned", {}):
                ai_state["learned"].pop(key)
                save_json(STATE_FILE, ai_state)
                return {"reply": f"Removed learned definition for '{key}'.", "meta":{"intent":"memory"}}
            else:
                return {"reply": f"No learned definition found for '{key}'.", "meta":{"intent":"error"}}

    # math detection (safe subset)
    math_expr = re.sub(r"[^0-9\.\+\-\*\/\%\(\)\s\^]", "", user)
    if any(op in math_expr for op in "+-*/%") and re.search(r"\d", math_expr):
        try:
            expr = math_expr.replace("^", "**")
            res = eval(expr, {"__builtins__": None}, {"math": math, **{k:getattr(math,k) for k in dir(math) if not k.startswith("_")}})
            return {"reply": f"Math result: {res}", "meta":{"intent":"math"}}
        except Exception:
            pass

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
            w = normalize_key(m.group(1))
            d = m.group(2).strip()
            ai_state.setdefault("learned", {})[w] = {"definition": d, "type":"learned", "examples": []}
            save_json(STATE_FILE, ai_state)
            # retrain small model incrementally
            retrain_small_model()
            train_markov_from_data()
            return {"reply": f"Learned definition for '{w}'.", "meta":{"intent":"learning"}}
        m2 = re.match(r'\s*([A-Za-z\'\-]+)\s*$', rest)
        if m2:
            key = normalize_key(m2.group(1))
            defs = merged_dictionary()
            if key in defs:
                entry = defs[key]
                return {"reply": format_definition_reply(key, entry), "meta":{"intent":"definition"}}
            else:
                return {"reply": f"No definition found for '{key}'. Use '/define {key}: <definition>' to teach me.", "meta":{"intent":"definition"}}
        return {"reply":"Usage: /define word: definition", "meta":{"intent":"define"}}

    # natural "X means Y" teaching
    w,d = try_extract_definition(user)
    if w and d:
        ai_state.setdefault("learned", {})[w] = {"definition": d, "type":"learned", "examples": []}
        save_json(STATE_FILE, ai_state)
        retrain_small_model()
        train_markov_from_data()
        return {"reply": f"Understood — saved '{w}' = {d}", "meta":{"intent":"learning"}}

    # intent classification by NN
    # ensure VOCAB and NN_MODEL exist
    global VOCAB, NN_MODEL
    if not VOCAB:
        VOCAB = build_vocab()
    if NN_MODEL is None:
        NN_MODEL = SimpleNN(len(VOCAB), max(16, len(VOCAB)//8), len(INTENTS))
    xvec = text_to_vector(user, VOCAB)
    intent_idx = NN_MODEL.predict(xvec)
    intent = INTENTS[intent_idx]

    # intent-driven responses
    if intent == "fact":
        ans, conf = lookup_fact(user)
        if ans:
            return {"reply": str(ans), "meta":{"intent":"fact","confidence":conf}}

    if intent == "define":
        # try direct word lookup
        key = normalize_key(user)
        defs = merged_dictionary()
        if key in defs:
            return {"reply": format_definition_reply(key, defs[key]), "meta":{"intent":"definition"}}
        # fallback: look for pattern "meaning of X"
        m = re.search(r'\bmeaning of ([a-zA-Z\'\- ]+)\b', lower)
        if m:
            k = normalize_key(m.group(1))
            if k in defs:
                return {"reply": format_definition_reply(k, defs[k]), "meta":{"intent":"definition"}}
        # no definition
        return {"reply": "I don't have that definition yet. Teach me: `/define word: definition` or say 'X means Y'.", "meta":{"intent":"definition"}}

    if intent == "time":
        return {"reply": f"The current time is {datetime.now().strftime('%H:%M:%S')}", "meta":{"intent":"time"}}
    if intent == "date":
        return {"reply": f"Today's date is {datetime.now().strftime('%Y-%m-%d')}", "meta":{"intent":"date"}}
    if intent == "math":
        # try compute
        math_expr = re.sub(r"[^0-9\.\+\-\*\/\%\(\)\s\^]", "", user)
        try:
            expr = math_expr.replace("^","**")
            res = eval(expr, {"__builtins__": None}, {"math": math, **{k:getattr(math,k) for k in dir(math) if not k.startswith("_")}})
            return {"reply": f"Math result: {res}", "meta":{"intent":"math"}}
        except Exception:
            pass

    # retrieval from memory/learned
    mem = retrieve_from_memory_or_learned(user)
    if mem:
        return {"reply": mem, "meta":{"intent":"memory"}}

    # Markov generation fallback
    gen = MARKOV.generate(seed=user, max_words=40)
    if gen:
        return {"reply": gen.capitalize() + ".", "meta":{"intent":"gen"}}

    return {"reply": "I don't know that yet. You can teach me with 'X means Y' or '/define X: Y'.", "meta":{"intent":"unknown"}}

# small helper to retrain the NN incrementally (rebuild vocab and retrain)
def retrain_small_model():
    global VOCAB, NN_MODEL
    VOCAB = build_vocab()
    NN_MODEL = SimpleNN(len(VOCAB), max(16, len(VOCAB)//8), len(INTENTS))
    examples = build_training_examples(VOCAB)
    if examples:
        NN_MODEL.train(examples, epochs=60, lr=0.06)

# helper to format definitions
def format_definition_reply(key: str, entry: Dict[str, Any]) -> str:
    typ = entry.get("type","")
    definition = entry.get("definition","")
    examples = entry.get("examples",[])
    ex_text = ("\nExamples:\n - " + "\n - ".join(examples)) if examples else ""
    return f"**{key}** ({typ}): {definition}{ex_text}"

# ----------------- Streamlit UI -----------------
st.set_page_config(page_title="Jack — Offline ML + GPT-lite", layout="wide")
st.title("Jack — Offline Hybrid GPT-lite + ML (Evolving)")

left, right = st.columns([3,1])

with right:
    st.header("Controls")
    if st.button("Clear memory & learned"):
        ai_state["conversations"].clear(); ai_state["learned"].clear(); save_json(STATE_FILE, ai_state)
        retrain_small_model(); train_markov_from_data()
        st.success("Cleared memory & learned definitions.")
    if st.button("Export state (download)"):
        st.download_button("Download ai_state.json", data=json.dumps(ai_state, ensure_ascii=False, indent=2), file_name="ai_state.json")
    uploaded = st.file_uploader("Upload dictionary.json (merge)", type=["json"])
    if uploaded:
        try:
            ext = json.load(uploaded)
            if isinstance(ext, dict):
                for k,v in ext.items():
                    DICTIONARY[k.lower()] = v
                st.success("Merged uploaded dictionary into runtime dictionary.")
                retrain_small_model(); train_markov_from_data()
            else:
                st.error("dictionary.json must be an object mapping words to entries.")
        except Exception as e:
            st.error(f"Failed to load dictionary: {e}")

with left:
    st.subheader("Conversation")
    # show last messages
    hist = ai_state.get("conversations", [])
    for m in hist[-300:]:
        who = "You" if m.get("role","user")=="user" else "Jack"
        t = m.get("time","")
        st.markdown(f"**{who}**  <span style='color:gray;font-size:12px'>{t}</span>", unsafe_allow_html=True)
        st.write(m.get("text",""))

    user_input = st.text_area("Message (Shift+Enter for newline)", height=120)
    c1, c2, c3 = st.columns([1,1,1])
    if c1.button("Send"):
        ui = user_input.strip()
        if ui:
            out = compose_reply(ui)
            reply = out.get("reply","")
            # save conversation
            ai_state.setdefault("conversations", []).append({"role":"user","text":ui,"time":datetime.now().isoformat()})
            ai_state.setdefault("conversations", []).append({"role":"assistant","text":reply,"time":datetime.now().isoformat()})
            save_json(STATE_FILE, ai_state)
            retrain_small_model(); train_markov_from_data()
            st.experimental_rerun()
    if c2.button("Complete"):
        ui = user_input.strip()
        if ui:
            comp = MARKOV.generate(seed=ui, max_words=40) or (ui + " ...")
            ai_state.setdefault("conversations", []).append({"role":"user","text":ui,"time":datetime.now().isoformat()})
            ai_state.setdefault("conversations", []).append({"role":"assistant","text":comp,"time":datetime.now().isoformat()})
            save_json(STATE_FILE, ai_state)
            st.experimental_rerun()
    if c3.button("Teach"):
        ui = user_input.strip()
        m = re.match(r'\s*([^\:]+)\s*[:\-]\s*(.+)', ui)
        if m:
            w = normalize_key(m.group(1))
            d = m.group(2).strip()
            ai_state.setdefault("learned", {})[w] = {"definition": d, "type":"learned", "examples": []}
            save_json(STATE_FILE, ai_state)
            retrain_small_model(); train_markov_from_data()
            st.success(f"Learned '{w}'.")
            st.experimental_rerun()
        else:
            st.warning("To teach: enter `word: definition` (e.g. gravity: a force that pulls)")

st.markdown("---")
st.markdown("**Usage examples:**")
st.markdown(
"""
- Ask a fact: `Who was the first president of the U.S.?`  
- Teach a definition: `gravity means the force that attracts` or `/define gravity: a force that attracts`  
- Math: `12 * (3 + 4)`  
- Time/date: `what is the time?` or `what is the date?`  
- Commands: `/clear`, `/delete 3` (delete conversation #3), `/delete gravity` (remove learned def)
"""
)

# End of file
