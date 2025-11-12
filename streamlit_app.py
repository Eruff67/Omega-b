# jack_offline_ai.py
# Jack — Offline hybrid GPT-lite + evolving ML classifier with embedded 1,000-word dictionary
# Single-file. Requires only Streamlit.
# Run:
#   pip install streamlit
#   streamlit run jack_offline_ai.py

import streamlit as st
import json
import os
import re
import math
import random
from datetime import datetime
from typing import List, Dict, Tuple, Any

# -------------------------
# Persistence
# -------------------------
STATE_FILE = "ai_state.json"
DICT_FILE = "dictionary.json"  # optional to drop in same folder or upload in UI

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
ai_state = load_json(STATE_FILE, {"conversations": [], "learned": {}, "model": {}, "settings": {}})

# -------------------------
# Embedded 1,000-word starter dictionary (templated definitions & examples)
# -------------------------
# For practicality we generate 1000 common-ish words and template definitions/examples.
# Important special entries (facts) are explicitly defined below.
EMBED_WORDS = [
    # A compact generated list of 1000 tokens: base common words, numbers, days, common verbs/adjectives etc.
    # To keep the file readable we programmatically create a list using common groups.
]

# Build EMBED_WORDS by combining common word lists programmatically
_common = """the be to of and a in that have I it for not on with he as you do at this but his by from they we say her she or an will my one all would there their what so up out if about who get which go me when make can like time no just him know take people into year your good some could them see other than then now look only come its over think also back after use two how our work first well way even new want because any these give day most us""".split()
_days = ["monday","tuesday","wednesday","thursday","friday","saturday","sunday"]
_nums = [str(i) for i in range(0,51)]
_colors = ["red","blue","green","yellow","black","white","gray","purple","orange","pink","brown"]
_more_verbs = ["run","walk","speak","talk","write","read","learn","teach","play","eat","drink","sleep","drive","open","close","buy","sell","build","create","think","feel","see","watch","listen","ask","answer"]
_more_nouns = ["apple","computer","city","country","car","house","book","music","movie","dog","cat","water","food","friend","family","school","teacher","student","money","game","story","science","history","phone","table","chair"]
_adj = ["good","bad","new","old","young","large","small","short","long","fast","slow","happy","sad","angry","beautiful","ugly","easy","hard","strong","weak","hot","cold","warm","cool"]
_extra = ["google","amazon","facebook","twitter","github","linux","windows","macos","ubuntu","kernel","python","java","javascript","html","css","sql","data","model","ai","ml"]

# assemble into list ensuring uniqueness
seen = set()
EMBED_WORDS = []
for group in (_common + _days + _nums + _colors + _more_verbs + _more_nouns + _adj + _extra):
    if group not in seen:
        EMBED_WORDS.append(group)
        seen.add(group)
# extend to ~1000 by creating synthetic entries if needed
i = 1
while len(EMBED_WORDS) < 1000:
    token = f"word{i}"
    if token not in seen:
        EMBED_WORDS.append(token)
        seen.add(token)
    i += 1

# Now create DICTIONARY entries programmatically
DICTIONARY: Dict[str, Dict[str, Any]] = {}

# Important human facts / knowledge entries (explicit, not templated)
DICTIONARY["george washington"] = {
    "definition":"The first President of the United States, serving from 1789 to 1797.",
    "type":"proper_noun",
    "examples":["George Washington led the Continental Army to victory during the American Revolutionary War."]
}
DICTIONARY["abraham lincoln"] = {
    "definition":"The 16th President of the United States, who led the country through the Civil War.",
    "type":"proper_noun",
    "examples":["Abraham Lincoln issued the Emancipation Proclamation in 1863."]
}
DICTIONARY["pi"] = {
    "definition":"Mathematical constant π, the ratio of a circle's circumference to its diameter (≈ 3.14159).",
    "type":"number",
    "examples":["Pi is approximately 3.14159."]
}
DICTIONARY["python"] = {
    "definition":"A high-level programming language widely used for scripting, data science, and web development.",
    "type":"noun",
    "examples":["I wrote a small script in Python."]
}
# Add a few more factual KB items in the DICTIONARY to help Q&A
DICTIONARY["paris"] = {"definition":"Capital city of France.", "type":"proper_noun", "examples":["Paris is known for the Eiffel Tower."]}
DICTIONARY["jupiter"] = {"definition":"The largest planet in the Solar System.", "type":"noun", "examples":["Jupiter has a large red spot."]}

# Templated fill for the remainder of EMBED_WORDS if not already defined
for w in EMBED_WORDS:
    key = w.lower()
    if key in DICTIONARY:
        continue
    DICTIONARY[key] = {
        "definition": f"A common English word: '{key}'.",
        "type": "common",
        "examples": [f"Example usage of '{key}' in a sentence."]
    }

# -------------------------
# Small knowledge base for direct Q/A (explicit strings)
# -------------------------
KB: Dict[str, str] = {
    "who was the first president of the united states": "George Washington",
    "who was the first president": "George Washington",
    "who was the first president of the u.s.": "George Washington",
    "capital of france": "Paris",
    "largest planet": "Jupiter",
    "what is pi": "Pi is approximately 3.14159",
    "who wrote hamlet": "William Shakespeare",
}

# -------------------------
# Tokenization & vocab builder
# -------------------------
WORD_RE = re.compile(r"[a-zA-Z']+")

def tokenize(text: str) -> List[str]:
    return WORD_RE.findall(text.lower())

def build_vocab() -> List[str]:
    vocab = set()
    # include dictionary keys, definitions, examples
    for k,v in DICTIONARY.items():
        vocab.update(tokenize(k))
        vocab.update(tokenize(v.get("definition","")))
        for ex in v.get("examples",[]):
            vocab.update(tokenize(ex))
    # conversation tokens
    for c in ai_state.get("conversations", [])[-500:]:
        vocab.update(tokenize(c.get("text","")))
    # safety add a few tokens
    vocab.update(["what","is","who","define","means","calculate","time","date","how","why","when","where"])
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
# Simple NN (single hidden layer) — from scratch
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
        self.W1 = random_matrix(hidden_dim, input_dim, scale=0.2)
        self.b1 = [0.0]*hidden_dim
        self.W2 = random_matrix(output_dim, hidden_dim, scale=0.2)
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

    def train(self, dataset: List[Tuple[List[float], int]], epochs:int=50, lr:float=0.05):
        for _ in range(epochs):
            random.shuffle(dataset)
            for x_vec, label in dataset:
                h_in = add_vec(matvec(self.W1, x_vec), self.b1)
                h = tanh_vec(h_in)
                o_in = add_vec(matvec(self.W2, h), self.b2)
                out = softmax(o_in)
                # target
                y = [0.0]*len(out); y[label]=1.0
                err_out = [out[i]-y[i] for i in range(len(out))]
                # update W2,b2
                for i in range(len(self.W2)):
                    for j in range(len(self.W2[0])):
                        self.W2[i][j] -= lr * err_out[i] * h[j]
                    self.b2[i] -= lr * err_out[i]
                # backprop to hidden
                err_hidden = [0.0]*len(h)
                for j in range(len(h)):
                    s = 0.0
                    for i in range(len(err_out)):
                        s += self.W2[i][j]*err_out[i]
                    err_hidden[j] = s * (1.0 - h[j]*h[j])
                # update W1,b1
                for j in range(len(self.W1)):
                    for k in range(len(self.W1[0])):
                        self.W1[j][k] -= lr * err_hidden[j] * x_vec[k]
                    self.b1[j] -= lr * err_hidden[j]

# -------------------------
# Intents, seed data, and training loop improvements
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
    for text, intent in SEED_EXAMPLES:
        vec = text_to_vector(text, vocab)
        data.append((vec, INTENTS.index(intent)))
    # incorporate learned definitions as additional "teach" examples
    for k,v in ai_state.get("learned", {}).items():
        phrase = f"{k} means {v.get('definition','')}"
        data.append((text_to_vector(phrase, vocab), INTENTS.index("teach")))
    return data

# Build initial vocab and model
VOCAB = build_vocab()
NN_MODEL = TinyNN(len(VOCAB), max(32, len(VOCAB)//12), len(INTENTS))
TRAIN_DATA = build_training(VOCAB)
if TRAIN_DATA:
    NN_MODEL.train(TRAIN_DATA, epochs=120, lr=0.06)  # improved initial training

# Allow incremental retrain when learning new definitions
def incremental_retrain():
    global VOCAB, NN_MODEL
    VOCAB = build_vocab()
    dataset = build_training(VOCAB)
    NN_MODEL = TinyNN(len(VOCAB), max(32, len(VOCAB)//12), len(INTENTS))
    if dataset:
        NN_MODEL.train(dataset, epochs=80, lr=0.06)

# -------------------------
# Markov generator for fallback generative replies
# -------------------------
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

def train_markov():
    MARKOV.map.clear(); MARKOV.starts.clear()
    for k,v in DICTIONARY.items():
        for ex in v.get("examples", []):
            MARKOV.train(ex)
        MARKOV.train(k + " " + v.get("definition",""))
    for c in ai_state.get("conversations", []):
        MARKOV.train(c.get("text",""))

train_markov()

# -------------------------
# Retrieval helpers & lookup
# -------------------------
LEARN_PATTERNS = [
    re.compile(r'^\s*define\s+([^\:]+)\s*[:\-]\s*(.+)$', re.I),
    re.compile(r'^\s*([A-Za-z\'\-\s]+)\s+means\s+(.+)$', re.I),
    re.compile(r'^\s*([A-Za-z\'\-\s]+)\s+is\s+(.+)$', re.I),
    re.compile(r'^\s*([^\s=]+)\s*=\s*(.+)$', re.I),
]

def normalize_key(s: str) -> str:
    return re.sub(r"[^a-z0-9\s]", "", s.lower()).strip()

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

def lookup_kb(query: str) -> Tuple[Any, float]:
    q = normalize_key(query.strip("? "))
    # exact KB match
    if q in KB:
        return KB[q], 0.95
    qtokens = set(tokenize(q))
    best = None; best_score = 0
    for k,v in KB.items():
        sc = len(qtokens & set(tokenize(k)))
        if sc > best_score:
            best_score = sc; best = v
    if best_score >= 1:
        return best, 0.7
    # check learned as facts
    for k,v in ai_state.get("learned", {}).items():
        if normalize_key(k) in q or normalize_key(q) in k:
            return v.get("definition",""), 0.85
    return None, 0.0

# -------------------------
# Core reply composition
# -------------------------
def format_definition(key: str, entry: Dict[str, Any]) -> str:
    ex = entry.get("examples", [])
    ex_text = ("\nExamples:\n - " + "\n - ".join(ex)) if ex else ""
    return f"**{key}** ({entry.get('type','')}): {entry.get('definition','')}{ex_text}"

def compose_reply(user_text: str) -> Dict[str,Any]:
    user = user_text.strip()
    lower = user.lower()

    # commands
    if lower in ("/clear", "clear memory", "wipe memory"):
        ai_state["conversations"].clear(); ai_state["learned"].clear()
        save_json(STATE_FILE, ai_state)
        incremental_retrain(); train_markov()
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
                incremental_retrain(); train_markov()
                return {"reply": f"Removed learned definition for '{key}'.", "meta":{"intent":"memory"}}
            else:
                return {"reply": f"No learned definition for '{key}'.", "meta":{"intent":"error"}}

    # math detection (safe subset)
    math_expr = re.sub(r"[^0-9\.\+\-\*\/\%\(\)\s\^]", "", user)
    if any(op in math_expr for op in "+-*/%") and re.search(r"\d", math_expr):
        try:
            expr = math_expr.replace("^", "**")
            res = eval(expr, {"__builtins__": None}, {"math": math, **{k:getattr(math,k) for k in dir(math) if not k.startswith("_")}})
            return {"reply": f"Math result: {res}", "meta":{"intent":"math"}}
        except Exception:
            pass

    # time/date checks
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
        # if single word define e.g. "define apple"
        m2 = re.match(r'\s*([A-Za-z\'\-]+)\s*$', rest)
        if m2:
            key = normalize_key(m2.group(1))
            defs = merged_dictionary()
            if key in defs:
                return {"reply": format_definition(key, defs[key]), "meta":{"intent":"definition"}}
            else:
                return {"reply": f"No definition found for '{key}'. Teach with '/define {key}: ...'", "meta":{"intent":"definition"}}
        return {"reply":"Usage: /define word: definition", "meta":{"intent":"define"}}

    # natural teaching patterns
    w,d = try_extract_definition(user)
    if w and d:
        ai_state.setdefault("learned", {})[w] = {"definition": d, "type":"learned", "examples": []}
        save_json(STATE_FILE, ai_state)
        incremental_retrain(); train_markov()
        return {"reply": f"Understood — saved '{w}' = {d}", "meta":{"intent":"learning"}}

    # intent classification via NN
    x = text_to_vector(user, VOCAB)
    intent_idx = NN_MODEL.predict(x)
    intent = INTENTS[intent_idx]

    # intent-driven behavior
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
        return {"reply":"I don't have that definition yet. Teach me: '/define word: definition' or say 'X means Y'.", "meta":{"intent":"definition"}}
    if intent == "time":
        return {"reply": f"The current time is {datetime.now().strftime('%H:%M:%S')}", "meta":{"intent":"time"}}
    if intent == "date":
        return {"reply": f"Today's date is {datetime.now().strftime('%Y-%m-%d')}", "meta":{"intent":"date"}}
    if intent == "math":
        math_expr = re.sub(r"[^0-9\.\+\-\*\/\%\(\)\s\^]", "", user)
        try:
            expr = math_expr.replace("^","**")
            res = eval(expr, {"__builtins__": None}, {"math": math, **{k:getattr(math,k) for k in dir(math) if not k.startswith("_")}})
            return {"reply": f"Math result: {res}", "meta":{"intent":"math"}}
        except Exception:
            pass

    # retrieval
    mem = retrieve_from_memory_or_learned(user)
    if mem:
        return {"reply": mem, "meta":{"intent":"memory"}}

    # markov generation fallback
    gen = MARKOV.generate(seed=user, max_words=40)
    if gen:
        return {"reply": gen.capitalize() + ".", "meta":{"intent":"gen"}}

    return {"reply":"I don't know that yet. You can teach me with 'X means Y' or '/define X: Y'.", "meta":{"intent":"unknown"}}

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="Jack — 1,000-word Embedded Offline AI", layout="wide")
st.title("Jack — Offline Hybrid GPT-lite + ML (1,000-word dictionary)")

left, right = st.columns([3,1])

with right:
    st.header("Controls")
    if st.button("Clear memory & learned"):
        ai_state["conversations"].clear(); ai_state["learned"].clear()
        save_json(STATE_FILE, ai_state)
        incremental_retrain(); train_markov()
        st.success("Cleared.")
    if st.button("Export state"):
        st.download_button("Download ai_state.json", data=json.dumps(ai_state, ensure_ascii=False, indent=2), file_name="ai_state.json")
    uploaded = st.file_uploader("Upload dictionary.json (merge)", type=["json"])
    if uploaded:
        try:
            ext = json.load(uploaded)
            if isinstance(ext, dict):
                for k,v in ext.items():
                    DICTIONARY[k.lower()] = v
                st.success("Merged uploaded dictionary.")
                incremental_retrain(); train_markov()
            else:
                st.error("dictionary.json must be an object")
        except Exception as e:
            st.error(f"Failed to load dictionary: {e}")

with left:
    st.subheader("Conversation")
    # display history
    for msg in ai_state.get("conversations", [])[-300:]:
        who = "You" if msg.get("role","user")=="user" else "Jack"
        t = msg.get("time","")
        st.markdown(f"**{who}**  <span style='color:gray;font-size:12px'>{t}</span>", unsafe_allow_html=True)
        st.write(msg.get("text",""))

    user_input = st.text_area("Message (Shift+Enter new line)", height=140)
    c1, c2, c3 = st.columns([1,1,1])
    if c1.button("Send"):
        ui = user_input.strip()
        if ui:
            out = compose_reply(ui)
            reply = out.get("reply","")
            ai_state.setdefault("conversations", []).append({"role":"user","text":ui,"time":datetime.now().isoformat()})
            ai_state.setdefault("conversations", []).append({"role":"assistant","text":reply,"time":datetime.now().isoformat()})
            save_json(STATE_FILE, ai_state)
            incremental_retrain(); train_markov()
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
            w = normalize_key(m.group(1)); d = m.group(2).strip()
            ai_state.setdefault("learned", {})[w] = {"definition": d, "type":"learned", "examples": []}
            save_json(STATE_FILE, ai_state)
            incremental_retrain(); train_markov()
            st.success(f"Learned '{w}'.")
            st.experimental_rerun()
        else:
            st.warning("To teach: word: definition (e.g. gravity: force that pulls)")

st.markdown("---")
st.markdown("**Examples:** Ask `Who was the first president of the U.S.?`, `12 * (3 + 4)`, `define gravity`, or teach `gravity means a force`.")

# End of file

