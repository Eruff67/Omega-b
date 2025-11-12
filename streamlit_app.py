# jack_offline_core.py
# Jack — Offline AI core (no UI). Single-file.
# Features:
# - Embedded ~1000-word templated dictionary
# - From-scratch TinyNN intent classifier
# - Markov fallback generative model
# - Persistent learned definitions + conversation history (ai_state.json)
# - Teaching: "X means Y" or "/define X: Y"
# - Safe math evaluation, time/date, KB lookup
# - CLI REPL when run directly
#
# Run:
#   python jack_offline_core.py

import json
import os
import re
import math
import random
from datetime import datetime
from typing import List, Dict, Tuple, Any, Optional

# -------------------------
# File names
# -------------------------
STATE_FILE = "ai_state.json"
DICT_FILE = "dictionary.json"

# -------------------------
# Persistence helpers
# -------------------------
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

# -------------------------
# Global state (persistent)
# -------------------------
ai_state: Dict[str, Any] = load_json(STATE_FILE, {"conversations": [], "learned": {}, "settings": {}, "model_meta": {}})

# -------------------------
# Build embedded 1,000-word dictionary (templated)
# -------------------------
_common = """the be to of and a in that have i it for not on with he as you do at this but his by from they we say her she or will my one all would there their what so up out if about who get which go me when make can like time no just him know take people into year your good some could them see other than then now look only come its over think also back after use two how our work first well way even new want because any these give day most us""".split()
_days = ["monday","tuesday","wednesday","thursday","friday","saturday","sunday"]
_nums = [str(i) for i in range(0,101)]
_colors = ["red","blue","green","yellow","black","white","gray","purple","orange","pink","brown"]
_more_verbs = ["run","walk","speak","talk","write","read","learn","teach","play","eat","drink","sleep","drive","open","close","buy","sell","build","create","think","feel","see","watch","listen","ask","answer","help","love","hate","start","stop"]
_more_nouns = ["apple","computer","city","country","car","house","book","music","movie","dog","cat","water","food","friend","family","school","teacher","student","money","game","story","science","history","phone","table","chair","garden","ocean","river"]
_adj = ["good","bad","new","old","young","large","small","short","long","fast","slow","happy","sad","angry","beautiful","ugly","easy","hard","strong","weak","hot","cold","warm","cool"]
_extra = ["google","amazon","facebook","twitter","github","linux","windows","macos","ubuntu","kernel","python","java","javascript","html","css","sql","data","model","ai","ml","neural","network"]

seen = set()
EMBED_WORDS: List[str] = []
for token in (_common + _days + _nums + _colors + _more_verbs + _more_nouns + _adj + _extra):
    t = token.lower()
    if t not in seen:
        EMBED_WORDS.append(t)
        seen.add(t)

i = 1
while len(EMBED_WORDS) < 1000:
    token = f"word{i}"
    if token not in seen:
        EMBED_WORDS.append(token)
        seen.add(token)
    i += 1

# DICTIONARY: templated entries + a few high-quality facts
DICTIONARY: Dict[str, Dict[str, Any]] = {}

DICTIONARY["george washington"] = {
    "definition": "The first President of the United States (1789–1797).",
    "type": "proper_noun",
    "examples": ["George Washington led the Continental Army to victory during the American Revolutionary War."]
}
DICTIONARY["abraham lincoln"] = {
    "definition": "The 16th President of the United States who led the nation through the Civil War.",
    "type": "proper_noun",
    "examples": ["Abraham Lincoln issued the Emancipation Proclamation in 1863."]
}
DICTIONARY["paris"] = {"definition":"Capital of France.","type":"proper_noun","examples":["Paris is famous for the Eiffel Tower."]}
DICTIONARY["jupiter"] = {"definition":"The largest planet in the Solar System.","type":"noun","examples":["Jupiter is a gas giant."]}
DICTIONARY["pi"] = {"definition":"Mathematical constant π ≈ 3.14159.","type":"number","examples":["Pi is used to compute circle measurements."]}
DICTIONARY["python"] = {"definition":"A high-level programming language.","type":"noun","examples":["I wrote a script in Python."]}

for w in EMBED_WORDS:
    key = w.lower()
    if key in DICTIONARY:
        continue
    DICTIONARY[key] = {
        "definition": f"A common English word: '{key}'.",
        "type": "common",
        "examples": [f"Example usage of '{key}' in a sentence."]
    }

# merge external dictionary.json if exists
external = load_json(DICT_FILE, None)
if external and isinstance(external, dict):
    for k,v in external.items():
        DICTIONARY[k.lower()] = v

# -------------------------
# Knowledge base (KB) for Q/A
# -------------------------
KB: Dict[str, str] = {
    "who was the first president of the united states": "George Washington",
    "who was the first president": "George Washington",
    "capital of france": "Paris",
    "largest planet": "Jupiter",
    "what is pi": "Pi is approximately 3.14159",
    "who wrote hamlet": "William Shakespeare",
}

# -------------------------
# Tokenization and vectorizer
# -------------------------
WORD_RE = re.compile(r"[a-zA-Z']+")

def tokenize(text: str) -> List[str]:
    return WORD_RE.findall(str(text).lower())

def build_vocab() -> List[str]:
    vocab = set()
    for k,v in DICTIONARY.items():
        vocab.update(tokenize(k))
        vocab.update(tokenize(v.get("definition","")))
        for ex in v.get("examples",[]):
            vocab.update(tokenize(ex))
    for conv in ai_state.get("conversations", [])[-500:]:
        vocab.update(tokenize(conv.get("text","")))
    vocab.update(["what","is","who","define","means","calculate","time","date","how","why","when","where","math"])
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
# TinyNN implementation
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

    def train(self, dataset: List[Tuple[List[float], int]], epochs:int=60, lr:float=0.05):
        if not dataset:
            return
        for epoch in range(epochs):
            random.shuffle(dataset)
            for x_vec, label in dataset:
                h_in = add_vec(matvec(self.W1, x_vec), self.b1)
                h = tanh_vec(h_in)
                o_in = add_vec(matvec(self.W2, h), self.b2)
                out = softmax(o_in)
                y = [0.0]*len(out); y[label] = 1.0
                err_out = [out[i] - y[i] for i in range(len(out))]
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
                        s += self.W2[i][j] * err_out[i]
                    err_hidden[j] = s * (1.0 - h[j]*h[j])
                # update W1,b1
                for j in range(len(self.W1)):
                    for k in range(len(self.W1[0])):
                        self.W1[j][k] -= lr * err_hidden[j] * x_vec[k]
                    self.b1[j] -= lr * err_hidden[j]

# -------------------------
# Intents, examples, training builder
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
        data.append((text_to_vector(text, vocab), INTENTS.index(intent)))
    for k,v in ai_state.get("learned", {}).items():
        phrase = f"{k} means {v.get('definition','')}"
        data.append((text_to_vector(phrase, vocab), INTENTS.index("teach")))
    return data

# -------------------------
# Markov generator
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
# Retrieval + lookup helpers
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
            left = m.group(1).strip()
            right = m.group(2).strip().rstrip(".")
            left_token = left.split()[0]
            return normalize_key(left_token), right
    return None, None

def retrieve_from_memory_or_learned(query: str) -> Optional[str]:
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

def lookup_kb(query: str) -> Tuple[Optional[str], float]:
    q = normalize_key(query.strip("? "))
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
    for k,v in ai_state.get("learned", {}).items():
        if normalize_key(k) in q or normalize_key(q) in k:
            return v.get("definition",""), 0.85
    return None, 0.0

# -------------------------
# Model build & training (heavy initial)
# -------------------------
VOCAB: List[str] = []
NN_MODEL: Optional[TinyNN] = None

def build_and_train_model():
    global VOCAB, NN_MODEL
    VOCAB = build_vocab()
    hidden = max(64, max(16, len(VOCAB)//8))
    NN_MODEL = TinyNN(len(VOCAB), hidden, len(INTENTS))
    dataset = build_training(VOCAB)
    if dataset:
        NN_MODEL.train(dataset, epochs=160, lr=0.06)

build_and_train_model()

def incremental_retrain():
    build_and_train_model()

# -------------------------
# Compose reply (core)
# -------------------------
def format_definition(key: str, entry: Dict[str,Any]) -> str:
    ex = entry.get("examples", [])
    ex_text = ("\nExamples:\n - " + "\n - ".join(ex)) if ex else ""
    return f"{key} ({entry.get('type','')}): {entry.get('definition','')}{ex_text}"

def safe_eval_math(expr: str):
    try:
        filtered = re.sub(r"[^0-9\.\+\-\*\/\%\(\)\s\^]", "", expr)
        if not re.search(r"\d", filtered):
            return None
        filtered = filtered.replace("^", "**")
        result = eval(filtered, {"__builtins__": None}, {"math": math, **{k:getattr(math,k) for k in dir(math) if not k.startswith("_")}})
        return result
    except Exception:
        return None

def merged_dictionary() -> Dict[str, Dict[str,Any]]:
    d = {**DICTIONARY}
    for k,v in ai_state.get("learned", {}).items():
        d[k.lower()] = {"definition": v.get("definition",""), "type": v.get("type","learned"), "examples": v.get("examples", [])}
    return d

def compose_reply(user_text: str) -> Dict[str,Any]:
    user = user_text.strip()
    lower = user.lower()

    # commands
    if lower in ("/clear", "clear chat"):
        ai_state["conversations"].clear()
        save_json(STATE_FILE, ai_state)
        train_markov()
        return {"reply":"Chat cleared.", "meta":{"intent":"memory"}}

    if lower in ("/forget", "forget"):
        ai_state["learned"].clear()
        save_json(STATE_FILE, ai_state)
        incremental_retrain(); train_markov()
        return {"reply":"Learned items forgotten.", "meta":{"intent":"memory"}}

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

    # safe math
    math_res = safe_eval_math(user)
    if math_res is not None:
        return {"reply": f"Math result: {math_res}", "meta":{"intent":"math"}}

    # time/date direct
    if re.search(r"\bwhat(?:'s| is)? the time\b|\btime now\b|\bcurrent time\b", lower):
        return {"reply": f"The current time is {datetime.now().strftime('%H:%M:%S')}", "meta":{"intent":"time"}}
    if re.search(r"\bwhat(?:'s| is)? the date\b|\bcurrent date\b|\bdate today\b", lower):
        return {"reply": f"Today's date is {datetime.now().strftime('%Y-%m-%d')}", "meta":{"intent":"date"}}

    # explicit define
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

    # natural teaching patterns
    w, d = try_extract_definition(user)
    if w and d:
        ai_state.setdefault("learned", {})[w] = {"definition": d, "type":"learned", "examples": []}
        save_json(STATE_FILE, ai_state)
        incremental_retrain(); train_markov()
        return {"reply": f"Saved learned definition: '{w}' = {d}", "meta":{"intent":"learning"}}

    # classification by NN
    if VOCAB and NN_MODEL:
        xvec = text_to_vector(user, VOCAB)
        intent_idx = NN_MODEL.predict(xvec)
        intent = INTENTS[intent_idx]
    else:
        intent = "chat"

    # handle intents
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
        return {"reply":"I don't have that definition yet. Teach me with '/define word: meaning'.", "meta":{"intent":"definition"}}

    if intent == "time":
        return {"reply": f"The current time is {datetime.now().strftime('%H:%M:%S')}", "meta":{"intent":"time"}}
    if intent == "date":
        return {"reply": f"Today's date is {datetime.now().strftime('%Y-%m-%d')}", "meta":{"intent":"date"}}
    if intent == "math":
        if math_res is not None:
            return {"reply": f"Math result: {math_res}", "meta":{"intent":"math"}}

    # retrieval
    mem = retrieve_from_memory_or_learned(user)
    if mem:
        return {"reply": mem, "meta":{"intent":"memory"}}

    # markov fallback
    gen = MARKOV.generate(seed=user, max_words=50)
    if gen:
        return {"reply": gen.capitalize() + ".", "meta":{"intent":"gen"}}

    return {"reply":"I don't know that yet. Teach me with 'X means Y' or '/define X: Y'.", "meta":{"intent":"unknown"}}

# -------------------------
# Convenience: teach via API
# -------------------------
def teach_definition(key: str, definition: str):
    k = normalize_key(key)
    ai_state.setdefault("learned", {})[k] = {"definition": definition, "type":"learned", "examples": []}
    save_json(STATE_FILE, ai_state)
    incremental_retrain(); train_markov()
    return f"Learned '{k}'."

# -------------------------
# Export/import utilities
# -------------------------
def export_state(path: str = "ai_state_export.json"):
    save_json(path, ai_state)
    return path

def import_dictionary(path: str):
    try:
        with open(path, "r", encoding="utf-8") as f:
            ext = json.load(f)
        if isinstance(ext, dict):
            for k,v in ext.items():
                DICTIONARY[k.lower()] = v
            incremental_retrain(); train_markov()
            return True, "Merged dictionary."
        return False, "dictionary must be a JSON object mapping words to entries."
    except Exception as e:
        return False, str(e)

# -------------------------
# CLI REPL when run directly
# -------------------------
def repl():
    print("Jack — Offline AI core (no interface). Type '/help' for commands.")
    while True:
        try:
            s = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break
        if not s:
            continue
        if s in ("/exit","/quit"):
            print("Goodbye.")
            break
        if s == "/help":
            print("Commands:\n  /help\n  /exit or /quit\n  /define word: meaning  (teach)\n  /clear  (clear chat)\n  /forget (forget learned)\n  /export [path]\n  /importdict path/to/file.json\n  /history  (show conversation history)\n  /delete N  (delete conversation number N)\n")
            continue
        if s.startswith("/export"):
            parts = s.split(None,1)
            path = parts[1].strip() if len(parts)>1 else "ai_state_export.json"
            p = export_state(path)
            print(f"Exported state to {p}")
            continue
        if s.startswith("/importdict"):
            parts = s.split(None,1)
            if len(parts) < 2:
                print("Usage: /importdict path/to/dictionary.json")
                continue
            ok, msg = import_dictionary(parts[1].strip())
            print(msg)
            continue
        if s == "/history":
            for i,conv in enumerate(ai_state.get("conversations",[]),1):
                role = conv.get("role","")
                text = conv.get("text","")
                print(f"{i}. {role}: {text}")
            continue
        # handle delete command
        if s.startswith("/delete "):
            out = compose_reply(s)  # compose_reply handles /delete too
            print("Jack:", out.get("reply",""))
            continue
        # teach shortcut
        if s.startswith("/define ") or " means " in s or " is " in s:
            out = compose_reply(s)
            print("Jack:", out.get("reply",""))
            ai_state.setdefault("conversations", []).append({"role":"user","text":s,"time":datetime.now().isoformat()})
            ai_state.setdefault("conversations", []).append({"role":"assistant","text":out.get("reply",""),"time":datetime.now().isoformat()})
            save_json(STATE_FILE, ai_state)
            continue
        # normal query
        out = compose_reply(s)
        print("Jack:", out.get("reply",""))
        ai_state.setdefault("conversations", []).append({"role":"user","text":s,"time":datetime.now().isoformat()})
        ai_state.setdefault("conversations", []).append({"role":"assistant","text":out.get("reply",""),"time":datetime.now().isoformat()})
        save_json(STATE_FILE, ai_state)

# -------------------------
# Only run repl if executed directly
# -------------------------
if __name__ == "__main__":
    repl()
