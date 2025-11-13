# jack_offline_fast_start.py
# Faster startup: lazy-load big dictionary and on-demand model rebuild.
# Run:
#   pip install streamlit
#   streamlit run jack_offline_fast_start.py

import streamlit as st
import json, os, re, math, random
from datetime import datetime
from typing import List, Dict, Tuple, Any, Optional

# -------------------------
# Files & Persistence
# -------------------------
STATE_FILE = "ai_state.json"
DICT_FILE = "dictionary.json"       # persisted large dict (generated once on demand)
MARKOV_FILE = "markov_state.json"

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

# persistent small state (convos, learned, flags)
ai_state = load_json(STATE_FILE, {"conversations": [], "learned": {}, "settings": {}, "model_dirty": True})

# -------------------------
# Light-weight tokenizer (keeps punctuation separate)
# -------------------------
WORD_RE = re.compile(r"[A-Za-z']+|[.,!?:;]")

def tokenize(text: str) -> List[str]:
    return WORD_RE.findall((text or "").lower())

# -------------------------
# Minimal base dictionary loaded at startup (keeps app light)
# -------------------------
MINI_BASE_DICT = {
    "i": {"definition":"first-person pronoun","type":"pronoun","examples":["i went home.","i think that's correct."]},
    "you": {"definition":"second-person pronoun","type":"pronoun","examples":["you are kind.","can you help me?"]},
    "the": {"definition":"definite article","type":"article","examples":["the book is on the table.","the sky is blue."]},
    "a": {"definition":"indefinite article","type":"article","examples":["a dog barked.","a good idea."]},
    "and": {"definition":"conjunction","type":"conj","examples":["bread and butter.","he and she."]},
    # small curated set to keep startup fast
    "eat": {"definition":"to consume food","type":"verb","examples":["i eat breakfast.","eat slowly."]},
    "drink": {"definition":"to consume liquid","type":"verb","examples":["drink water.","he drinks coffee."]},
    "food": {"definition":"substances eaten for nutrition","type":"noun","examples":["food is necessary.","fresh food tastes good."]},
    "__corpus__": {"definition":"small starter corpus","type":"corpus","examples":["the cat sat on the mat.","do you like apples?"]}
}

# If a full dictionary file exists, we'll lazily load it when needed.
BASE_DICT = MINI_BASE_DICT.copy()

def merged_dictionary() -> Dict[str, Dict[str,Any]]:
    """Merge base dict with learned items."""
    d = {k.lower(): dict(v) for k,v in BASE_DICT.items()}
    for k,v in ai_state.get("learned", {}).items():
        d[k.lower()] = {"definition": v.get("definition",""), "type": v.get("type","learned"), "examples": v.get("examples",[])}
    return d

# -------------------------
# Compact KB (keeps small for fast startup)
# -------------------------
KB = {
    "capital of france": "Paris.",
    "what is pi": "Pi (π) ≈ 3.14159.",
    "what is python": "Python is a high-level programming language.",
    "who wrote hamlet": "William Shakespeare."
}

# -------------------------
# Lightweight vectorizer & vocab builder (on demand)
# -------------------------
_cached_vocab = []
_cached_key = None

def build_vocab(force: bool=False) -> List[str]:
    global _cached_vocab, _cached_key
    md = merged_dictionary()
    key = (len(md), len(ai_state.get("learned",{})), len(ai_state.get("conversations",[])))
    if not force and _cached_vocab and key == _cached_key:
        return _cached_vocab
    vocab = set()
    for k,v in md.items():
        vocab.update(tokenize(k))
        vocab.update(tokenize(v.get("definition","")))
        for ex in v.get("examples",[]):
            vocab.update(tokenize(ex))
    for c in ai_state.get("conversations", [])[-200:]:
        vocab.update(tokenize(c.get("text","")))
    vocab.update(["what","who","when","where","why","how","define","time","date"])
    _cached_vocab = sorted(vocab)
    _cached_key = key
    return _cached_vocab

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
# TinyNN (very small & trained only on demand)
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

    def forward(self, x):
        h_in = add_vec(matvec(self.W1, x), self.b1)
        h = tanh_vec(h_in)
        o_in = add_vec(matvec(self.W2, h), self.b2)
        out = softmax(o_in)
        return h, out

    def predict(self, x):
        _, out = self.forward(x)
        return max(range(len(out)), key=lambda i: out[i])

    def train(self, dataset, epochs=8, lr=0.06):
        if not dataset: return
        for _ in range(epochs):
            random.shuffle(dataset)
            for x_vec, label in dataset:
                h_in = add_vec(matvec(self.W1, x_vec), self.b1)
                h = tanh_vec(h_in)
                o_in = add_vec(matvec(self.W2, h), self.b2)
                out = softmax(o_in)
                y = [0.0]*len(out); y[label] = 1.0
                err_out = [out[i] - y[i] for i in range(len(out))]
                for i in range(len(self.W2)):
                    for j in range(len(self.W2[0])):
                        self.W2[i][j] -= lr * err_out[i] * h[j]
                    self.b2[i] -= lr * err_out[i]
                err_hidden = [0.0]*len(h)
                for j in range(len(h)):
                    s = 0.0
                    for i in range(len(err_out)):
                        s += self.W2[i][j] * err_out[i]
                    err_hidden[j] = s * (1.0 - h[j]*h[j])
                for j in range(len(self.W1)):
                    for k in range(len(self.W1[0])):
                        self.W1[j][k] -= lr * err_hidden[j] * x_vec[k]
                    self.b1[j] -= lr * err_hidden[j]

# placeholders (constructed on rebuild)
VOCAB: List[str] = []
NN_MODEL: Optional[TinyNN] = None

# -------------------------
# Markov (lightweight and fast; training sampled on rebuild)
# -------------------------
class Markov:
    def __init__(self):
        self.map = {}
        self.starts = []

    def train(self, text: str):
        toks = tokenize(text)
        if len(toks) < 3: return
        self.starts.append((toks[0].lower(), toks[1].lower()))
        for i in range(len(toks)-2):
            key = (toks[i].lower(), toks[i+1].lower())
            nxt = toks[i+2].lower()
            self.map.setdefault(key, {})
            self.map[key][nxt] = self.map[key].get(nxt, 0) + 1

    def generate(self, seed=None, max_words=40, capitalize_if=True):
        # simple, fast backoff: choose best next by frequency
        seed_tokens = set(tokenize(seed)) if seed else set()
        if seed:
            toks = tokenize(seed)
            if len(toks) >= 2:
                key = (toks[-2].lower(), toks[-1].lower())
                if key not in self.map:
                    # quick backoff: pick a random start
                    key = random.choice(self.starts) if self.starts else None
                if key:
                    out = [key[0], key[1]]
                    for _ in range(max_words-2):
                        choices = self.map.get((out[-2], out[-1]), {})
                        if not choices: break
                        nxt = max(choices.items(), key=lambda kv: kv[1])[0]
                        out.append(nxt)
                        if re.fullmatch(r"[\.!\?;,:]", nxt):
                            break
                    s = " ".join(out)
                    s = re.sub(r"\s+([,\.\?!;:])", r"\1", s)
                    if capitalize_if and s and s[0].isalpha():
                        s = s[0].upper() + s[1:]
                    if not re.search(r"[\.!\?]$", s):
                        s = s + "."
                    return s
        # random start fallback
        if not self.starts:
            return ""
        key = random.choice(self.starts)
        out = [key[0], key[1]]
        for _ in range(max_words-2):
            choices = self.map.get((out[-2], out[-1]), {})
            if not choices: break
            nxt = max(choices.items(), key=lambda kv: kv[1])[0]
            out.append(nxt)
            if re.fullmatch(r"[\.!\?;,:]", nxt):
                break
        s = " ".join(out)
        s = re.sub(r"\s+([,\.\?!;:])", r"\1", s)
        if s and s[0].isalpha():
            s = s[0].upper() + s[1:]
        if not re.search(r"[\.!\?]$", s):
            s = s + "."
        return s

    def generate_paragraph(self, seed=None, topic=None, num_sentences=3):
        sentences = []
        cur = seed
        for i in range(num_sentences):
            s = self.generate(seed=cur, capitalize_if=(i==0))
            if not s:
                break
            sentences.append(s)
            toks = tokenize(s)
            if len(toks) >= 2:
                cur = " ".join(toks[-2:])
            else:
                cur = None
        return " ".join(sentences)

MARKOV = Markov()

def load_markov_if_exists():
    ser = load_json(MARKOV_FILE, None)
    if ser and isinstance(ser, dict) and "map" in ser:
        # deserialize
        starts = ser.get("starts", [])
        m = {}
        for k,v in ser.get("map", {}).items():
            a,b = k.split("||")
            m[(a,b)] = v
        MARKOV.starts = starts
        MARKOV.map = m
        return True
    return False

# try load persisted markov (fast path) — if available we avoid rebuilding
_markov_loaded = load_markov_if_exists()

# -------------------------
# Fast sampled Markov training used only when rebuilding
# -------------------------
def sampled_markov_train(limit_examples=2000):
    MARKOV.map.clear(); MARKOV.starts.clear()
    md = merged_dictionary()
    # gather examples — we will sample to keep training fast
    examples = []
    for k,v in md.items():
        for ex in v.get("examples", []):
            examples.append(ex)
        examples.append(k + " " + v.get("definition",""))
    # include conversation history too
    examples.extend(c.get("text","") for c in ai_state.get("conversations", []))
    random.shuffle(examples)
    # cap examples to limit for speed
    for ex in examples[:limit_examples]:
        MARKOV.train(ex)
    # persist
    try:
        ser = {"starts": MARKOV.starts, "map": {f"{a}||{b}":nxts for (a,b),nxts in MARKOV.map.items()}}
        save_json(MARKOV_FILE, ser)
    except Exception:
        pass

# -------------------------
# Build & train model (on demand via UI) — fast settings
# -------------------------
INTENTS = ["define","fact","math","time","date","teach","chat"]
SEED_EXAMPLES = [
    ("what is gravity", "fact"),
    ("who was the first president of the united states", "fact"),
    ("define gravity", "define"),
    ("calculate 12 * 7", "math"),
    ("what time is it", "time"),
    ("what is today's date", "date"),
    ("tell me a story", "chat"),
]

def build_and_train_model(force: bool=False):
    global VOCAB, NN_MODEL
    VOCAB = build_vocab(force=force)
    # small hidden layer for speed
    hidden_dim = max(16, len(VOCAB)//20 or 16)
    NN_MODEL = TinyNN(len(VOCAB), hidden_dim, len(INTENTS))
    dataset = []
    for text,intent in SEED_EXAMPLES:
        dataset.append((text_to_vector(text, VOCAB), INTENTS.index(intent)))
    for k,v in ai_state.get("learned", {}).items():
        phrase = f"{k} means {v.get('definition','')}"
        dataset.append((text_to_vector(phrase, VOCAB), INTENTS.index("teach")))
    if dataset:
        NN_MODEL.train(dataset, epochs=6, lr=0.06)
    # sample Markov training to be fast
    sampled_markov_train(limit_examples=1500)
    ai_state["model_dirty"] = False
    save_json(STATE_FILE, ai_state)

# -------------------------
# Utilities (definitions/learn)
# -------------------------
LEARN_PATTERNS = [
    re.compile(r'^\s*define\s+([^\:]+)\s*[:\-]\s*(.+)$', re.I),
    re.compile(r'^\s*([A-Za-z\'\-\s]+)\s+means\s+(.+)$', re.I),
    re.compile(r'^\s*([A-Za-z\'\-\s]+)\s+is\s+(.+)$', re.I),
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
# Compose reply
# -------------------------
def safe_eval_math(expr: str):
    try:
        filtered = re.sub(r"[^0-9\.\+\-\*\/\%\(\)\s\^]", "", expr)
        if not re.search(r"\d", filtered): return None
        filtered = filtered.replace("^", "**")
        result = eval(filtered, {"__builtins__": None}, {"math": math})
        return result
    except Exception:
        return None

def format_definition(key: str, entry: Dict[str,Any]) -> str:
    ex = entry.get("examples", [])
    ex_text = ("\nExamples:\n - " + "\n - ".join(ex)) if ex else ""
    return f"{key} ({entry.get('type','')}): {entry.get('definition','')}{ex_text}"

def compose_reply(user_text: str, topic: Optional[str]=None, paragraph_sentences: Optional[int]=None) -> Dict[str,Any]:
    user = user_text.strip()
    lower = user.lower()
    # command handlers
    if lower in ("/clear", "clear chat"):
        ai_state["conversations"].clear(); save_json(STATE_FILE, ai_state); return {"reply":"Chat cleared.","meta":{"intent":"memory"}}
    if lower in ("/forget", "forget"):
        ai_state["learned"].clear(); save_json(STATE_FILE, ai_state); ai_state["model_dirty"]=True; save_json(STATE_FILE, ai_state); return {"reply":"Learned memory cleared.","meta":{"intent":"memory"}}
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
                ai_state["learned"].pop(key); save_json(STATE_FILE, ai_state); ai_state["model_dirty"]=True; save_json(STATE_FILE, ai_state); return {"reply": f"Removed learned definition for '{key}'.", "meta":{"intent":"memory"}}
            else:
                return {"reply": f"No learned definition for '{key}'.", "meta":{"intent":"error"}}
    # math
    math_res = safe_eval_math(user)
    if math_res is not None:
        return {"reply": f"Math result: {math_res}", "meta":{"intent":"math"}}
    # time/date
    if re.search(r"\bwhat(?:'s| is)? the time\b|\btime now\b|\bcurrent time\b", lower):
        return {"reply": f"The current time is {datetime.now().strftime('%H:%M:%S')}", "meta":{"intent":"time"}}
    if re.search(r"\bwhat(?:'s| is)? the date\b|\bcurrent date\b|\bdate today\b", lower):
        return {"reply": f"Today's date is {datetime.now().strftime('%Y-%m-%d')}", "meta":{"intent":"date"}}
    # define command
    if lower.startswith("/define ") or lower.startswith("define "):
        rest = user.split(None,1)[1] if len(user.split(None,1))>1 else ""
        m = re.match(r'\s*([^\:]+)\s*[:\-]\s*(.+)', rest)
        if m:
            w = normalize_key(m.group(1)); d = m.group(2).strip()
            ai_state.setdefault("learned", {})[w] = {"definition": d, "type":"learned", "examples": []}
            save_json(STATE_FILE, ai_state)
            ai_state["model_dirty"] = True; save_json(STATE_FILE, ai_state)
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
    # natural teach patterns
    w,d = try_extract_definition(user)
    if w and d:
        ai_state.setdefault("learned", {})[w] = {"definition": d, "type":"learned", "examples": []}
        save_json(STATE_FILE, ai_state)
        ai_state["model_dirty"] = True; save_json(STATE_FILE, ai_state)
        return {"reply": f"Saved learned definition: '{w}' = {d}", "meta":{"intent":"learning"}}
    # quick KB lookup
    ans, conf = lookup_kb(user)
    if ans:
        return {"reply": str(ans), "meta":{"intent":"fact","confidence":conf}}
    # retrieval
    mem = retrieve_from_memory_or_learned(user)
    if mem:
        return {"reply": mem, "meta":{"intent":"memory"}}
    # paragraph generation if requested
    if paragraph_sentences and paragraph_sentences > 0:
        para = MARKOV.generate_paragraph(seed=(user if user else None), topic=topic, num_sentences=paragraph_sentences)
        if para:
            return {"reply": para, "meta":{"intent":"gen_paragraph"}}
    # markov single generation
    gen = MARKOV.generate(seed=user)
    if gen:
        if re.search(r"[\.!\?]\s*$", user):
            reply_text = gen
        else:
            reply_text = (user.rstrip() + " " + gen).strip()
        return {"reply": reply_text, "meta":{"intent":"gen"}}
    return {"reply": "I don't know that yet. Teach me with 'X means Y' or '/define X: Y'.", "meta":{"intent":"unknown"}}

# -------------------------
# Dictionary generation (expensive) — run only when user asks
# -------------------------
def generate_large_dictionary(min_entries: int=2000):
    # simple programmatic builder (similar to earlier blocks) but run only when triggered
    def examples_for(word, typ):
        if typ == "noun":
            return [f"the {word} is on the table.", f"i saw a {word} yesterday."]
        if typ == "verb":
            return [f"i {word} every day.", f"please {word} carefully."]
        if typ == "adj":
            return [f"that is very {word}.", f"the {word} example."]
        if typ == "adv":
            return [f"do it {word}.", f"they moved {word}"]
        if typ == "food":
            return [f"i like {word}.", f"{word} is delicious."]
        return [f"{word} example."]
    BASE = {}
    # seeds
    seeds = []
    # small curated lists (shortened here for speed but plenty to reach 2k after morphs)
    seeds += "cat dog bird fish cow horse goat pig chicken turkey apple banana orange grape bread cheese rice pasta pizza salad soup sandwich tomato potato carrot onion garlic pepper".split()
    seeds += "eat drink cook bake boil fry chop slice mix stir serve taste walk run jump drive read write play learn teach think know make get take give find see watch listen create build open close start stop continue".split()
    seeds += "red blue green black white yellow pink purple brown gray silver gold".split()
    # add more stems by programmatic families (numbers, materials, body parts, places...)
    seeds += "meter kilometer gram kilogram liter ounce pound inch foot yard mile second minute hour day week month year".split()
    seeds += "wood metal plastic glass stone brick concrete paper cloth leather cotton silk wool linen".split()
    seeds += "head face eye ear nose mouth neck shoulder arm hand finger leg knee foot toe".split()
    seeds += "paris london berlin rome madrid tokyo beijing moscow washington ottawa canberra".split()
    for s in seeds:
        s = s.lower().replace(" ", "_")
        if s not in BASE:
            typ = "noun"
            if s in ("eat","drink","cook","bake","boil","fry","chop","slice","mix","stir","serve","taste","walk","run","jump","drive","read","write","play","learn","teach","think","know","make","get","take","give"):
                typ = "verb"
            BASE[s] = {"definition": f"{typ} {s.replace('_',' ')}", "type": typ, "examples": examples_for(s.replace("_"," "), typ)}
    # morphological variants to reach target
    def make_plural(w):
        if w.endswith(("s","x","z","ch","sh")): return w + "es"
        if w.endswith("y") and len(w)>1 and w[-2] not in "aeiou": return w[:-1] + "ies"
        return w + "s"
    def make_ing(w):
        if w.endswith("ie"): return w[:-2] + "ying"
        if w.endswith("e") and not w.endswith("ee"): return w[:-1] + "ing"
        if len(w)>=3 and (w[-1] not in "aeiou" and w[-2] in "aeiou" and w[-3] not in "aeiou"): return w + w[-1] + "ing"
        return w + "ing"
    i = 0
    words = list(BASE.keys())
    while len(BASE) < min_entries and i < 20000:
        base = words[i % len(words)]
        i += 1
        v = make_plural(base)
        if v not in BASE and len(v) < 40:
            BASE[v] = {"definition": f"plural of {base.replace('_',' ')}", "type":"noun", "examples":[f"the {v.replace('_',' ')} are here."]}
        # verbs -> ing, ed
        if BASE[base]["type"] == "verb":
            ing = make_ing(base)
            if ing not in BASE:
                BASE[ing] = {"definition": f"{ing} (form)", "type":"verb", "examples":[f"i am {ing.replace('_',' ')}."]}
        words = list(BASE.keys())
    # ensure fill if still short
    cntr = 1
    while len(BASE) < min_entries:
        key = f"term_{cntr}"
        if key not in BASE:
            BASE[key] = {"definition": f"synthetic filler {cntr}", "type":"noun", "examples":[f"{key} example."]}
        cntr += 1
    return BASE

# -------------------------
# UI & controls
# -------------------------
st.set_page_config(page_title="Jack — Fast Start", layout="wide")
st.title("Jack — Fast-start Offline AI")

left, right = st.columns([3,1])

with right:
    st.header("Performance Controls")
    st.markdown("This app defers heavy work to you. Click buttons below when you want the big dictionary or a model rebuild.")
    if st.button("Load full dictionary (generate or load dictionary.json)"):
        # if dictionary file exists, load it; otherwise generate and save
        if os.path.exists(DICT_FILE):
            try:
                big = load_json(DICT_FILE, None)
                if isinstance(big, dict) and len(big) > 1000:
                    BASE_DICT.update({k:v for k,v in big.items()})
                    st.success(f"Loaded dictionary.json with {len(big)} entries.")
                else:
                    st.warning("dictionary.json exists but looks small or invalid; regenerating.")
                    with st.spinner("Generating large dictionary (one-time): this may take a few seconds..."):
                        big = generate_large_dictionary(min_entries=2000)
                        save_json(DICT_FILE, big)
                        BASE_DICT.update({k:v for k,v in big.items()})
                        st.success(f"Generated and saved dictionary.json with {len(big)} entries.")
            except Exception as e:
                st.error(f"Failed to load dictionary.json: {e}")
        else:
            with st.spinner("Generating large dictionary (one-time): this may take a few seconds..."):
                big = generate_large_dictionary(min_entries=2000)
                save_json(DICT_FILE, big)
                BASE_DICT.update({k:v for k,v in big.items()})
                st.success(f"Generated and saved dictionary.json with {len(big)} entries.")
        # Mark model dirty to reflect new vocab
        ai_state["model_dirty"] = True; save_json(STATE_FILE, ai_state)

    st.markdown("---")
    st.write("Model status:")
    if ai_state.get("model_dirty", False):
        st.warning("Model marked DIRTY — rebuild recommended.")
    else:
        st.success("Model up-to-date.")

    if st.button("Rebuild Model (train small NN + sample-Markov)"):
        with st.spinner("Rebuilding (fast) — this should be quick..."):
            build_and_train_model(force=True)
            st.success("Rebuild complete — model_dirty cleared.")
            st.rerun()

    st.markdown("---")
    st.write("Persisted Markov: " + ("loaded" if _markov_loaded else "not found"))
    if st.button("Clear learned memories"):
        ai_state["learned"].clear(); save_json(STATE_FILE, ai_state); st.success("Learned cleared."); st.rerun()

with left:
    st.subheader("Conversation")
    history = ai_state.get("conversations", [])[-200:]
    for m in history:
        who = "You" if m.get("role","user")=="user" else "Jack"
        t = m.get("time","")
        st.markdown(f"**{who}**  <span style='color:gray;font-size:12px'>{t}</span>", unsafe_allow_html=True)
        st.write(m.get("text",""))

    st.markdown("---")
    user_input = st.text_area("Message (Shift+Enter = newline)", height=120)
    topic_input = st.text_input("Topic (optional; used for paragraph biasing)", value="")
    num_sentences = st.slider("Paragraph length (sentences)", min_value=1, max_value=6, value=2)

    c1,c2,c3 = st.columns([1,1,1])
    if c1.button("Send"):
        ui = user_input.strip()
        if ui:
            out = compose_reply(ui)
            reply = out.get("reply","")
            ai_state.setdefault("conversations", []).append({"role":"user","text":ui,"time":datetime.now().isoformat()})
            ai_state.setdefault("conversations", []).append({"role":"assistant","text":reply,"time":datetime.now().isoformat()})
            save_json(STATE_FILE, ai_state)
            # light on-the-fly training: add the user+reply to markov to improve future generations
            MARKOV.train(ui); MARKOV.train(reply)
            ai_state["model_dirty"] = True; save_json(STATE_FILE, ai_state)
            st.rerun()

    if c2.button("Generate Paragraph"):
        ui = user_input.strip()
        para = MARKOV.generate_paragraph(seed=(ui if ui else None), topic=(topic_input or None), num_sentences=num_sentences)
        if para:
            if ui:
                ai_state.setdefault("conversations", []).append({"role":"user","text":ui,"time":datetime.now().isoformat()})
            ai_state.setdefault("conversations", []).append({"role":"assistant","text":para,"time":datetime.now().isoformat()})
            save_json(STATE_FILE, ai_state)
            MARKOV.train(para)
            ai_state["model_dirty"] = True; save_json(STATE_FILE, ai_state)
            st.rerun()

    if c3.button("Teach (word: definition)"):
        ui = user_input.strip()
        m = re.match(r'\s*([^\:]+)\s*[:\-]\s*(.+)', ui)
        if m:
            w = normalize_key(m.group(1)); d = m.group(2).strip()
            ai_state.setdefault("learned", {})[w] = {"definition": d, "type":"learned", "examples": []}
            save_json(STATE_FILE, ai_state)
            ai_state["model_dirty"] = True; save_json(STATE_FILE, ai_state)
            st.success(f"Learned '{w}'.")
            st.rerun()
        else:
            st.warning("To teach: enter `word: definition` (e.g. gravity: a force that pulls)")

st.markdown("---")
st.markdown("**Examples / Commands**")
st.markdown("""
- Ask a fact: `Who was the first president of the U.S.?`  
- Teach: `gravity means a force that pulls` or `/define gravity: a force that pulls`  
- Paragraph: type a seed (optional) and a Topic, then click **Generate Paragraph**  
- Commands: `/clear` (clear conversation), `/forget` (clear learned memories), `/delete N` (delete convo #N)
""")
