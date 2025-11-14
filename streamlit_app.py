# omega_b_ephemeral_mode.py
# Ephemeral-mode version: heavy assets kept in memory and remade when reset.
# Run:
#   pip install streamlit scikit-learn
#   streamlit run omega_b_ephemeral_mode.py

import streamlit as st
import json, os, re, math, random, uuid, threading, hashlib, time
from datetime import datetime
from typing import List, Dict, Tuple, Any, Optional

# -------------------------
# EPHEMERAL MODE CONFIG
# -------------------------
# If True -> do NOT write large files to disk; everything rebuilt in memory on reset/restart.
# If False -> legacy behavior: save/load device-scoped files.
EPHEMERAL_MODE = True

# -------------------------
# Small helpers for ephemeral in-memory storage (when EPHEMERAL_MODE=True)
# -------------------------
def _ephemeral_store():
    """Return a dict used as in-memory 'filesystem' for ephemeral mode."""
    return st.session_state.setdefault("_ephemeral_store", {})

def load_json(path: str, default):
    if EPHEMERAL_MODE:
        return _ephemeral_store().get(path, default)
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return default

def save_json(path: str, data):
    if EPHEMERAL_MODE:
        _ephemeral_store()[path] = data
        return
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print("Failed saving", path, e)

def remove_json(path: str):
    if EPHEMERAL_MODE:
        _ephemeral_store().pop(path, None)
    else:
        try:
            if os.path.exists(path):
                os.remove(path)
        except Exception:
            pass

# -------------------------
# Device-specific isolation (per-device conversations + privacy)
# -------------------------
DEVICE_ID_FILE = ".jack_device_id"

def get_or_create_device_id():
    try:
        if not EPHEMERAL_MODE and os.path.exists(DEVICE_ID_FILE):
            with open(DEVICE_ID_FILE, "r", encoding="utf-8") as f:
                return f.read().strip()
        new_id = str(uuid.uuid4())[:8]
        if not EPHEMERAL_MODE:
            try:
                with open(DEVICE_ID_FILE, "w", encoding="utf-8") as f:
                    f.write(new_id)
            except Exception:
                pass
        return new_id
    except Exception:
        sid = st.session_state.get("_sid")
        if not sid:
            sid = hashlib.sha1(str(uuid.uuid4()).encode()).hexdigest()[:8]
            st.session_state["_sid"] = sid
        return sid

DEVICE_ID = get_or_create_device_id()

# filenames (only used when EPHEMERAL_MODE=False)
STATE_FILE = f"ai_state_{DEVICE_ID}.json"
DICT_FILE = f"dictionary_{DEVICE_ID}.json"
MARKOV_FILE = f"markov_state_{DEVICE_ID}.json"
KB_FILE = f"kb_massive_{DEVICE_ID}.json"
COUNTRIES_FILE = f"countries_capitals_{DEVICE_ID}.json"

print(f"[Omega-B] Ephemeral mode = {EPHEMERAL_MODE} — device ID: {DEVICE_ID}")

# persistent small state: if ephemeral, keep in session_state; otherwise load from disk
if EPHEMERAL_MODE:
    ai_state = st.session_state.setdefault("ai_state", {"conversations": [], "learned": {}, "settings": {"persona":"neutral"}, "model_dirty": True})
else:
    ai_state = load_json(STATE_FILE, {"conversations": [], "learned": {}, "settings": {"persona":"neutral"}, "model_dirty": True})

# When saving ai_state, obey EPHEMERAL_MODE semantics (in-memory vs file)
def save_state():
    if EPHEMERAL_MODE:
        st.session_state["ai_state"] = ai_state
    else:
        save_json(STATE_FILE, ai_state)

# -------------------------
# sklearn (TF-IDF semantic)
# -------------------------
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except Exception:
    try:
        os.system("pip install scikit-learn")
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
    except Exception:
        TfidfVectorizer = None
        cosine_similarity = None

_vectorizer = None
_matrix = None
_indexed_keys = []
_vector_lock = threading.Lock()

# -------------------------
# Tokenizer
# -------------------------
WORD_RE = re.compile(r"[A-Za-z']+|[.,!?:;]")

def tokenize(text: str) -> List[str]:
    return WORD_RE.findall((text or "").lower())

# -------------------------
# Mini dictionary (startup)
# -------------------------
MINI_BASE_DICT = {
    "i": {"definition":"first-person pronoun","type":"pronoun","examples":["i went home.","i think that's correct."]},
    "you": {"definition":"second-person pronoun","type":"pronoun","examples":["you are kind.","can you help me?"]},
    "the": {"definition":"definite article","type":"article","examples":["the book is on the table.","the sky is blue."]},
    "a": {"definition":"indefinite article","type":"article","examples":["a dog barked.","a good idea."]},
    "__corpus__": {"definition":"small starter corpus","type":"corpus","examples":["the cat sat on the mat.","do you like apples?"]}
}
BASE_DICT = MINI_BASE_DICT.copy()

def merged_dictionary() -> Dict[str, Dict[str,Any]]:
    d = {k.lower(): dict(v) for k,v in BASE_DICT.items()}
    for k,v in ai_state.get("learned", {}).items():
        d[k.lower()] = {"definition": v.get("definition",""), "type": v.get("type","learned"), "examples": v.get("examples",[])}
    return d

# -------------------------
# Compact KB (startup)
# -------------------------
KB = {
    "capital of france": "Paris.",
    "what is pi": "Pi (π) ≈ 3.14159.",
    "what is python": "Python is a high-level programming language.",
    "who wrote hamlet": "William Shakespeare."
}

# -------------------------
# Vocab / vectors helpers
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
# TinyNN (on-demand)
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

VOCAB: List[str] = []
NN_MODEL: Optional[TinyNN] = None

# -------------------------
# Markov (improved + in-memory only when ephemeral)
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

    def _best_choice(self, choices: Dict[str,int], seed_tokens:set) -> Optional[str]:
        best = None; best_score = -1
        for token, freq in choices.items():
            score = freq
            if token in seed_tokens:
                score += max(1, freq//2)
            for s in seed_tokens:
                if s and (s in token or token in s):
                    score += max(1, freq//3)
            if score > best_score:
                best_score = score; best = token
        return best

    def generate(self, seed=None, max_words=40, capitalize_if=True):
        seed_tokens = set(tokenize(seed)) if seed else set()
        if seed:
            toks = tokenize(seed)
            if len(toks) >= 2:
                key = (toks[-2].lower(), toks[-1].lower())
                if key not in self.map:
                    candidate_keys = [k for k in self.map.keys() if (k[0] in seed_tokens or k[1] in seed_tokens)]
                    key = random.choice(candidate_keys) if candidate_keys else (random.choice(self.starts) if self.starts else None)
                if key:
                    out = [key[0], key[1]]
                    for _ in range(max_words-2):
                        choices = self.map.get((out[-2], out[-1]), {})
                        if not choices: break
                        nxt = self._best_choice(choices, seed_tokens) or max(choices.items(), key=lambda kv: kv[1])[0]
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

# In ephemeral mode we do not persist markov; but if non-ephemeral try to load it
def load_markov_if_exists():
    if EPHEMERAL_MODE:
        return False
    ser = load_json(MARKOV_FILE, None)
    if ser and isinstance(ser, dict) and "map" in ser:
        starts = ser.get("starts", [])
        m = {}
        for k,v in ser.get("map", {}).items():
            a,b = k.split("||")
            m[(a,b)] = v
        MARKOV.starts = starts
        MARKOV.map = m
        return True
    return False

_markov_loaded = load_markov_if_exists()

def sampled_markov_train(limit_examples=2000):
    MARKOV.map.clear(); MARKOV.starts.clear()
    md = merged_dictionary()
    examples = []
    for k,v in md.items():
        for ex in v.get("examples", []):
            examples.append(ex)
        examples.append(k + " " + v.get("definition",""))
    examples.extend(c.get("text","") for c in ai_state.get("conversations", []))
    random.shuffle(examples)
    for ex in examples[:limit_examples]:
        MARKOV.train(ex)
    if not EPHEMERAL_MODE:
        try:
            ser = {"starts": MARKOV.starts, "map": {f"{a}||{b}":nxts for (a,b),nxts in MARKOV.map.items()}}
            save_json(MARKOV_FILE, ser)
        except Exception:
            pass

# -------------------------
# Build & train model (on demand)
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
    sampled_markov_train(limit_examples=1500)
    ai_state["model_dirty"] = False
    save_state()

# -------------------------
# Semantic helpers (TF-IDF)
# -------------------------
def rebuild_semantic_index(force: bool=False):
    global _vectorizer, _matrix, _indexed_keys
    if TfidfVectorizer is None:
        return
    with _vector_lock:
        md = merged_dictionary()
        corpus = []
        keys = []
        for k, v in md.items():
            corpus.append(f"{k} {v.get('definition','')} {' '.join(v.get('examples',[]))}")
            keys.append(k)
        try:
            _vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
            _matrix = _vectorizer.fit_transform(corpus)
            _indexed_keys = keys
            ai_state["model_dirty"] = False
            save_state()
        except Exception as e:
            print("Failed to build semantic index:", e)

def find_similar_terms(query: str, topn: int = 6):
    global _vectorizer, _matrix, _indexed_keys
    if TfidfVectorizer is None:
        return []
    with _vector_lock:
        if _vectorizer is None or _matrix is None:
            rebuild_semantic_index()
        try:
            vec = _vectorizer.transform([query])
            sims = cosine_similarity(vec, _matrix)[0]
            pairs = list(enumerate(sims))
            pairs.sort(key=lambda x: x[1], reverse=True)
            top_idx = [i for i,_ in pairs[:topn]]
            return [( _indexed_keys[i], float(sims[i]) ) for i in top_idx if sims[i] > 0.08]
        except Exception:
            return []

def semantic_answer(query: str) -> Optional[str]:
    results = find_similar_terms(query)
    if not results:
        return None
    best_key, score = results[0]
    defs = merged_dictionary()
    if best_key in defs:
        entry = defs[best_key]
        ex = entry.get("examples", [])
        ex_text = (" Examples: " + " | ".join(ex)) if ex else ""
        return f"{best_key.capitalize()} ({entry.get('type','')}): {entry.get('definition','')}{ex_text} (score {score:.2f})"
    return None

# -------------------------
# Utilities (learn/patterns)
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
# Persona
# -------------------------
def apply_persona(text: str, persona: str) -> str:
    persona = (persona or "neutral").lower()
    if persona == "cowboy":
        if not text.endswith((".", "!", "?")):
            text = text + "."
        return f"Howdy — {text} Y'all take care now."
    if persona == "pirate":
        return f"Avast! {text} Arr!"
    if persona == "scientist":
        return f"As a scientist, here's a concise view: {text}"
    if persona == "formal":
        return f"{text} Please let me know if you require further clarification."
    if persona == "casual":
        return f"{text} — cool?"
    return text

def respond(reply_text: str, meta: Dict[str,Any]) -> Dict[str,Any]:
    persona = ai_state.get("settings", {}).get("persona", "neutral")
    reply_text = apply_persona(reply_text, persona)
    return {"reply": reply_text, "meta": meta}

# -------------------------
# Massive KB generator (in-memory only if EPHEMERAL_MODE True)
# -------------------------
def build_massive_kb(save_to_file: bool = False,
                     people_count: int = 1200,
                     foods_count: int = 600,
                     research_count: int = 600,
                     recipes_per_food: int = 1,
                     min_total: int = 5000) -> dict:
    # identical generator as before but note: we will save only if not ephemeral or if save_to_file True and not ephemeral
    kb = {}
    # small seeds and then bulk generation (kept concise here)
    kb.update({
        "what is pi":"Pi (π) ≈ 3.14159.",
        "who wrote hamlet":"William Shakespeare."
    })
    # generate people, foods, research, recipes as prior code...
    # (for brevity in this script body I'm using the same algorithm as earlier; the generator is unchanged)
    # --- people
    first_names = ["alex","sam","morgan","jordan","taylor","casey","ashley","jamie","chris","pat","drew","blake","riley","cameron","logan","maria","ana","linda","mike","david","sarah","jose","juan","li","wei","mina","rahul","ola","kofi","fatima"]
    last_names = ["smith","johnson","williams","brown","jones","garcia","miller","davis","martinez","hernandez","lopez","gonzalez","wilson","anderson","thomas","taylor","moore","jackson","martin","lee","perez"]
    professions = ["researcher","engineer","teacher","doctor","nurse","artist","musician","writer","journalist","scientist","chef","designer","developer","entrepreneur","farmer","pilot","lawyer","architect","pharmacist","historian"]
    people = []; seen=set()
    for a,b in [(a,b) for a in first_names for b in last_names]:
        if len(people) >= people_count: break
        name = f"{a.capitalize()} {b.capitalize()}"
        if name in seen: continue
        seen.add(name)
        prof = random.choice(professions)
        short = f"{name} is a {prof} known for work in {random.choice(['education','technology','healthcare','art','music','community service','research'])}."
        kb[f"who is {name.lower()}"] = short
        people.append(name)
    idx = 1
    while len(people) < people_count:
        fn = random.choice(first_names).capitalize()
        ln = f"Person{idx}"
        name = f"{fn} {ln}"
        if name in seen:
            idx += 1; continue
        seen.add(name)
        prof = random.choice(professions)
        short = f"{name} is a {prof} with experience in {random.choice(['policy','data','design','research','education','business'])}."
        kb[f"who is {name.lower()}"] = short
        people.append(name); idx += 1
    # --- foods + recipes
    base_foods = ["apple","banana","orange","pear","grape","mango","pineapple","strawberry","blueberry","kiwi","tomato","potato","carrot","lettuce","spinach","onion","garlic","pepper","cucumber","broccoli","rice","pasta","pizza","sandwich","soup","salad","chicken","beef","pork","lamb","salmon","tuna","shrimp","tofu","yogurt","butter","milk","coffee","tea","juice"]
    adjectives = ["roasted","grilled","fresh","smoked","spiced","creamy","crispy","candied","marinated","pickled","herbed","zesty","sweet","savory"]
    foods=[]
    while len(foods) < foods_count:
        stem=random.choice(base_foods)
        adj=random.choice(adjectives) if random.random()<0.6 else ""
        name=(adj+" "+stem).strip() if adj else stem
        key=name.replace(" ","_").lower()
        if key not in foods:
            foods.append(key); kb[f"what is {name}"]=f"{name.capitalize()} is a food item commonly used in many cuisines."
    cnt=1
    while len(foods) < foods_count:
        fkey=f"food_item_{cnt}"
        if fkey not in foods:
            foods.append(fkey); kb[f"what is {fkey.replace('_',' ')}"]=f"{fkey.replace('_',' ').capitalize()} is a synthetic food item."
        cnt+=1
    actions=["preheat the oven to 180°C","slice the ingredients","mix well","simmer for 10 minutes","boil until tender","fry until golden","bake until done","serve warm","season with salt and pepper","garnish with herbs"]
    for f in foods:
        dish=f.replace("_"," ")
        steps=random.choice([1,2,3])
        recipe_text=" ".join([f"{i+1}. {random.choice(actions).capitalize()}." for i in range(steps)])
        kb[f"how to cook {dish}"]=recipe_text; kb[f"recipe for {dish}"]=recipe_text
    # --- research topics
    research_fields = ["machine learning","biotechnology","quantum computing","renewable energy","material science","neuroscience","robotics","climate change","astrobiology","public health","economics","anthropology","sociology","linguistics","cybersecurity"]
    research_topics=[]
    while len(research_topics) < research_count:
        field=random.choice(research_fields)
        modifier=random.choice(["applications","methods","ethics","optimization","benchmarks","robustness","interpretability","privacy","scalability"])
        topic=f"{field} {modifier}"
        if topic not in research_topics:
            research_topics.append(topic); kb[f"what is research on {topic}"]=f"Research on {topic} explores methods and applications in {field}."
    ridx=1
    while len(research_topics) < research_count:
        t=f"research topic {ridx}"
        research_topics.append(t); kb[f"what is research on {t}"]=f"Research on {t} investigates related questions and methods."
        ridx+=1
    # enough filler to reach min_total
    j=1
    while len(kb) < min_total:
        q=f"what is placeholder_topic_{j}"; kb[q]=f"Placeholder topic {j} for KB expansion."; j+=1
    # save to 'disk' only if not ephemeral and if caller asked to save
    if not EPHEMERAL_MODE and save_to_file:
        try:
            save_json(KB_FILE, kb)
        except Exception:
            pass
    elif EPHEMERAL_MODE:
        # store in-session for immediate use if ephemeral
        save_json(KB_FILE, kb)
    return kb

# -------------------------
# Geography & planets (in-memory or saved only when permitted)
# -------------------------
def enrich_kb_with_geography(kb: dict, save_local_copy: bool=False) -> dict:
    # embedded minimal but wide coverage; same as prior code
    embedded = [
        {"country":"France","capital":"Paris","continent":"Europe"},
        {"country":"Germany","capital":"Berlin","continent":"Europe"},
        {"country":"Spain","capital":"Madrid","continent":"Europe"},
        {"country":"Italy","capital":"Rome","continent":"Europe"},
        {"country":"United States","capital":"Washington, D.C.","continent":"North America"},
        {"country":"Canada","capital":"Ottawa","continent":"North America"},
        {"country":"China","capital":"Beijing","continent":"Asia"},
        {"country":"India","capital":"New Delhi","continent":"Asia"},
        {"country":"Brazil","capital":"Brasília","continent":"South America"},
        {"country":"Australia","capital":"Canberra","continent":"Oceania"},
        # (more can be generated or added)
    ]
    for item in embedded:
        country = str(item.get("country","")).strip()
        if not country: continue
        cap = str(item.get("capital","")).strip() or "N/A"
        cont = str(item.get("continent","")).strip() or "Unknown"
        kb[f"capital of {country.lower()}"] = f"{cap} (continent: {cont})"
        kb[f"what continent is {country.lower()}"] = cont
    planets = {
        "mercury":"Mercury is the smallest planet and closest to the Sun; it has extreme temperature variations.",
        "venus":"Venus is similar in size to Earth but has a thick, toxic atmosphere and very high surface temperatures.",
        "earth":"Earth is the third planet from the Sun and the only known planet to support life.",
        "mars":"Mars is the red planet with the largest volcano (Olympus Mons) and evidence of past water.",
        "jupiter":"Jupiter is the largest planet, a gas giant with a prominent storm called the Great Red Spot.",
        "saturn":"Saturn is a gas giant known for its extensive rings.",
        "uranus":"Uranus is an ice giant with a tilted rotation axis.",
        "neptune":"Neptune is an ice giant known for strong winds and storms.",
        "pluto":"Pluto is a dwarf planet in the Kuiper Belt."
    }
    for p,desc in planets.items():
        kb[f"what is {p}"] = desc
        kb[f"is {p} a planet"] = "Yes." if p != "pluto" else "Pluto is a dwarf planet."
    # save if non-ephemeral and requested
    if not EPHEMERAL_MODE and save_local_copy:
        try:
            save_json(COUNTRIES_FILE, {"countries": embedded})
        except Exception:
            pass
    elif EPHEMERAL_MODE:
        save_json(COUNTRIES_FILE, {"countries": embedded})
    return kb

# -------------------------
# Compose reply
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

def compose_reply(user_text: str, topic: Optional[str]=None, paragraph_sentences: Optional[int]=None) -> Dict[str,Any]:
    user = user_text.strip()
    lower = user.lower()
    if lower in ("/clear", "clear chat"):
        ai_state["conversations"].clear(); save_state(); return respond("Chat cleared.", {"intent":"memory"})
    if lower in ("/forget", "forget"):
        ai_state["learned"].clear(); save_state(); ai_state["model_dirty"]=True; save_state()
        threading.Thread(target=rebuild_semantic_index).start()
        return respond("Learned memory cleared.", {"intent":"memory"})
    if lower.startswith("/delete "):
        arg = lower[len("/delete "):].strip()
        if arg.isdigit():
            idx = int(arg)-1
            if 0 <= idx < len(ai_state.get("conversations", [])):
                removed = ai_state["conversations"].pop(idx)
                save_state()
                return respond(f"Deleted conversation #{idx+1}: {removed.get('text')}", {"intent":"memory"})
            else:
                return respond("Invalid conversation index.", {"intent":"error"})
        else:
            key = normalize_key(arg)
            if key in ai_state.get("learned", {}):
                ai_state["learned"].pop(key); save_state(); ai_state["model_dirty"]=True; save_state()
                threading.Thread(target=rebuild_semantic_index).start()
                return respond(f"Removed learned definition for '{key}'.", {"intent":"memory"})
            else:
                return respond(f"No learned definition for '{key}'.", {"intent":"error"})
    if lower.startswith("/persona ") or lower.startswith("persona "):
        parts = user.split(None,1)
        if len(parts) > 1:
            p = parts[1].strip().lower()
            ai_state.setdefault("settings", {})["persona"] = p
            save_state()
            return respond(f"Persona set to '{p}'.", {"intent":"persona"})
        else:
            return respond(f"Current persona: {ai_state.get('settings',{}).get('persona','neutral')}", {"intent":"persona"})
    math_res = safe_eval_math(user)
    if math_res is not None:
        return respond(f"Math result: {math_res}", {"intent":"math"})
    if re.search(r"\bwhat(?:'s| is)? the time\b|\btime now\b|\bcurrent time\b", lower):
        return respond(f"The current time is {datetime.now().strftime('%H:%M:%S')}", {"intent":"time"})
    if re.search(r"\bwhat(?:'s| is)? the date\b|\bcurrent date\b|\bdate today\b", lower):
        return respond(f"Today's date is {datetime.now().strftime('%Y-%m-%d')}", {"intent":"date"})
    if lower.startswith("/define ") or lower.startswith("define "):
        rest = user.split(None,1)[1] if len(user.split(None,1))>1 else ""
        m = re.match(r'\s*([^\:]+)\s*[:\-]\s*(.+)', rest)
        if m:
            w = normalize_key(m.group(1)); d = m.group(2).strip()
            ai_state.setdefault("learned", {})[w] = {"definition": d, "type":"learned", "examples": []}
            save_state()
            ai_state["model_dirty"] = True; save_state()
            threading.Thread(target=rebuild_semantic_index).start()
            return respond(f"Learned definition for '{w}'.", {"intent":"learning"})
        m2 = re.match(r'\s*([A-Za-z\'\- ]+)\s*$', rest)
        if m2:
            key = normalize_key(m2.group(1))
            defs = merged_dictionary()
            if key in defs:
                return respond(format_definition(key, defs[key]), {"intent":"definition"})
            else:
                return respond(f"No definition for '{key}'. Use '/define {key}: <meaning>' to teach me.", {"intent":"definition"})
        return respond("Usage: /define word: definition", {"intent":"define"})
    w,d = try_extract_definition(user)
    if w and d:
        ai_state.setdefault("learned", {})[w] = {"definition": d, "type":"learned", "examples": []}
        save_state()
        ai_state["model_dirty"] = True; save_state()
        threading.Thread(target=rebuild_semantic_index).start()
        return respond(f"Saved learned definition: '{w}' = {d}", {"intent":"learning"})
    ans, conf = lookup_kb(user)
    if ans:
        return respond(str(ans), {"intent":"fact","confidence":conf})
    mem = retrieve_from_memory_or_learned(user)
    if mem:
        return respond(mem, {"intent":"memory"})
    sem = semantic_answer(user)
    if sem:
        return respond(sem, {"intent":"semantic"})
    if paragraph_sentences and paragraph_sentences > 0:
        para = MARKOV.generate_paragraph(seed=(user if user else None), topic=topic, num_sentences=paragraph_sentences)
        if para:
            return respond(para, {"intent":"gen_paragraph"})
    gen = MARKOV.generate(seed=user)
    if gen:
        if re.search(r"[\.!\?]\s*$", user):
            reply_text = gen
        else:
            reply_text = (user.rstrip() + " " + gen).strip()
        return respond(reply_text, {"intent":"gen"})
    return respond("I don't know that yet. Teach me with 'X means Y' or '/define X: Y'.", {"intent":"unknown"})

# -------------------------
# UI & Controls (ephemeral friendly)
# -------------------------
st.set_page_config(page_title="Omega-B (ephemeral)", layout="wide")
st.title("Omega-B (ephemeral mode)" if EPHEMERAL_MODE else "Omega-B")

left, right = st.columns([3,1])

with right:
    st.header("Quick Controls")
    st.markdown("Ephemeral mode is " + ("ON — heavy assets are in-memory and remade on reset." if EPHEMERAL_MODE else "OFF — assets saved to disk."))

    if st.button("Reset (clear all runtime assets)"):
        # Clear ai_state entirely (ephemeral)
        ai_state.clear()
        ai_state.update({"conversations": [], "learned": {}, "settings": {"persona":"neutral"}, "model_dirty": True})
        # clear ephemeral store if in-memory
        if EPHEMERAL_MODE:
            st.session_state.pop("_ephemeral_store", None)
        else:
            # remove disk files
            remove_json(STATE_FILE); remove_json(DICT_FILE); remove_json(MARKOV_FILE); remove_json(KB_FILE); remove_json(COUNTRIES_FILE)
        # rebuild small runtime artifacts
        save_state()
        # reset markov object
        MARKOV.map.clear(); MARKOV.starts.clear()
        st.success("Reset complete (in-memory assets cleared).")
        st.experimental_rerun()

    st.markdown("---")
    if st.button("Generate big KB (in-memory)"):
        with st.spinner("Generating big KB..."):
            big = build_massive_kb(save_to_file=False, people_count=800, foods_count=400, research_count=400, min_total=2500)
            # store in memory (ephemeral) or save to disk if allowed
            save_json(KB_FILE, big)
            KB.update(big)
            enrich_kb_with_geography(KB, save_local_copy=False)
            ai_state["model_dirty"] = True; save_state()
            sampled_markov_train(limit_examples=1500)
        st.success("Big KB generated and loaded (in-memory).")

    st.markdown("---")
    st.write("Model status:")
    if ai_state.get("model_dirty", False):
        st.warning("Model DIRTY — rebuild recommended.")
    else:
        st.success("Model up-to-date.")
    if st.button("Rebuild Model (fast)"):
        with st.spinner("Rebuilding model..."):
            build_and_train_model(force=True)
        st.success("Model rebuilt.")
    if st.button("Rebuild Semantic TF-IDF"):
        with st.spinner("Rebuilding TF-IDF..."):
            rebuild_semantic_index(force=True)
        st.success("Semantic rebuilt.")

    st.markdown("---")
    if st.button("Clear learned memories"):
        ai_state["learned"].clear(); save_state(); st.success("Learned cleared.")

with left:
    st.subheader("Conversation")
    history = ai_state.get("conversations", [])[-200:]
    for m in history:
        who = "You" if m.get("role","user")=="user" else "Omega"
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
            save_state()
            MARKOV.train(ui); MARKOV.train(reply)
            ai_state["model_dirty"] = True; save_state()
            threading.Thread(target=rebuild_semantic_index).start()
            st.experimental_rerun()

    if c2.button("Generate Paragraph"):
        ui = user_input.strip()
        para = MARKOV.generate_paragraph(seed=(ui if ui else None), topic=(topic_input or None), num_sentences=num_sentences)
        if para:
            if ui:
                ai_state.setdefault("conversations", []).append({"role":"user","text":ui,"time":datetime.now().isoformat()})
            ai_state.setdefault("conversations", []).append({"role":"assistant","text":para,"time":datetime.now().isoformat()})
            save_state()
            MARKOV.train(para)
            ai_state["model_dirty"] = True; save_state()
            threading.Thread(target=rebuild_semantic_index).start()
            st.experimental_rerun()

    if c3.button("Teach (word: definition)"):
        ui = user_input.strip()
        m = re.match(r'\s*([^\:]+)\s*[:\-]\s*(.+)', ui)
        if m:
            w = normalize_key(m.group(1)); d = m.group(2).strip()
            ai_state.setdefault("learned", {})[w] = {"definition": d, "type":"learned", "examples": []}
            save_state()
            ai_state["model_dirty"] = True; save_state()
            threading.Thread(target=rebuild_semantic_index).start()
            st.success(f"Learned '{w}'.")
            st.experimental_rerun()
        else:
            st.warning("To teach: enter `word: definition` (e.g. gravity: a force that pulls)")

st.markdown("---")
st.markdown("**Examples / Commands**")
st.markdown("""
- Ask a fact: `Who was the first president of the U.S.?`  
- Teach: `gravity means a force that pulls` or `/define gravity: a force that pulls`  
- Persona: `/persona cowboy` or `/persona scientist`  
- Paragraph: type a seed (optional) and a Topic, then click **Generate Paragraph**  
- Commands: `/clear` (clear conversation), `/forget` (clear learned memories), `/delete N` (delete convo #N)
""")
st.caption(f"Device session id: {DEVICE_ID} (ephemeral mode = {EPHEMERAL_MODE}).")
