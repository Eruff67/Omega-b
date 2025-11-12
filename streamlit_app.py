# ==============================================================
# Jack.AI — Super Offline Edition
# Personal Property (© You)
# Single-file, offline Streamlit app
# ==============================================================


# IMPORT GITHUB REPO


# END OF GIRHUB IMPORTS

# RETRIEVE SKLEARN MODULE



# END OF SKLEARN RETRIEVAL


# jack_ai_super.py
"""
Jack.AI — Single-file Offline Smart Assistant (Personal property)
Features:
 - Streamlit UI (Chat / Memory / Train / Links / Debug)
 - TF-IDF retrieval + RAG-style synth (sklearn if available)
 - Simple, easy training (save examples / add from chat)
 - Safe Python math evaluator (supports math.* functions)
 - Link analysis: fetch page, extract text, summarize, save as memory
 - All data persisted under ./jack_data (no .nginx, no extra files)
Dependencies (recommended):
  pip install streamlit requests beautifulsoup4 scikit-learn numpy
"""

from __future__ import annotations
import os, json, time, re, math, random, threading
from datetime import datetime
from typing import List, Dict, Any, Optional
from io import BytesIO

import streamlit as st

# Optional / recommended libs (graceful fallback)
SKLEARN_AVAILABLE = False
REQUESTS_AVAILABLE = False
BS4_AVAILABLE = False
NUMPY_AVAILABLE = False
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except Exception:
    NUMPY_AVAILABLE = False
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.neural_network import MLPRegressor
    from sklearn.linear_model import LinearRegression
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False
try:
    import requests
    REQUESTS_AVAILABLE = True
except Exception:
    REQUESTS_AVAILABLE = False
try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except Exception:
    BS4_AVAILABLE = False

# -------------------------
# Storage
# -------------------------
DATA_DIR = os.path.join(os.getcwd(), "jack_data")
os.makedirs(DATA_DIR, exist_ok=True)

FILES = {
    "chat": "chat.json",
    "mem": "memories.json",
    "examples": "examples.json",
    "tiny": "tiny_model.json",
    "persona": "persona.json",
    "debug": "debug.log"
}
def path_for(k): return os.path.join(DATA_DIR, FILES[k])

def load_json(k, fallback):
    p = path_for(k)
    if os.path.exists(p):
        try:
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return fallback
    return fallback

def save_json(k, obj):
    p = path_for(k)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

# -------------------------
# Session state init
# -------------------------
if "chat" not in st.session_state:
    st.session_state.chat = load_json("chat", [])
if "memories" not in st.session_state:
    st.session_state.memories = load_json("mem", [])
if "examples" not in st.session_state:
    st.session_state.examples = load_json("examples", [])
if "tiny_model" not in st.session_state:
    st.session_state.tiny_model = load_json("tiny", {"trained": False})
if "persona" not in st.session_state:
    st.session_state.persona = load_json("persona", {"name":"Jack", "system_prompt":"You are Jack, a helpful assistant.", "politeness":1.0})
if "debug_log" not in st.session_state:
    st.session_state.debug_log = ""
if "runtime" not in st.session_state:
    st.session_state.runtime = {"tfidf": None, "reg": None, "markov": None}

# -------------------------
# Utilities
# -------------------------
def now_iso(): return datetime.utcnow().isoformat() + "Z"
def now_human(): return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
def debug(msg: str):
    st.session_state.debug_log = f"{now_human()} - {msg}\n" + st.session_state.debug_log
    try:
        save_json("debug", st.session_state.debug_log)
    except Exception:
        pass

def persist_all():
    save_json("chat", st.session_state.chat)
    save_json("mem", st.session_state.memories)
    save_json("examples", st.session_state.examples)
    save_json("tiny", st.session_state.tiny_model)
    save_json("persona", st.session_state.persona)
    debug("State persisted")

# -------------------------
# Tokenizer & TF-IDF fallback
# -------------------------
token_re = re.compile(r"[a-z0-9']+")

def tokenize(text: str) -> List[str]:
    if not text: return []
    return token_re.findall(text.lower())

def build_vocab(docs: List[str]) -> Dict[str,int]:
    v = {}
    idx = 0
    for d in docs:
        for t in set(tokenize(d)):
            if t not in v:
                v[t] = idx; idx += 1
    return v

def compute_idf(docs: List[str], vocab: Dict[str,int]) -> List[float]:
    N = max(1, len(docs))
    df = [0]*len(vocab)
    for d in docs:
        seen = set()
        for t in set(tokenize(d)):
            if t in vocab and t not in seen:
                df[vocab[t]] += 1
                seen.add(t)
    return [math.log((1+N)/(1+x))+1 for x in df]

def to_tfidf_vec(text: str, vocab: Dict[str,int], idf: List[float]) -> List[float]:
    vec = [0.0]*len(vocab)
    toks = tokenize(text)
    L = len(toks)
    if L == 0: return vec
    tf = {}
    for t in toks:
        if t in vocab:
            tf[vocab[t]] = tf.get(vocab[t], 0) + 1
    for k,v in tf.items():
        vec[k] = (v / L) * (idf[k] if k < len(idf) else 1.0)
    norm = math.sqrt(sum(x*x for x in vec)) or 1.0
    return [x / norm for x in vec]

def cosine(a: List[float], b: List[float]) -> float:
    if not a or not b: return 0.0
    num = sum((ai or 0.0)*(bi or 0.0) for ai,bi in zip(a,b))
    na = math.sqrt(sum((ai or 0.0)*(ai or 0.0) for ai in a)) or 1.0
    nb = math.sqrt(sum((bi or 0.0)*(bi or 0.0) for bi in b)) or 1.0
    return num / (na*nb)

# -------------------------
# Build vector store
# -------------------------
def build_vector_store():
    docs = []
    meta = []
    for m in st.session_state.memories:
        text = m.get("value") or m.get("text") or ""
        if text:
            docs.append(text); meta.append({"type":"memory","id":m.get("key", m.get("time"))})
    for ex in st.session_state.examples:
        if ex.get("user"):
            docs.append(ex.get("user")); meta.append({"type":"example_user"})
        if ex.get("assistant"):
            docs.append(ex.get("assistant")); meta.append({"type":"example_assistant"})
    # include recent chat
    for c in st.session_state.chat[-60:]:
        docs.append(c.get("content","")); meta.append({"type":c.get("role","")})
    if not docs:
        return {"vocab":{}, "idf":[], "embeddings":[], "docs":[], "meta":[]}
    vocab = build_vocab(docs)
    idf = compute_idf(docs, vocab)
    embeddings = [to_tfidf_vec(d, vocab, idf) for d in docs]
    return {"vocab":vocab, "idf":idf, "embeddings":embeddings, "docs":docs, "meta":meta}

# -------------------------
# Retrieval
# -------------------------
def retrieve(query: str, top_k: int = 4):
    store = build_vector_store()
    if not store["docs"]:
        return []
    qv = to_tfidf_vec(query, store["vocab"], store["idf"])
    sims = [(i, cosine(qv, vec)) for i,vec in enumerate(store["embeddings"])]
    sims.sort(key=lambda x: x[1], reverse=True)
    top = []
    for i,score in sims[:top_k]:
        top.append({"doc": store["docs"][i], "score": score, "meta": store["meta"][i]})
    return top

# -------------------------
# Simple summarizer
# -------------------------
def simple_summarize(text: str, max_sentences: int = 3) -> str:
    sents = re.split(r'(?<=[.!?])\s+', (text or "").strip())
    if not sents: return ""
    if len(sents) <= max_sentences:
        return " ".join(sents)
    vocab = build_vocab(sents)
    if not vocab:
        return " ".join(sents[:max_sentences])
    idf = compute_idf(sents, vocab)
    scored = []
    for s in sents:
        v = to_tfidf_vec(s, vocab, idf)
        scored.append((sum(v), s))
    scored.sort(reverse=True, key=lambda x: x[0])
    top = [s for _,s in scored[:max_sentences]]
    return " ".join(top)

# -------------------------
# Markov chain fallback (n-gram)
# -------------------------
def build_markov(n=2):
    corpus = []
    for e in st.session_state.examples:
        if e.get("assistant"): corpus.append(e.get("assistant"))
    for m in st.session_state.memories:
        corpus.append(m.get("value",""))
    for c in st.session_state.chat:
        if c.get("role")=="assistant": corpus.append(c.get("content",""))
    text = "\n".join([t for t in corpus if t])
    toks = tokenize(text)
    if not toks:
        return {}
    model = {}
    for i in range(len(toks)-n):
        key = tuple(toks[i:i+n])
        nxt = toks[i+n]
        model.setdefault(key, []).append(nxt)
    return model

def markov_generate(model, length=30):
    if not model:
        return ""
    key = random.choice(list(model.keys()))
    out = list(key)
    for _ in range(length):
        choices = model.get(tuple(out[-len(key):]), None)
        if not choices: break
        out.append(random.choice(choices))
    return " ".join(out)

# -------------------------
# Training tiny model (sklearn if available)
# -------------------------
def train_tiny_model():
    """
    Trains a mapping from user prompt -> assistant TF-IDF vector.
    Stores a tiny_model flag and feature count in st.session_state.tiny_model.
    Also stores runtime objects (tfidf vectorizer & regressor) in session_state.runtime for immediate use.
    """
    exs = [e for e in st.session_state.examples if e.get("user") and e.get("assistant")]
    if not exs:
        return {"trained": False, "reason": "no examples"}
    if not SKLEARN_AVAILABLE:
        # fallback: record that training isn't done
        st.session_state.tiny_model = {"trained": False, "reason": "sklearn not installed"}
        return st.session_state.tiny_model
    texts = [e["user"] for e in exs]
    responses = [e["assistant"] for e in exs]
    vec = TfidfVectorizer(max_features=2000)
    X = vec.fit_transform(texts)
    Y = vec.transform(responses)
    # Try MLP multi-output regressor, fallback to LinearRegression
    try:
        reg = MLPRegressor(hidden_layer_sizes=(256,), max_iter=400)
        reg.fit(X.toarray(), Y.toarray())
    except Exception:
        reg = LinearRegression()
        reg.fit(X.toarray(), Y.toarray())
    st.session_state.runtime["tfidf"] = vec
    st.session_state.runtime["reg"] = reg
    st.session_state.tiny_model = {"trained": True, "examples": len(exs), "features": X.shape[1]}
    persist_all()
    debug(f"Tiny model trained: examples={len(exs)}, features={X.shape[1]}")
    return st.session_state.tiny_model

def tiny_predict_direct(user_text: str) -> Optional[str]:
    if not st.session_state.tiny_model.get("trained"):
        return None
    vec = st.session_state.runtime.get("tfidf")
    reg = st.session_state.runtime.get("reg")
    if not vec or not reg:
        return None
    Xq = vec.transform([user_text]).toarray()
    Ypred = reg.predict(Xq)  # shape (1, n_features)
    # pick top words from vectorizer vocabulary to synthesize response tokens
    inv_vocab = {v:k for k,v in vec.vocabulary_.items()}
    top_idx = list(np.argsort(-np.abs(Ypred[0]))[:30]) if NUMPY_AVAILABLE else []
    words = [inv_vocab.get(int(i),"") for i in top_idx if inv_vocab.get(int(i),"")]
    if words:
        return " ".join(words[:60])
    return None

# -------------------------
# RAG-style synthesizer
# -------------------------
def heuristic_answer(user_text: str, contexts: List[str]) -> str:
    qtokens = set(tokenize(user_text))
    scored = []
    for p in contexts:
        overlap = len(qtokens.intersection(set(tokenize(p))))
        scored.append((overlap, p))
    scored.sort(reverse=True)
    top = scored[0][1] if scored else (contexts[0] if contexts else "")
    if "how" in user_text.lower() or "what" in user_text.lower():
        return f"I found this relevant: \"{top[:300]}\" — a practical next step is to break the task down and test each step."
    if "why" in user_text.lower():
        return f"This passage suggests: \"{top[:300]}\" — likely cause: configuration or missing dependency."
    return f"{top[:300]} ... In summary: try isolating the variables and testing assumptions."

def synthesize_reply(user_text: str) -> str:
    # 1) Retrieval
    retrieved = retrieve(user_text, top_k=6)
    contexts = [r["doc"] for r in retrieved if r["score"] > 0.03]
    # 2) Tiny predictor
    tiny = None
    try:
        tiny = tiny_predict_direct(user_text) if SKLEARN_AVAILABLE else None
    except Exception as e:
        debug(f"tiny predict error: {e}")
        tiny = None
    # Compose best answer
    persona_name = st.session_state.persona.get("name","Jack")
    if contexts and len(contexts) > 0:
        summary = simple_summarize(" ".join(contexts), 3)
        answer = f"{persona_name}: Based on what I recall, {summary}\n\n{heuristic_answer(user_text, contexts)}"
        return answer
    if tiny:
        return f"{persona_name}: {tiny}"
    # markov fallback
    mc = build_markov()
    gen = markov_generate(mc, 40)
    if gen:
        return f"{persona_name}: {gen}"
    # final fallback
    return f"{persona_name}: I don't have enough info yet — try /learn or paste a link to teach me."

# -------------------------
# Safe python math evaluator
# -------------------------
import ast

ALLOWED_MATH_FUNCS = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
# allow 'abs', 'min', 'max', 'round' too
ALLOWED_GLOBALS = {"__builtins__": None}
ALLOWED_GLOBALS.update(ALLOWED_MATH_FUNCS)
ALLOWED_GLOBALS.update({"abs": abs, "min": min, "max": max, "round": round})

class SafeEval(ast.NodeVisitor):
    ALLOWED_NODES = (
        ast.Expression, ast.BinOp, ast.UnaryOp, ast.Num, ast.Load,
        ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod,
        ast.USub, ast.UAdd, ast.Call, ast.Name, ast.Tuple, ast.List,
        ast.Dict, ast.Constant, ast.FloorDiv
    )
    def __init__(self, expr: str):
        self.expr = expr
    def visit(self, node):
        if not isinstance(node, self.ALLOWED_NODES):
            raise ValueError(f"Disallowed expression: {type(node).__name__}")
        return super().visit(node)
    def eval(self):
        node = ast.parse(self.expr, mode='eval')
        self.visit(node)
        code = compile(node, "<safe>", "eval")
        return eval(code, ALLOWED_GLOBALS, {})

def safe_eval(expr: str):
    try:
        se = SafeEval(expr)
        return se.eval()
    except Exception as e:
        return f"Error evaluating expression: {e}"

# -------------------------
# Link analysis (fetch + parse) — will run only if requests+bs4 installed
# -------------------------
def analyze_link(url: str) -> Dict[str,Any]:
    if not (REQUESTS_AVAILABLE and BS4_AVAILABLE):
        return {"success": False, "error": "requests or bs4 not installed on this environment."}
    try:
        r = requests.get(url, timeout=12, headers={"User-Agent":"JackAI-Offline/1.0"})
        r.raise_for_status()
    except Exception as e:
        return {"success": False, "error": f"Fetch failed: {e}"}
    try:
        soup = BeautifulSoup(r.text, "html.parser")
        # remove script/style
        for tag in soup(["script","style","noscript","header","footer","nav","form","svg","img","meta","link"]):
            tag.decompose()
        title = soup.title.string.strip() if soup.title and soup.title.string else ""
        # gather paragraphs and headings
        pieces = []
        for el in soup.find_all(["h1","h2","h3","p","li"]):
            txt = el.get_text(separator=" ", strip=True)
            if txt:
                pieces.append(txt)
        fulltext = "\n\n".join(pieces)
        summary = simple_summarize(fulltext, max_sentences=5)
        # save as memory
        key = f"link_{now_iso()}"
        st.session_state.memories.append({"key": key, "value": f"Title: {title}\n\n{summary}", "source": url, "time": now_iso()})
        persist_all()
        return {"success": True, "title": title, "summary": summary, "key": key}
    except Exception as e:
        return {"success": False, "error": f"Parse failed: {e}"}

# -------------------------
# UI: Streamlit layout
# -------------------------
st.set_page_config(page_title="Jack.AI — Smart Offline", layout="wide")
st.markdown("<style>body{background:#071227;color:#e6eef8;} .stButton>button{background:#4acbff;color:#000}</style>", unsafe_allow_html=True)
st.title("Jack.AI — Smart Offline (single-file)")

# Sidebar controls
with st.sidebar:
    st.header("Controls")
    tab = st.radio("Tab", ["Chat","Memory","Train","Links","Settings","Debug"])
    st.markdown("---")
    st.write(f"Persona: **{st.session_state.persona.get('name','Jack')}**")
    if st.button("Save State"):
        persist_all(); st.success("Saved")
    if st.button("Wipe All Data"):
        if st.confirm("Wipe all local data? This will delete chat, memories, examples. Continue?"):
            # remove files
            for fn in FILES.values():
                p = os.path.join(DATA_DIR, fn)
                try:
                    if os.path.exists(p): os.remove(p)
                except Exception:
                    pass
            st.session_state.chat=[]; st.session_state.memories=[]; st.session_state.examples=[]
            st.success("Wiped")
    st.markdown("---")
    st.write("Environment:")
    st.write(f"- sklearn: {SKLEARN_AVAILABLE}")
    st.write(f"- numpy: {NUMPY_AVAILABLE}")
    st.write(f"- requests: {REQUESTS_AVAILABLE}")
    st.write(f"- bs4: {BS4_AVAILABLE}")

# -------------------------
# Chat tab
# -------------------------
if tab == "Chat":
    st.header("💬 Chat")
    col1, col2 = st.columns([3,1])
    with col1:
        # display last messages
        for msg in st.session_state.chat[-60:]:
            who = "You" if msg["role"]=="user" else st.session_state.persona.get("name","Jack")
            st.markdown(f"**{who}** — *{msg.get('time','')}*")
            st.write(msg.get("content",""))
        user_input = st.text_area("Message (or /command). Shift+Enter = newline", height=120)
        send = st.button("Send")
        if send and user_input and user_input.strip():
            text = user_input.strip()
            st.session_state.chat.append({"role":"user","content":text,"time":now_iso()})
            persist_all()
            # If it's a math expression special prefix: "/math <expr>"
            if text.startswith("/math "):
                expr = text[len("/math "):].strip()
                res = safe_eval(expr)
                reply = f"Math result: {res}"
                st.session_state.chat.append({"role":"assistant","content":reply,"time":now_iso()})
                persist_all()
                st.experimental_rerun()
            # slash commands
            elif text.startswith("/"):

