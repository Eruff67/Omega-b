# jack_ai_super.py
"""
Jack.AI — Offline Mini-Language Model Style (Single-file)
Personal property — no external configs, no .nginx, copy/paste ready.

Features:
- Streamlit UI (Chat / Memory / Train / Links / Debug)
- Offline English understanding using a hybrid:
    * TF-IDF retrieval for factual context (fast)
    * Markov-chain n-gram generator for fluent, creative replies (mini-LM style)
    * RAG-style synthesizer that composes answers from retrieved context + Markov
- Easy training: add examples, auto-generate examples from memories, one-button training
- Safe Python math evaluator (`/math <expr>`)
- Link analysis: fetch page with requests + BeautifulSoup if available (optional)
- Persistence under ./jack_data (created automatically)
- Works offline (no external API). If requests/bs4/sklearn installed, features are enhanced but optional.

Run:
  pip install streamlit requests beautifulsoup4 scikit-learn numpy
  streamlit run jack_ai_super.py

If you don't want to install extras, the app still runs with reduced features.
"""

from __future__ import annotations
import os, json, re, math, random, time
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from io import BytesIO

import streamlit as st

# Optional enhancements (graceful fallback)
HAS_REQUESTS = False
HAS_BS4 = False
HAS_SKLEARN = False
HAS_NUMPY = False
try:
    import requests; HAS_REQUESTS = True
except Exception:
    HAS_REQUESTS = False
try:
    from bs4 import BeautifulSoup; HAS_BS4 = True
except Exception:
    HAS_BS4 = False
try:
    import numpy as np; HAS_NUMPY = True
except Exception:
    HAS_NUMPY = False
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.neural_network import MLPRegressor
    from sklearn.linear_model import LinearRegression
    HAS_SKLEARN = True
except Exception:
    HAS_SKLEARN = False

# -------------------------
# Storage & persistence
# -------------------------
DATA_DIR = os.path.join(os.getcwd(), "jack_data")
os.makedirs(DATA_DIR, exist_ok=True)

FILES = {
    "chat": "chat.json",
    "mem": "memories.json",
    "examples": "examples.json",
    "persona": "persona.json",
    "tiny": "tiny_model.json",
    "debug": "debug.log"
}
def fp(k): return os.path.join(DATA_DIR, FILES[k])

def load(k, fallback):
    p = fp(k)
    try:
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return fallback

def save(k, v):
    try:
        with open(fp(k), "w", encoding="utf-8") as f:
            json.dump(v, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

# -------------------------
# Session state initialization
# -------------------------
if "chat" not in st.session_state:
    st.session_state.chat = load("chat", [])
if "memories" not in st.session_state:
    st.session_state.memories = load("mem", [])
if "examples" not in st.session_state:
    st.session_state.examples = load("examples", [])
if "persona" not in st.session_state:
    st.session_state.persona = load("persona", {"name": "Jack", "tone": "balanced"})
if "tiny" not in st.session_state:
    st.session_state.tiny = load("tiny", {"trained": False})
if "runtime" not in st.session_state:
    st.session_state.runtime = {"tfidf": None, "reg": None, "markov": None}
if "debug" not in st.session_state:
    st.session_state.debug = ""

def now_iso(): return datetime.utcnow().isoformat() + "Z"
def now_human(): return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
def log_debug(msg: str):
    st.session_state.debug = f"{now_human()} - {msg}\n" + st.session_state.debug
    try: save("debug", st.session_state.debug)
    except: pass

# -------------------------
# Tokenizer & TF-IDF fallback utilities
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
                df[vocab[t]] += 1; seen.add(t)
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
# Vector store builder
# -------------------------
def build_vector_store():
    docs = []
    meta = []
    for m in st.session_state.memories:
        t = m.get("value") or m.get("text") or ""
        if t: docs.append(t); meta.append({"type":"memory","time":m.get("time")})
    for ex in st.session_state.examples:
        if ex.get("user"): docs.append(ex.get("user")); meta.append({"type":"example_user"})
        if ex.get("assistant"): docs.append(ex.get("assistant")); meta.append({"type":"example_assistant"})
    for c in st.session_state.chat[-80:]:
        docs.append(c.get("content","")); meta.append({"type": c.get("role","")})
    if not docs:
        return {"vocab":{}, "idf":[], "embeddings":[], "docs":[], "meta":[]}
    vocab = build_vocab(docs)
    idf = compute_idf(docs, vocab)
    embeddings = [to_tfidf_vec(d, vocab, idf) for d in docs]
    return {"vocab":vocab, "idf":idf, "embeddings":embeddings, "docs":docs, "meta":meta}

# -------------------------
# Retrieval function
# -------------------------
def retrieve(query: str, top_k: int = 5):
    store = build_vector_store()
    if not store["docs"]:
        return []
    qv = to_tfidf_vec(query, store["vocab"], store["idf"])
    sims = [(i, cosine(qv, vec)) for i,vec in enumerate(store["embeddings"])]
    sims.sort(key=lambda x: x[1], reverse=True)
    out = []
    for i,score in sims[:top_k]:
        out.append({"doc": store["docs"][i], "score": score, "meta": store["meta"][i]})
    return out

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
# Markov chain mini-LM
# -------------------------
def build_markov_model(n: int = 2) -> Dict[Tuple[str,...], List[str]]:
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
    st.session_state.runtime["markov"] = model
    return model

def markov_generate(model: Dict[Tuple[str,...], List[str]], length: int = 40) -> str:
    if not model:
        return ""
    key = random.choice(list(model.keys()))
    out = list(key)
    for _ in range(length):
        choices = model.get(tuple(out[-len(key):]), None)
        if not choices: break
        out.append(random.choice(choices))
    return " ".join(out[:length])

# -------------------------
# Tiny predictor (optional sklearn)
# -------------------------
def train_tiny_model():
    exs = [e for e in st.session_state.examples if e.get("user") and e.get("assistant")]
    if not exs:
        st.session_state.tiny = {"trained": False, "reason": "no examples"}
        save("tiny", st.session_state.tiny)
        return st.session_state.tiny
    if not HAS_SKLEARN:
        st.session_state.tiny = {"trained": False, "reason": "sklearn missing"}
        save("tiny", st.session_state.tiny)
        return st.session_state.tiny
    texts = [e["user"] for e in exs]
    responses = [e["assistant"] for e in exs]
    vec = TfidfVectorizer(max_features=2000)
    X = vec.fit_transform(texts)
    Y = vec.transform(responses)
    try:
        reg = MLPRegressor(hidden_layer_sizes=(256,), max_iter=400)
        reg.fit(X.toarray(), Y.toarray())
    except Exception:
        reg = LinearRegression()
        reg.fit(X.toarray(), Y.toarray())
    st.session_state.runtime["tfidf"] = vec
    st.session_state.runtime["reg"] = reg
    st.session_state.tiny = {"trained": True, "examples": len(exs), "features": X.shape[1]}
    save("tiny", st.session_state.tiny)
    log_debug(f"Tiny trained: examples={len(exs)} features={X.shape[1]}")
    return st.session_state.tiny

def tiny_predict(user_text: str) -> Optional[str]:
    vec = st.session_state.runtime.get("tfidf")
    reg = st.session_state.runtime.get("reg")
    if not vec or not reg:
        return None
    Xq = vec.transform([user_text]).toarray()
    Ypred = reg.predict(Xq)
    inv = {v:k for k,v in vec.vocabulary_.items()}
    if HAS_NUMPY:
        idxs = list(np.argsort(-np.abs(Ypred[0]))[:30])
    else:
        idxs = sorted(range(len(Ypred[0])), key=lambda i: -abs(Ypred[0][i]))[:30]
    words = [inv.get(i,"") for i in idxs if inv.get(i,"")]
    return " ".join(words[:60]) if words else None

# -------------------------
# RAG-style synthesizer (hybrid)
# -------------------------
def heuristic_compose(user_text: str, contexts: List[str]) -> str:
    qtokens = set(tokenize(user_text))
    scored = []
    for p in contexts:
        overlap = len(qtokens.intersection(set(tokenize(p))))
        scored.append((overlap, p))
    scored.sort(reverse=True)
    top = scored[0][1] if scored else (contexts[0] if contexts else "")
    if any(w in user_text.lower() for w in ("how","what","step","instructions","guide")):
        return f"I found this relevant: \"{top[:300]}\". Practical next steps: break the task into small parts, test each step, and verify outputs."
    if "why" in user_text.lower():
        return f"This suggests: \"{top[:300]}\" — likely causes include missing configuration or incorrect assumptions."
    return f"{top[:320]} ... Summary: test assumptions, isolate variables, iterate."

def synthesize_reply(user_text: str) -> str:
    persona_name = st.session_state.persona.get("name","Jack")
    retrieved = retrieve(user_text, top_k=6)
    contexts = [r["doc"] for r in retrieved if r["score"] > 0.02]
    # try tiny model if trained
    tiny_text = None
    try:
        if st.session_state.tiny.get("trained"):
            tiny_text = tiny_predict(user_text)
    except Exception as e:
        log_debug(f"tiny predict error: {e}")
        tiny_text = None
    # Compose answer
    if contexts:
        summary = simple_summarize(" ".join(contexts), 3)
        heuristic = heuristic_compose(user_text, contexts)
        reply = f"{persona_name}: Based on what I recall, {summary}\n\n{heuristic}"
        return reply
    if tiny_text:
        return f"{persona_name}: {tiny_text}"
    # Markov fallback for more fluent English
    mc = st.session_state.runtime.get("markov") or build_markov_model()
    gen = markov_generate(mc, 50)
    if gen:
        return f"{persona_name}: {gen}"
    return f"{persona_name}: I don't have enough context yet. Teach me with /learn or paste a link in Links."

# helper to ensure markov stored
def build_markov_model():
    return build_markov_model if False else build_markov_model  # placeholder to keep function defined

# -------------------------
# Safe Python math evaluator
# -------------------------
import ast
ALLOWED_MATH = {k: getattr(math,k) for k in dir(math) if not k.startswith("_")}
ALLOWED_GLOBALS = {"__builtins__": None}
ALLOWED_GLOBALS.update(ALLOWED_MATH)
ALLOWED_GLOBALS.update({"abs":abs, "min":min, "max":max, "round":round})

class SafeEval(ast.NodeVisitor):
    ALLOWED_TYPES = (
        ast.Expression, ast.BinOp, ast.UnaryOp, ast.Num, ast.Load,
        ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod,
        ast.USub, ast.UAdd, ast.Call, ast.Name, ast.Tuple, ast.List,
        ast.Dict, ast.Constant, ast.FloorDiv
    )
    def __init__(self, expr: str):
        self.expr = expr
    def visit(self, node):
        if not isinstance(node, self.ALLOWED_TYPES):
            raise ValueError(f"Disallowed node: {type(node).__name__}")
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
        return f"Error: {e}"

# -------------------------
# Link analysis (optional)
# -------------------------
def analyze_link(url: str) -> Dict[str,Any]:
    if not (HAS_REQUESTS and HAS_BS4):
        return {"success": False, "error": "requests or bs4 not available"}
    try:
        r = requests.get(url, timeout=12, headers={"User-Agent":"JackAI-Offline/1.0"})
        r.raise_for_status()
    except Exception as e:
        return {"success": False, "error": f"Fetch failed: {e}"}
    try:
        soup = BeautifulSoup(r.text, "html.parser")
        for tag in soup(["script","style","noscript","header","footer","nav","form","svg","img","meta","link"]):
            tag.decompose()
        title = soup.title.string.strip() if soup.title and soup.title.string else ""
        pieces = []
        for el in soup.find_all(["h1","h2","h3","p","li"]):
            txt = el.get_text(separator=" ", strip=True)
            if txt and len(txt) > 20:
                pieces.append(txt)
        full = "\n\n".join(pieces)
        summary = simple_summarize(full, max_sentences=5)
        key = f"link_{now_iso()}"
        st.session_state.memories.append({"key": key, "value": f"Title: {title}\n\n{summary}", "source": url, "time": now_iso()})
        save("mem", st.session_state.memories)
        return {"success": True, "title": title, "summary": summary, "key": key}
    except Exception as e:
        return {"success": False, "error": f"Parse error: {e}"}

# -------------------------
# UI: Streamlit layout
# -------------------------
st.set_page_config(page_title="Jack.AI — Offline Mini-LM", layout="wide")
st.markdown("<style>body{background:#071227;color:#e6eef8} .stButton>button{background:#4acbff;color:#000}</style>", unsafe_allow_html=True)
st.title("Jack.AI — Offline Mini-LM (Markov + TF-IDF)")

with st.sidebar:
    st.header("Controls")
    tab = st.radio("Tab", ["Chat","Memory","Train","Links","Settings","Debug"])
    st.markdown("---")
    st.write(f"Persona: **{st.session_state.persona.get('name','Jack')}** (tone: {st.session_state.persona.get('tone','balanced')})")
    if st.button("Save state"):
        save("chat", st.session_state.chat); save("mem", st.session_state.memories); save("examples", st.session_state.examples); save("persona", st.session_state.persona); save("tiny", st.session_state.tiny)
        st.success("Saved")
    if st.button("Wipe all local data"):
        if st.confirm("Wipe all local data? This will delete saved chat, memories, and examples. Continue?"):
            for fn in FILES.values():
                try:
                    os.remove(os.path.join(DATA_DIR, fn))
                except Exception:
                    pass
            st.session_state.chat=[]; st.session_state.memories=[]; st.session_state.examples=[]
            st.success("Wiped local data")
    st.markdown("---")
    st.write("Environment:")
    st.write(f"- requests: {HAS_REQUESTS}")
    st.write(f"- bs4: {HAS_BS4}")
    st.write(f"- sklearn: {HAS_SKLEARN}")
    st.write(f"- numpy: {HAS_NUMPY}")

# Chat tab
if tab == "Chat":
    st.header("💬 Chat")
    col1, col2 = st.columns([3,1])
    with col1:
        for msg in st.session_state.chat[-80:]:
            who = "You" if msg["role"]=="user" else st.session_state.persona.get("name","Jack")
            st.markdown(f"**{who}** — *{msg.get('time','')}*")
            st.write(msg.get("content",""))
        user_text = st.text_area("Message (Shift+Enter newline). Commands: /learn, /persona <name>, /math <expr>, /train, /summarize", height=140)
        send = st.button("Send")
        if send and user_text and user_text.strip():
            txt = user_text.strip()
            st.session_state.chat.append({"role":"user","content":txt,"time":now_iso()})
            save("chat", st.session_state.chat)
            # math
            if txt.startswith("/math "):
                expr = txt[len("/math "):].strip()
                res = safe_eval(expr)
                reply = f"Math: {res}"
                st.session_state.chat.append({"role":"assistant","content":reply,"time":now_iso()})
                save("chat", st.session_state.chat); st.experimental_rerun()
            # learn
            elif txt.startswith("/learn "):
                mem = txt[len("/learn "):].strip()
                if mem:
                    st.session_state.memories.append({"key":f"mem_{now_iso()}","value":mem,"time":now_iso()})
                    save("mem", st.session_state.memories)
                    st.session_state.chat.append({"role":"assistant","content":"Learned and saved to memory.","time":now_iso()})
                    st.experimental_rerun()
            elif txt.startswith("/persona "):
                nm = txt[len("/persona "):].strip()
                st.session_state.persona["name"] = nm
                save("persona", st.session_state.persona)
                st.session_state.chat.append({"role":"assistant","content":f"Persona set to {nm}.","time":now_iso()})
                st.experimental_rerun()
            elif txt.startswith("/train"):
                with st.spinner("Training tiny predictor..."):
                    res = train_tiny_model()
                st.session_state.chat.append({"role":"assistant","content":f"Training result: {res}","time":now_iso()})
                st.experimental_rerun()
            elif txt.startswith("/summarize"):
                big = " ".join([m.get("value","") for m in st.session_state.memories])
                s = simple_summarize(big, 4)
                st.session_state.chat.append({"role":"assistant","content":f"Summary: {s}","time":now_iso()})
                save("chat", st.session_state.chat); st.experimental_rerun()
            else:
                # normal reply
                with st.spinner("Thinking..."):
                    # ensure markov built
                    st.session_state.runtime["markov"] = st.session_state.runtime.get("markov") or build_markov_model(2)
                    reply = synthesize_reply(txt)
                st.session_state.chat.append({"role":"assistant","content":reply,"time":now_iso()})
                save("chat", st.session_state.chat)
                st.experimental_rerun()
    with col2:
        st.subheader("Quick Tools")
        if st.button("Add last user->assistant as example"):
            found = False
            for i in range(len(st.session_state.chat)-1, 0, -1):
                if st.session_state.chat[i]["role"]=="assistant" and st.session_state.chat[i-1]["role"]=="user":
                    st.session_state.examples.append({"user": st.session_state.chat[i-1]["content"], "assistant": st.session_state.chat[i]["content"], "time": now_iso()})
                    save("examples", st.session_state.examples)
                    st.success("Example added"); found = True; break
            if not found: st.warning("No recent pair found")
        if st.button("Show retrieval for last user"):
            last = next((c for c in reversed(st.session_state.chat) if c["role"]=="user"), None)
            if last:
                st.json(retrieve(last["content"], top_k=6))
            else:
                st.info("No user message yet")
        if st.button("Regenerate last assistant"):
            last_assist_idx = None
            for i in range(len(st.session_state.chat)-1, -1, -1):
                if st.session_state.chat[i]["role"]=="assistant":
                    last_assist_idx = i; break
            if last_assist_idx is not None and last_assist_idx>0:
                user_msg = st.session_state.chat[last_assist_idx-1]["content"]
                reply = synthesize_reply(user_msg)
                st.session_state.chat[last_assist_idx]["content"] = reply
                save("chat", st.session_state.chat)
                st.experimental_rerun()

# Memory tab
elif tab == "Memory":
    st.header("🧠 Memories & Examples")
    with st.expander("Add memory manually"):
        mem_text = st.text_area("Memory text")
        if st.button("Save memory"):
            if mem_text.strip():
                st.session_state.memories.append({"key":f"mem_{now_iso()}","value":mem_text.strip(),"time":now_iso()})
                save("mem", st.session_state.memories)
                st.success("Saved memory")
    with st.expander("Add example"):
        u = st.text_input("User example")
        a = st.text_input("Assistant example")
        if st.button("Add example"):
            if u.strip() and a.strip():
                st.session_state.examples.append({"user":u.strip(),"assistant":a.strip(),"time":now_iso()})
                save("examples", st.session_state.examples)
                st.success("Example added")
    st.markdown("Recent memories:")
    for m in reversed(st.session_state.memories[-200:]):
        st.write(f"- {m.get('time')}: {m.get('value')[:400]}")
    st.markdown("Recent examples:")
    for e in reversed(st.session_state.examples[-200:]):
        st.write(f"- U: {e.get('user')[:200]} → A: {e.get('assistant')[:200]}")

# Train tab
elif tab == "Train":
    st.header("⚙️ Training")
    st.write("Examples:", len(st.session_state.examples))
    if HAS_SKLEARN:
        if st.button("Train tiny predictor now"):
            with st.spinner("Training..."):
                res = train_tiny_model()
            st.success("Training finished")
            st.json(res)
    else:
        st.warning("sklearn not available — install scikit-learn for stronger predictor.")
    if st.button("Auto-generate examples from memories"):
        created = 0
        for mem in st.session_state.memories[-200:]:
            txt = mem.get("value","")
            if len(txt.split()) > 8:
                first = re.split(r'[.!?]', txt)[0][:200]
                summ = simple_summarize(txt, 2)
                st.session_state.examples.append({"user": first, "assistant": summ, "time": now_iso()})
                created += 1
        save("examples", st.session_state.examples)
        st.success(f"Created {created} examples")

# Links tab
elif tab == "Links":
    st.header("🔗 Link analysis (optional)")
    st.write("Paste a URL and click Analyze. Requires requests and beautifulsoup4 installed.")
    url = st.text_input("URL (https://...)")
    if st.button("Analyze and learn"):
        if not url.strip():
            st.warning("Enter a URL")
        else:
            if not (HAS_REQUESTS and HAS_BS4):
                st.error("requests and bs4 required for link analysis. Install them to enable.")
            else:
                with st.spinner("Fetching and analyzing..."):
                    res = analyze_link(url.strip())
                if res.get("success"):
                    st.success("Link analyzed and saved to memories")
                    st.write("Title:", res.get("title"))
                    st.write("Summary:", res.get("summary"))
                else:
                    st.error("Failed: " + str(res.get("error")))
    st.markdown("Recent link memories:")
    for m in [mm for mm in st.session_state.memories if mm.get("source")][-20:][::-1]:
        st.write(f"- {m.get('time')}: {m.get('value')[:300]} (source: {m.get('source')})")

# Settings tab
elif tab == "Settings":
    st.header("⚙️ Settings")
    nm = st.text_input("Assistant name", st.session_state.persona.get("name","Jack"))
    tone = st.selectbox("Tone", ["balanced","concise","creative","sarcastic"], index=0)
    if st.button("Set persona"):
        st.session_state.persona["name"] = nm
        st.session_state.persona["tone"] = tone
        save("persona", st.session_state.persona)
        st.success("Persona set")
    st.write("Data directory:", DATA_DIR)
    st.write("Recommended packages for full features: requests, beautifulsoup4, scikit-learn, numpy")

# Debug tab
elif tab == "Debug":
    st.header("🐞 Debug")
    st.write("Debug log (recent):")
    st.code(st.session_state.debug[:4000])
    st.json({
        "chat_len": len(st.session_state.chat),
        "memories_len": len(st.session_state.memories),
        "examples_len": len(st.session_state.examples),
        "tiny": st.session_state.tiny,
        "HAS_REQUESTS": HAS_REQUESTS,
        "HAS_BS4": HAS_BS4,
        "HAS_SKLEARN": HAS_SKLEARN,
        "HAS_NUMPY": HAS_NUMPY
    })

# End of file
