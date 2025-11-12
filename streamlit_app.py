# jack_offline_ai.py
"""
Jack — Offline advanced generative AI (single file)
- Requires only Streamlit (pip install streamlit)
- Conversational replies, generative text (Markov-style + templates), math, time/date
- Learns definitions from conversation and /define commands
- Persistent storage: ai_state.json (conversations + learned_definitions)
- Supports merging an external dictionary.json (word -> {definition,type,examples})
"""

import streamlit as st
import json
import os
import re
import math
import random
from datetime import datetime
from typing import Dict, Any, List, Tuple

# ---------- Files ----------
MEMORY_FILE = "ai_state.json"
DICT_FILE = "dictionary.json"  # optional external file to drop/upload

# ---------- Load / Init ----------
def load_json(path: str, default):
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return default

ai_state = load_json(MEMORY_FILE, {"conversations": [], "learned": {}, "settings": {}})

# ---------- Seed dictionary (expanded) ----------
# This is a substantial sample of words — expand offline if desired by dropping dictionary.json
SEED_DICTIONARY: Dict[str, Dict[str, Any]] = {
    "apple": {"definition": "A round fruit with red or green skin and a whitish interior.", "type": "noun",
              "examples": ["I ate an apple for lunch.", "Apple pie is delicious."]},
    "run": {"definition": "To move swiftly on foot.", "type": "verb",
            "examples": ["I run every morning.", "She runs faster than me."]},
    "hello": {"definition": "A greeting.", "type": "interjection",
              "examples": ["Hello! How are you?", "She said hello and waved."]},
    "time": {"definition": "The ongoing progression of events; measured in seconds, minutes, hours.", "type": "noun",
             "examples": ["What time is it?", "Time flies when you're busy."]},
    "learn": {"definition": "To gain knowledge or skill through study or experience.", "type": "verb",
              "examples": ["I want to learn Python.", "She learned quickly."]},
    "gravity": {"definition": "The natural force that makes objects attract to one another.", "type": "noun",
                "examples": ["Gravity pulls objects toward Earth.", "The apple fell due to gravity."]},
    "computer": {"definition": "An electronic device for storing and processing data.", "type": "noun",
                 "examples": ["I bought a new computer.", "The computer is slow today."]},
    "python": {"definition": "A high-level programming language.", "type": "noun",
               "examples": ["Python is easy to learn.", "I wrote a script in Python."]},
    "book": {"definition": "A set of written or printed pages bound together.", "type": "noun",
             "examples": ["I read a book last night.", "The library has many books."]},
    "happy": {"definition": "Feeling or showing pleasure or contentment.", "type": "adjective",
              "examples": ["I am happy today.", "He felt happy after hearing the news."]},
    "sad": {"definition": "Feeling sorrow or unhappiness.", "type": "adjective",
            "examples": ["She was sad after the movie.", "It made me sad to hear the news."]},
    "music": {"definition": "Vocal or instrumental sounds combined to produce harmony.", "type": "noun",
              "examples": ["I listen to music every day.", "She loves classical music."]},
    "runny": {"definition": "Flowing or producing a liquid (often used for nose).", "type": "adjective",
              "examples": ["I have a runny nose.", "The sauce is runny."]},
    "beautiful": {"definition": "Pleasant to the senses or mind; aesthetically pleasing.", "type": "adjective",
                  "examples": ["The sunset is beautiful.", "She has a beautiful voice."]},
    "eat": {"definition": "To take food into the mouth, chew, and swallow.", "type": "verb",
            "examples": ["I eat breakfast every day.", "He eats quickly."]},
    "drink": {"definition": "To take liquids into the mouth and swallow.", "type": "verb",
              "examples": ["I drink water daily.", "She drank tea."]},
    "water": {"definition": "A clear, colorless, odorless liquid essential for life.", "type": "noun",
              "examples": ["I drink water every day.", "The water is cold."]},
    "swim": {"definition": "To propel oneself through water using limbs.", "type": "verb",
             "examples": ["We swim in the pool.", "Fish swim quickly."]},
    "angry": {"definition": "Feeling strong annoyance or displeasure.", "type": "adjective",
              "examples": ["I was angry at the delay.", "He was angry with me."]},
    "write": {"definition": "To form letters, words, or symbols on a surface.", "type": "verb",
              "examples": ["I write in my notebook.", "She writes stories weekly."]},
    "fast": {"definition": "Moving at high speed.", "type": "adjective",
             "examples": ["The car is fast.", "He runs fast."]},
    "house": {"definition": "A building for human habitation.", "type": "noun",
              "examples": ["I live in a house.", "The house is big."]},
    "think": {"definition": "To have ideas or form opinions.", "type": "verb",
              "examples": ["I think it will rain.", "She thinks carefully."]},
    "smart": {"definition": "Having quick intelligence or cleverness.", "type": "adjective",
              "examples": ["The AI is smart.", "He is very smart."]},
    "dog": {"definition": "A domesticated carnivorous mammal (canine).", "type": "noun",
            "examples": ["I have a dog.", "The dog barked."]},
    "cat": {"definition": "A small domesticated carnivorous mammal (feline).", "type": "noun",
            "examples": ["The cat slept.", "Cats like milk."]},
    "sleep": {"definition": "A natural periodic state of rest.", "type": "verb",
              "examples": ["I sleep eight hours.", "She is sleeping now."]},
    "cold": {"definition": "At a low temperature.", "type": "adjective",
             "examples": ["It is cold outside.", "The water is cold."]},
    # --- expand with many more words for practicality (below we add many common words) ---
}

# Add many more common words programmatically (to expand dictionary size quickly)
more_words = {
    "and":"Conjunction used to connect words or clauses.",
    "or":"Conjunction used to indicate alternatives.",
    "if":"Conjunction used to introduce conditional clauses.",
    "but":"Conjunction expressing contrast.",
    "the":"Definite article.",
    "a":"Indefinite article.",
    "an":"Indefinite article before vowels.",
    "is":"Third-person singular form of 'be'.",
    "are":"Plural form of 'be'.",
    "was":"Past tense of 'be'.",
    "were":"Past plural of 'be'.",
    "have":"To possess or hold.",
    "has":"Third-person singular of 'have'.",
    "do":"To perform an action.",
    "does":"Third-person singular of 'do'.",
    "did":"Past tense of 'do'.",
    "can":"Ability or permission.",
    "could":"Past tense or conditional of 'can'.",
    "will":"Future auxiliary verb.",
    "would":"Conditional auxiliary.",
    "should":"Advice or obligation.",
    "may":"Possibility or permission.",
    "must":"Strong obligation.",
    "say":"To utter words.",
    "said":"Past of 'say'.",
    "go":"To move from one place to another.",
    "went":"Past of 'go'.",
    "come":"To approach.",
    "came":"Past of 'come'.",
    "make":"To create.",
    "made":"Past of 'make'.",
    "know":"To have knowledge.",
    "knew":"Past of 'know'.",
    "see":"To perceive with eyes.",
    "saw":"Past of 'see'.",
    "use":"To put to use.",
    "used":"Past of 'use'.",
    "work":"To labor or operate.",
    "worked":"Past of 'work'.",
    "life":"The existence of living beings.",
    "world":"The earth and its inhabitants.",
    "people":"Plural of person.",
    "good":"Positive quality.",
    "bad":"Negative quality.",
    "new":"Not existing before.",
    "old":"Having existed for a long time.",
    "first":"Coming before all others.",
    "last":"Final or most recent.",
    "big":"Large in size.",
    "small":"Little in size.",
    "many":"A large number of.",
    "few":"A small number of.",
    "most":"The majority of.",
    "some":"An unspecified quantity.",
    "all":"Entire quantity.",
    "none":"Not any.",
    "every":"Each one of a group.",
    "always":"At all times.",
    "never":"Not at any time.",
    "often":"Frequently.",
    "sometimes":"Occasionally.",
    "rarely":"Not often.",
    "today":"This day.",
    "yesterday":"The day before today.",
    "tomorrow":"The day after today.",
    "morning":"Early part of the day.",
    "afternoon":"Part of the day after noon.",
    "evening":"Late part of the day.",
    "night":"The period of darkness.",
    "year":"Twelve months.",
    "month":"About 30 days.",
    "week":"Seven days.",
    "hour":"Sixty minutes."
}
# inject more_words into SEED_DICTIONARY with simple struct
for w,k in more_words.items():
    if w not in SEED_DICTIONARY:
        SEED_DICTIONARY[w] = {"definition": k, "type": "common", "examples": []}

# Merge external dictionary if provided (lowercase keys)
external_dict = load_json(DICT_FILE, None)
if external_dict and isinstance(external_dict, dict):
    # external entries likely full objects; merge
    DICTIONARY = {**{k.lower():v for k,v in SEED_DICTIONARY.items()}, **{k.lower():v for k,v in external_dict.items()}}
else:
    DICTIONARY = {k.lower(): v for k, v in SEED_DICTIONARY.items()}

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
    return re.findall(r"[a-zA-Z']+", text.lower())

def merged_definitions() -> Dict[str, Dict[str, Any]]:
    d = {**DICTIONARY}
    learned = ai_state.get("learned", {})
    for k, v in learned.items():
        d[k.lower()] = {"definition": v.get("definition",""), "type": v.get("type","learned"), "examples": v.get("examples", [])}
    return d

# ---------- Learning patterns ----------
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
            return normalize_word(left_token), right
    return None, None

# ---------- Retrieval & Generative helpers ----------
def format_definition_reply(key: str, entry: Dict[str,Any]) -> str:
    typ = entry.get("type","")
    definition = entry.get("definition","")
    examples = entry.get("examples",[])
    ex_text = ("\nExamples:\n - " + "\n - ".join(examples)) if examples else ""
    return f"**{key}** ({typ}): {definition}{ex_text}"

def retrieve_from_memory_or_learned(query: str) -> str:
    qtokens = set(simple_tokenize(query))
    best_score = 0
    best_text = None
    # search past conversations (assistant replies and user messages)
    for conv in ai_state.get("conversations", []):
        text = (conv.get("text","") or "")
        tokens = set(simple_tokenize(text))
        score = len(qtokens & tokens)
        if score > best_score:
            best_score = score
            best_text = text
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

# Markov chain (word-level) trained from examples + history
class MarkovGenerator:
    def __init__(self):
        self.model = {}  # (w1,w2) -> {w3:count}
        self.start_tokens = []

    def train_from_text(self, text: str):
        toks = simple_tokenize(text)
        if len(toks) < 3:
            return
        self.start_tokens.append((toks[0].lower(), toks[1].lower()))
        for i in range(len(toks)-2):
            key = (toks[i].lower(), toks[i+1].lower())
            nxt = toks[i+2].lower()
            self.model.setdefault(key, {})
            self.model[key][nxt] = self.model[key].get(nxt,0) + 1

    def generate(self, seed: str=None, max_words:int=50) -> str:
        if seed:
            seed_tokens = simple_tokenize(seed)
            if len(seed_tokens) >= 2:
                key = (seed_tokens[-2].lower(), seed_tokens[-1].lower())
            elif len(seed_tokens) == 1 and self.start_tokens:
                key = self.start_tokens[0]
            else:
                key = random.choice(self.start_tokens) if self.start_tokens else None
        else:
            key = random.choice(self.start_tokens) if self.start_tokens else None

        if not key:
            return ""

        out = [key[0], key[1]]
        for _ in range(max_words-2):
            choices = self.model.get(tuple(out[-2:]), None)
            if not choices:
                break
            # weighted random
            total = sum(choices.values())
            r = random.randint(1, total)
            acc = 0
            for word,count in choices.items():
                acc += count
                if r <= acc:
                    out.append(word)
                    break
        return " ".join(out)

# Build and keep a markov model in-memory
markov = MarkovGenerator()
def train_markov():
    # train from dictionary examples and recent conversation history
    markov.model.clear()
    markov.start_tokens.clear()
    defs = merged_definitions()
    for k,v in defs.items():
        for ex in v.get("examples", []):
            markov.train_from_text(ex)
    for conv in ai_state.get("conversations", [])[-500:]:
        txt = conv.get("text","")
        markov.train_from_text(txt)
# ensure initial train
train_markov()

# Simple sentence completion heuristic
COMMON_NEXT = {
    "i": ["am","think","want","will","have"],
    "you": ["are","have","should","can","must"],
    "we": ["are","will","should"],
    "they": ["are","have","will"],
    "the": ["same","best","next","first","end"],
    "it": ["is","was","will","seems"]
}

def complete_sentence_simple(text: str, max_words:int=20) -> str:
    tokens = simple_tokenize(text)
    out = tokens[:]
    for _ in range(max_words):
        last = out[-1] if out else ""
        cand = COMMON_NEXT.get(last.lower())
        if not cand:
            break
        out.append(random.choice(cand))
    # reconstruct (naive)
    return " ".join(out)

# ---------- Answer logic ----------
def simple_guess(user: str) -> str:
    toks = simple_tokenize(user)
    if not toks:
        return "I need more words."
    top = toks[:12]
    return " ".join(top) + " ... (I can learn more if you define terms.)"

def compose_fallback_reply(user: str) -> str:
    if user.strip().endswith("?") or user.lower().startswith(("how","what","why","when","where","who")):
        # try to answer by retrieval or educated guess
        r = retrieve_from_memory_or_learned(user)
        if r:
            return f"I found something related: {r}"
        # use markov to generate a helpful-sounding reply
        gen = markov.generate(seed=user, max_words=40)
        if gen:
            return gen.capitalize() + "."
        return ("That's a good question. I don't have a full stored answer, but I can learn if you teach me "
                "definitions with 'X means Y' or '/define X: Y'.")
    # non-question: respond conversationally
    r = retrieve_from_memory_or_learned(user)
    if r:
        return r
    gen = markov.generate(seed=user, max_words=30)
    if gen:
        return gen.capitalize() + "."
    return "I hear you. Tell me more or teach me a definition."

# ---------- Core reply generation (rule-based + retrieval) ----------
def answer_query(user_text: str) -> Dict[str, Any]:
    user = user_text.strip()
    lower = user.lower()

    # Commands
    if lower in ("/clear", "clear memory", "wipe memory"):
        ai_state["conversations"].clear()
        ai_state["learned"].clear()
        save_state()
        train_markov()
        return {"reply":"Memory cleared.", "source":"memory", "confidence":1.0}

    if lower.startswith("/delete "):
        arg = lower[len("/delete "):].strip()
        if arg.isdigit():
            idx = int(arg)-1
            if 0 <= idx < len(ai_state["conversations"]):
                removed = ai_state["conversations"].pop(idx)
                save_state()
                train_markov()
                return {"reply": f"Deleted conversation #{idx+1}: {removed.get('text')}", "source":"memory", "confidence":0.9}
            else:
                return {"reply":"Invalid conversation index.", "source":"memory", "confidence":0.2}
        else:
            key = normalize_word(arg)
            if key in ai_state.get("learned", {}):
                ai_state["learned"].pop(key)
                save_state()
                train_markov()
                return {"reply": f"Removed learned definition for '{key}'.", "source":"memory", "confidence":0.9}
            else:
                return {"reply": f"No learned definition found for '{key}'.", "source":"memory", "confidence":0.2}

    # Math detection (allow floating point, operators)
    math_expr = re.sub(r"[^0-9\.\+\-\*\/\%\(\)\s\^]", "", user)
    if any(op in math_expr for op in "+-*/%") and re.search(r"\d", math_expr):
        try:
            expr = math_expr.replace("^", "**")
            # eval with restricted builtins; allow math functions
            safe_globals = {"__builtins__": None}
            safe_locals = {k: getattr(math,k) for k in dir(math) if not k.startswith("_")}
            result = eval(expr, safe_globals, safe_locals)
            return {"reply": f"Math result: {result}", "source":"math", "confidence":1.0}
        except Exception:
            pass

    # time / date
    if re.search(r"\bwhat(?:'s| is)? the time\b|\bcurrent time\b|\btime now\b", lower):
        return {"reply": f"The current time is {datetime.now().strftime('%H:%M:%S')}", "source":"time", "confidence":1.0}
    if re.search(r"\bwhat(?:'s| is)? the date\b|\bcurrent date\b|\bdate today\b", lower):
        return {"reply": f"Today's date is {datetime.now().strftime('%Y-%m-%d')}", "source":"date", "confidence":1.0}

    # explicit define command /define word: def
    if lower.startswith("/define ") or lower.startswith("define "):
        rest = user.split(None,1)[1] if len(user.split(None,1))>1 else ""
        m = re.match(r'\s*([^\:]+)\s*[:\-]\s*(.+)', rest)
        if m:
            w = normalize_word(m.group(1))
            d = m.group(2).strip()
            ai_state.setdefault("learned", {})[w] = {"definition": d, "type":"learned", "examples": []}
            save_state()
            train_markov()
            return {"reply": f"Learned definition for '{w}'.", "source":"learning", "confidence":1.0}
        # if just /define word -> show definition if available
        m2 = re.match(r'\s*([A-Za-z\'\-]+)\s*$', rest)
        if m2:
            key = normalize_word(m2.group(1))
            defs = merged_definitions()
            if key in defs:
                entry = defs[key]
                return {"reply": format_definition_reply(key, entry), "source":"definition", "confidence":0.95}
            else:
                return {"reply": f"No definition found for '{key}'. Use '/define {key}: <definition>' to teach me.", "source":"definition", "confidence":0.2}
        return {"reply": "Usage: /define word: definition", "source":"learning", "confidence":0.3}

    # natural-sentence learning
    w, d = try_extract_definition(user)
    if w and d:
        ai_state.setdefault("learned", {})[w] = {"definition": d, "type":"learned", "examples": []}
        save_state()
        train_markov()
        return {"reply": f"Understood — saved definition: '{w}' = {d}", "source":"learning", "confidence":0.98}

    # definition lookup patterns
    m = re.search(r'\bwhat(?:\'s| is)? the meaning of ([a-zA-Z\'\-]+)\b', lower) or \
        re.search(r'\bmeaning of ([a-zA-Z\'\-]+)\b', lower)
    if m:
        key = normalize_word(m.group(1))
        defs = merged_definitions()
        if key in defs:
            entry = defs[key]
            return {"reply": format_definition_reply(key, entry), "source":"definition", "confidence":0.95}

    m2 = re.search(r'\bdefine\s+([a-zA-Z\'\-]+)\b', lower)
    if m2:
        key = normalize_word(m2.group(1))
        defs = merged_definitions()
        if key in defs:
            entry = defs[key]
            return {"reply": format_definition_reply(key, entry), "source":"definition", "confidence":0.95}
        else:
            return {"reply": f"No definition found for '{key}'.", "source":"definition", "confidence":0.2}

    # single-word input -> define if known
    if re.fullmatch(r"[A-Za-z'\-]+", user.strip()):
        key = normalize_word(user.strip())
        defs = merged_definitions()
        if key in defs:
            entry = defs[key]
            return {"reply": format_definition_reply(key, entry), "source":"definition", "confidence":0.9}

    # retrieval
    sim_reply = retrieve_from_memory_or_learned(user)
    if sim_reply:
        return {"reply": sim_reply, "source":"memory", "confidence":0.75}

    # fallback / generative
    fallback = compose_fallback_reply(user)
    return {"reply": fallback, "source":"fallback", "confidence":0.4}

# ---------- Sentence completion (UI helper) ----------
def complete_sentence(user_text: str) -> str:
    # Try markov generation first
    gen = markov.generate(seed=user_text, max_words=30)
    if gen:
        return gen
    # fallback simple completion
    return complete_sentence_simple(user_text)

# ---------- Chat UI (Streamlit) ----------
st.set_page_config(page_title="Jack — Offline Generative AI", layout="wide")
st.title("Jack — Offline Advanced Generative AI 🤖")

col1, col2 = st.columns([3,1])

with col2:
    st.markdown("## Controls & Data")
    if st.button("Clear memory & learned defs"):
        ai_state["conversations"].clear()
        ai_state["learned"].clear()
        save_state()
        train_markov()
        st.success("Cleared all memory and learned definitions.")
    if st.button("Export state (download)"):
        st.download_button("Download ai_state.json", data=json.dumps(ai_state, ensure_ascii=False, indent=2), file_name="ai_state.json")
    uploaded = st.file_uploader("Upload dictionary.json (merge)", type=["json"])
    if uploaded:
        try:
            ext = json.load(uploaded)
            if isinstance(ext, dict):
                for k,v in ext.items():
                    DICTIONARY[k.lower()] = v
                st.success("Uploaded dictionary merged into runtime dictionary.")
                train_markov()
            else:
                st.error("dictionary.json must be a JSON object mapping words to entries.")
        except Exception as e:
            st.error(f"Failed to load dictionary: {e}")

with col1:
    st.subheader("Conversation")
    # show chat history
    history = ai_state.get("conversations", [])
    start = max(0, len(history)-300)
    for i, msg in enumerate(history[start:], start+1):
        role = msg.get("role","user")
        who = "You" if role=="user" else "Jack"
        t = msg.get("time","")
        st.markdown(f"**{who}**  <span style='color:gray;font-size:12px'>{t}</span>", unsafe_allow_html=True)
        st.write(msg.get("text",""))

    user_input = st.text_area("Type your message here (Shift+Enter newline):", key="user_input", height=120)

    c1, c2, c3 = st.columns([1,1,1])
    if c1.button("Send"):
        ui = user_input.strip()
        if ui:
            res = answer_query(ui)
            reply = res["reply"]
            # store conversation
            ai_state.setdefault("conversations", []).append({"role":"user","text":ui,"time":datetime.now().isoformat()})
            ai_state.setdefault("conversations", []).append({"role":"assistant","text":reply,"time":datetime.now().isoformat()})
            save_state()
            train_markov()
            st.experimental_rerun()

    if c2.button("Complete sentence"):
        ui = user_input.strip()
        if ui:
            comp = complete_sentence(ui)
            ai_state.setdefault("conversations", []).append({"role":"user","text":ui,"time":datetime.now().isoformat()})
            ai_state.setdefault("conversations", []).append({"role":"assistant","text":comp,"time":datetime.now().isoformat()})
            save_state()
            train_markov()
            st.experimental_rerun()

    if c3.button("Teach (define)"):
        ui = user_input.strip()
        m = re.match(r'\s*([^\:]+)\s*[:\-]\s*(.+)', ui)
        if m:
            w = normalize_word(m.group(1))
            d = m.group(2).strip()
            ai_state.setdefault("learned", {})[w] = {"definition": d, "type":"learned", "examples":[]}
            save_state()
            train_markov()
            st.success(f"Learned '{w}'.")
            st.experimental_rerun()
        else:
            st.warning("To teach, enter: word: definition  (e.g. gravity: the force that attracts)")

st.markdown("---")
st.markdown("**Quick usage tips:**")
st.markdown(
"""
- Ask definition questions: `What is gravity?`, `Define gravity`  
- Teach: `gravity means the force that attracts` or `/define gravity: the force that attracts`  
- Math: type expressions like `12 * (3 + 4)` and the app will calculate.  
- Commands: `/clear`, `/delete 3` (conversation #3), `/delete word` (remove learned def)  
- Upload `dictionary.json` to merge a large offline dictionary.
"""
)

# End of file
