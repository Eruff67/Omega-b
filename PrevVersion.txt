# jack_offline_fast_start.py
# Faster startup: lazy-load big dictionary and on-demand model rebuild.
# Run:
#   pip install streamlit
#   streamlit run jack_offline_fast_start.py

import streamlit as st
import json, os, re, math, random, uuid, threading
from datetime import datetime
from typing import List, Dict, Tuple, Any, Optional

# -------------------------
# Device-specific isolation (per-device conversations + privacy)
# -------------------------
DEVICE_ID_FILE = ".jack_device_id"

def get_or_create_device_id():
    if os.path.exists(DEVICE_ID_FILE):
        try:
            with open(DEVICE_ID_FILE, "r", encoding="utf-8") as f:
                return f.read().strip()
        except Exception:
            pass
    new_id = str(uuid.uuid4())[:8]  # short unique id per device
    try:
        with open(DEVICE_ID_FILE, "w", encoding="utf-8") as f:
            f.write(new_id)
    except Exception:
        pass
    return new_id

DEVICE_ID = get_or_create_device_id()

# -------------------------
# Files & Persistence (device-scoped)
# -------------------------
# ======================================================
# Device- and session-specific isolation layer
# ======================================================
import uuid, hashlib

DEVICE_ID_FILE = ".jack_device_id"

def get_or_create_device_id():
    """Create or read a unique local ID so every device/session is isolated."""
    try:
        # Try saving a small file on disk for this device
        if os.path.exists(DEVICE_ID_FILE):
            with open(DEVICE_ID_FILE, "r") as f:
                return f.read().strip()
        new_id = str(uuid.uuid4())[:8]
        with open(DEVICE_ID_FILE, "w") as f:
            f.write(new_id)
        return new_id
    except Exception:
        # Fallback for read-only or cloud environments: use Streamlit session hash
        sid = st.session_state.get("_sid")
        if not sid:
            sid = hashlib.sha1(str(uuid.uuid4()).encode()).hexdigest()[:8]
            st.session_state["_sid"] = sid
        return sid

DEVICE_ID = get_or_create_device_id()

# Use per-device JSON files so data never overlaps
STATE_FILE = f"ai_state_{DEVICE_ID}.json"
DICT_FILE = f"dictionary_{DEVICE_ID}.json"
MARKOV_FILE = f"markov_state_{DEVICE_ID}.json"

print(f"[Jack.AI] Private session active — device ID: {DEVICE_ID}")

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
ai_state = load_json(STATE_FILE, {"conversations": [], "learned": {}, "settings": {"persona":"neutral"}, "model_dirty": True})

# -------------------------
# sklearn (TF-IDF semantic) - install if missing
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

# global semantic objects
_vectorizer = None
_matrix = None
_indexed_keys = []
_vector_lock = threading.Lock()

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
    # pronouns & articles
    "i": {"definition":"first-person singular pronoun","type":"pronoun","examples":["i went home.","i think that's correct."]},
    "you": {"definition":"second-person singular/plural pronoun","type":"pronoun","examples":["you are kind.","can you help me?"]},
    "we": {"definition":"first-person plural pronoun","type":"pronoun","examples":["we will go tomorrow.","we agree."]},
    "they": {"definition":"third-person plural pronoun","type":"pronoun","examples":["they left early.","they are coming."]},
    "he": {"definition":"third-person singular male pronoun","type":"pronoun","examples":["he runs fast.","he is here."]},
    "she": {"definition":"third-person singular female pronoun","type":"pronoun","examples":["she smiled.","she works nightly."]},
    "it": {"definition":"third-person singular neutral pronoun","type":"pronoun","examples":["it is raining.","it works."]},
    "the": {"definition":"definite article","type":"article","examples":["the book is on the table.","the sky is blue."]},
    "a": {"definition":"indefinite article","type":"article","examples":["a dog barked.","a good idea."]},
    "an": {"definition":"indefinite article before vowel sounds","type":"article","examples":["an apple a day.","an honor."]},

    # common verbs
    "be": {"definition":"exist or have a specified quality","type":"verb","examples":["i want to be helpful.","there will be a meeting."]},
    "have": {"definition":"possess, own, or hold","type":"verb","examples":["i have a plan.","they have several options."]},
    "do": {"definition":"perform an action","type":"verb","examples":["do your best.","what did you do?"]},
    "say": {"definition":"utter words","type":"verb","examples":["please say it clearly.","they say it's fine."]},
    "go": {"definition":"move from one place to another","type":"verb","examples":["let's go now.","she goes to work."]},
    "get": {"definition":"obtain or receive","type":"verb","examples":["get some rest.","i got your message."]},
    "make": {"definition":"create or form","type":"verb","examples":["make a list.","we make progress."]},
    "know": {"definition":"be aware of or familiar with","type":"verb","examples":["i know the answer.","do you know him?"]},
    "think": {"definition":"use reasoning or intuition","type":"verb","examples":["i think it's right.","she thinks often."]},
    "see": {"definition":"perceive with the eyes","type":"verb","examples":["i see a bird.","did you see that?"]},
    "look": {"definition":"direct one's gaze toward","type":"verb","examples":["look at the map.","look both ways."]},
    "use": {"definition":"employ for a purpose","type":"verb","examples":["use a pen.","we use tools."]},
    "work": {"definition":"be engaged in physical or mental activity to achieve a result","type":"verb","examples":["i work daily.","this works well."]},
    "call": {"definition":"name or contact someone","type":"verb","examples":["call me later.","they called a meeting."]},
    "try": {"definition":"make an attempt","type":"verb","examples":["try again.","i will try my best."]},

    # action verbs (food & general)
    "eat": {"definition":"consume food","type":"verb","examples":["i eat breakfast.","eat slowly."]},
    "drink": {"definition":"consume a liquid","type":"verb","examples":["drink water.","he drinks coffee."]},
    "cook": {"definition":"prepare food by heating","type":"verb","examples":["cook the rice.","she cooks dinner."]},
    "bake": {"definition":"cook food by dry heat","type":"verb","examples":["bake a cake.","bake until golden."]},
    "stir": {"definition":"mix by moving a utensil in a circular pattern","type":"verb","examples":["stir the soup.","stir gently."]},
    "chop": {"definition":"cut into small pieces","type":"verb","examples":["chop the onions.","chop finely."]},
    "slice": {"definition":"cut into thin pieces","type":"verb","examples":["slice the bread.","slice thinly."]},
    "fry": {"definition":"cook in hot fat or oil","type":"verb","examples":["fry until golden.","fry the onions."]},
    "grill": {"definition":"cook over direct heat","type":"verb","examples":["grill the chicken.","grill on high heat."]},

    # common nouns: people / household / food / places
    "man": {"definition":"an adult male human","type":"noun","examples":["the man waved.","he was a kind man."]},
    "woman": {"definition":"an adult female human","type":"noun","examples":["the woman smiled.","she is a strong woman."]},
    "child": {"definition":"a young person","type":"noun","examples":["the child laughed.","children play often."]},
    "friend": {"definition":"a person attached by feelings of affection or personal regard","type":"noun","examples":["my friend helped me.","she is a friend."]},
    "family": {"definition":"a group of people related by blood or marriage","type":"noun","examples":["my family is large.","family gatherings are fun."]},
    "house": {"definition":"a building for human habitation","type":"noun","examples":["the house is on the corner.","we cleaned the house."]},
    "car": {"definition":"a road vehicle powered by an engine","type":"noun","examples":["the car stopped.","she drove the car."]},
    "city": {"definition":"a large town","type":"noun","examples":["the city is busy.","visit the city center."]},
    "country": {"definition":"a nation with its own government","type":"noun","examples":["i travel to another country.","the country is beautiful."]},
    "restaurant": {"definition":"a place where people pay to sit and eat meals","type":"noun","examples":["we ate at a restaurant.","the restaurant serves lunch."]},

    # foods (expanded manual list)
    "apple": {"definition":"a common fruit","type":"food","examples":["i like apples.","an apple a day keeps doctors away."]},
    "banana": {"definition":"a long yellow fruit","type":"food","examples":["banana smoothies are tasty.","peel the banana."]},
    "orange": {"definition":"a citrus fruit high in vitamin C","type":"food","examples":["orange juice is refreshing.","peel the orange."]},
    "bread": {"definition":"a staple food made from flour","type":"food","examples":["i bought fresh bread.","toast the bread."]},
    "cheese": {"definition":"a dairy product made from curdled milk","type":"food","examples":["cheese melts well.","she loves cheese."]},
    "rice": {"definition":"a cereal grain widely consumed","type":"food","examples":["cook rice with water.","rice is a staple."]},
    "pasta": {"definition":"an Italian staple made from dough","type":"food","examples":["boil pasta until al dente.","serve with sauce."]},
    "tomato": {"definition":"a red fruit often used as a vegetable","type":"food","examples":["slice the tomato.","tomato is common in salad."]},
    "potato": {"definition":"a starchy tuber","type":"food","examples":["bake the potato.","mashed potatoes are tasty."]},
    "chicken": {"definition":"meat from a domesticated bird","type":"food","examples":["roast the chicken.","chicken soup is warm."]},
    "beef": {"definition":"meat from cattle","type":"food","examples":["grill the beef.","beef stew is hearty."]},
    "fish": {"definition":"animals that live in water used for food","type":"food","examples":["grill the fish.","fish is a healthy option."]},
    "egg": {"definition":"an oval reproductive body produced by birds","type":"food","examples":["scramble the eggs.","boil the egg."]},
    "milk": {"definition":"a white liquid produced by mammals","type":"food","examples":["pour some milk.","milk in cereal."]},
    "butter": {"definition":"a dairy product made from churned cream","type":"food","examples":["spread butter on bread.","butter melts in the pan."]},
    "coffee": {"definition":"a brewed drink from roasted coffee beans","type":"food","examples":["i drink coffee in the morning.","black coffee is strong."]},
    "tea": {"definition":"a hot or cold drink from steeped tea leaves","type":"food","examples":["green tea is popular.","please pass the tea."]},
    "sugar": {"definition":"a sweet crystalline substance","type":"food","examples":["add sugar to taste.","sugar dissolves in tea."]},

    # adjectives/adverbs
    "good": {"definition":"having desirable qualities","type":"adj","examples":["a good idea.","she is good at it."]},
    "bad": {"definition":"not acceptable","type":"adj","examples":["that is bad.","a bad result."]},
    "happy": {"definition":"feeling or showing pleasure","type":"adj","examples":["she felt happy.","a happy ending."]},
    "sad": {"definition":"feeling sorrow","type":"adj","examples":["a sad story.","don't be sad."]},
    "quick": {"definition":"moving fast","type":"adj","examples":["a quick reply.","act quick."]},
    "slow": {"definition":"moving at low speed","type":"adj","examples":["a slow process.","do not be slow."]},
    "very": {"definition":"to a high degree","type":"adv","examples":["very good.","very quickly."]},
    "often": {"definition":"frequently","type":"adv","examples":["we often meet.","he often calls."]},
    "carefully": {"definition":"with care or attention","type":"adv","examples":["read carefully.","drive carefully."]},

    # prepositions/conjunctions/common phrases
    "in": {"definition":"expressing location or position","type":"prep","examples":["in the room.","living in the city."]},
    "on": {"definition":"positioned above and in contact with","type":"prep","examples":["on the table.","on monday."]},
    "at": {"definition":"used for specific times/places","type":"prep","examples":["at noon.","meet at the park."]},
    "with": {"definition":"accompanied by","type":"prep","examples":["with a friend.","cut with a knife."]},
    "for": {"definition":"with the purpose of","type":"prep","examples":["for example.","i did it for you."]},
    "and": {"definition":"conjunction joining words or phrases","type":"conj","examples":["bread and butter.","he and she."]},
    "but": {"definition":"conjunction showing contrast","type":"conj","examples":["i like it but...","it was small but useful."]},
    "or": {"definition":"conjunction indicating alternatives","type":"conj","examples":["tea or coffee?","now or later."]},
    "if": {"definition":"introducing a conditional clause","type":"conj","examples":["if it rains, we'll stay.","ask if needed."]},
    "because": {"definition":"for the reason that","type":"conj","examples":["i left because i was tired.","stay because it's safe."]},
    "when": {"definition":"at the time that","type":"conj","examples":["call me when you arrive.","when it rains."]},
    "where": {"definition":"in or to what place","type":"adv","examples":["where are you?","where did it go?"]},

    # numbers / time / date
    "one": {"definition":"the number 1","type":"number","examples":["one plus one equals two.","just one left."]},
    "two": {"definition":"the number 2","type":"number","examples":["two times three.","two of them."]},
    "three": {"definition":"the number 3","type":"number","examples":["three days.","three people."]},
    "today": {"definition":"the present day","type":"time","examples":["today is sunny.","what about today?"]},
    "tomorrow": {"definition":"the day after today","type":"time","examples":["see you tomorrow.","tomorrow we'll start."]},
    "yesterday": {"definition":"the day before today","type":"time","examples":["yesterday was busy.","remember yesterday?"]},

    # places & geography
    "paris": {"definition":"capital of France","type":"place","examples":["paris is beautiful in spring.","i visited paris."]},
    "london": {"definition":"capital of the UK","type":"place","examples":["london is busy.","visit london."]},
    "new york": {"definition":"major US city (state: New York)","type":"place","examples":["new york city is large.","i lived in new york."]},
    "tokyo": {"definition":"capital of Japan","type":"place","examples":["tokyo is a metropolis.","i like tokyo."]},
    "sydney": {"definition":"major Australian city","type":"place","examples":["sydney has a famous harbor.","visit sydney."]},

    # months / days
    "january": {"definition":"first month of the year","type":"time","examples":["in january we plan.","january is cold."]},
    "february": {"definition":"second month of the year","type":"time","examples":["valentines are in february."]},
    "march": {"definition":"third month of the year","type":"time","examples":["we travel in march."]},
    "monday": {"definition":"first weekday","type":"time","examples":["monday starts the week."]},
    "friday": {"definition":"fifth weekday","type":"time","examples":["friday is near weekend."]},

    # common names (people) - a few examples
    "george washington": {"definition":"First President of the United States (1789–1797).","type":"proper_noun","examples":["George Washington led the Continental Army."]},
    "abraham lincoln": {"definition":"16th U.S. President who led during the Civil War.","type":"proper_noun","examples":["Abraham Lincoln issued the Emancipation Proclamation."]},
    "isaac newton": {"definition":"English mathematician and physicist, formulated laws of motion and gravity.","type":"proper_noun","examples":["Isaac Newton published Principia Mathematica."]},
    "marie curie": {"definition":"Polish-French physicist and chemist who conducted pioneering radioactivity research.","type":"proper_noun","examples":["Marie Curie discovered polonium and radium."]},
    "william shakespeare": {"definition":"English playwright and poet, author of Hamlet and many plays.","type":"proper_noun","examples":["Shakespeare wrote Hamlet, Othello and Macbeth."]},

    # small corpus example
    "__corpus__": {"definition":"starter corpus","type":"corpus","examples":["the cat sat on the mat.","do you like apples?","i went to the market and bought fresh bread."]}
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
# Semantic helpers (sklearn TF-IDF + cosine)
# -------------------------
def rebuild_semantic_index(force: bool=False):
    """Build TF-IDF matrix over dictionary + learned definitions (thread-safe)."""
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
            # mark model as up-to-date
            ai_state["model_dirty"] = False
            save_json(STATE_FILE, ai_state)
        except Exception as e:
            print("Failed to build semantic index:", e)

def find_similar_terms(query: str, topn: int = 6):
    """Find similar words or definitions to a query using cosine similarity."""
    global _vectorizer, _matrix, _indexed_keys
    if TfidfVectorizer is None:
        return []
    with _vector_lock:
        if _vectorizer is None or _matrix is None:
            rebuild_semantic_index()
        try:
            vec = _vectorizer.transform([query])
            sims = cosine_similarity(vec, _matrix)[0]
            top_idx = sims.argsort()[::-1][:topn]
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
# Persona system (clean, colorful)
# -------------------------
def apply_persona(text: str, persona: str) -> str:
    persona = (persona or "neutral").lower()
    if persona == "cowboy":
        # informal, friendly, a bit folksy
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
    # neutral or unknown
    return text

# -------------------------
# Compose reply (central respond helper)
# -------------------------
def respond(reply_text: str, meta: Dict[str,Any]) -> Dict[str,Any]:
    persona = ai_state.get("settings", {}).get("persona", "neutral")
    reply_text = apply_persona(reply_text, persona)
    return {"reply": reply_text, "meta": meta}

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
        ai_state["conversations"].clear(); save_json(STATE_FILE, ai_state); return respond("Chat cleared.", {"intent":"memory"})
    if lower in ("/forget", "forget"):
        ai_state["learned"].clear(); save_json(STATE_FILE, ai_state); ai_state["model_dirty"]=True; save_json(STATE_FILE, ai_state)
        # trigger semantic rebuild in background
        threading.Thread(target=rebuild_semantic_index).start()
        return respond("Learned memory cleared.", {"intent":"memory"})
    if lower.startswith("/delete "):
        arg = lower[len("/delete "):].strip()
        if arg.isdigit():
            idx = int(arg)-1
            if 0 <= idx < len(ai_state.get("conversations", [])):
                removed = ai_state["conversations"].pop(idx)
                save_json(STATE_FILE, ai_state)
                return respond(f"Deleted conversation #{idx+1}: {removed.get('text')}", {"intent":"memory"})
            else:
                return respond("Invalid conversation index.", {"intent":"error"})
        else:
            key = normalize_key(arg)
            if key in ai_state.get("learned", {}):
                ai_state["learned"].pop(key); save_json(STATE_FILE, ai_state); ai_state["model_dirty"]=True; save_json(STATE_FILE, ai_state)
                threading.Thread(target=rebuild_semantic_index).start()
                return respond(f"Removed learned definition for '{key}'.", {"intent":"memory"})
            else:
                return respond(f"No learned definition for '{key}'.", {"intent":"error"})
    # persona command
    if lower.startswith("/persona ") or lower.startswith("persona "):
        parts = user.split(None,1)
        if len(parts) > 1:
            p = parts[1].strip().lower()
            ai_state.setdefault("settings", {})["persona"] = p
            save_json(STATE_FILE, ai_state)
            return respond(f"Persona set to '{p}'.", {"intent":"persona"})
        else:
            return respond(f"Current persona: {ai_state.get('settings',{}).get('persona','neutral')}", {"intent":"persona"})
    # math
    math_res = safe_eval_math(user)
    if math_res is not None:
        return respond(f"Math result: {math_res}", {"intent":"math"})
    # time/date
    if re.search(r"\bwhat(?:'s| is)? the time\b|\btime now\b|\bcurrent time\b", lower):
        return respond(f"The current time is {datetime.now().strftime('%H:%M:%S')}", {"intent":"time"})
    if re.search(r"\bwhat(?:'s| is)? the date\b|\bcurrent date\b|\bdate today\b", lower):
        return respond(f"Today's date is {datetime.now().strftime('%Y-%m-%d')}", {"intent":"date"})
    # define command
    if lower.startswith("/define ") or lower.startswith("define "):
        rest = user.split(None,1)[1] if len(user.split(None,1))>1 else ""
        m = re.match(r'\s*([^\:]+)\s*[:\-]\s*(.+)', rest)
        if m:
            w = normalize_key(m.group(1)); d = m.group(2).strip()
            ai_state.setdefault("learned", {})[w] = {"definition": d, "type":"learned", "examples": []}
            save_json(STATE_FILE, ai_state)
            ai_state["model_dirty"] = True; save_json(STATE_FILE, ai_state)
            # rebuild semantic index in background
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
    # natural teach patterns
    w,d = try_extract_definition(user)
    if w and d:
        ai_state.setdefault("learned", {})[w] = {"definition": d, "type":"learned", "examples": []}
        save_json(STATE_FILE, ai_state)
        ai_state["model_dirty"] = True; save_json(STATE_FILE, ai_state)
        threading.Thread(target=rebuild_semantic_index).start()
        return respond(f"Saved learned definition: '{w}' = {d}", {"intent":"learning"})
    # quick KB lookup
    ans, conf = lookup_kb(user)
    if ans:
        return respond(str(ans), {"intent":"fact","confidence":conf})
    # retrieval
    mem = retrieve_from_memory_or_learned(user)
    if mem:
        return respond(mem, {"intent":"memory"})
    # semantic dictionary search (sklearn powered)
    sem = semantic_answer(user)
    if sem:
        return respond(sem, {"intent":"semantic"})
    # paragraph generation if requested
    if paragraph_sentences and paragraph_sentences > 0:
        para = MARKOV.generate_paragraph(seed=(user if user else None), topic=topic, num_sentences=paragraph_sentences)
        if para:
            return respond(para, {"intent":"gen_paragraph"})
    # markov single generation
    gen = MARKOV.generate(seed=user)
    if gen:
        if re.search(r"[\.!\?]\s*$", user):
            reply_text = gen
        else:
            reply_text = (user.rstrip() + " " + gen).strip()
        return respond(reply_text, {"intent":"gen"})
    return respond("I don't know that yet. Teach me with 'X means Y' or '/define X: Y'.", {"intent":"unknown"})

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
st.set_page_config(page_title="Omega-B             ", layout="wide")
st.title("Omega-B")

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
        # Mark model dirty to reflect new vocab and rebuild semantic index in background
        ai_state["model_dirty"] = True; save_json(STATE_FILE, ai_state)
        threading.Thread(target=rebuild_semantic_index).start()

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

    if st.button("Rebuild Semantic Model (TF-IDF)"):
        with st.spinner("Rebuilding semantic TF-IDF index..."):
            rebuild_semantic_index(force=True)
            st.success("Semantic model rebuilt.")
            st.rerun()

    st.markdown("---")
    st.write("Persisted Markov: " + ("loaded" if _markov_loaded else "not found"))
    if st.button("Clear learned memories"):
        ai_state["learned"].clear(); save_json(STATE_FILE, ai_state); st.success("Learned cleared."); st.rerun()

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
            save_json(STATE_FILE, ai_state)
            # light on-the-fly training: add the user+reply to markov to improve future generations
            MARKOV.train(ui); MARKOV.train(reply)
            ai_state["model_dirty"] = True; save_json(STATE_FILE, ai_state)
            # trigger semantic rebuild asynchronously (keeps UI responsive)
            threading.Thread(target=rebuild_semantic_index).start()
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
            threading.Thread(target=rebuild_semantic_index).start()
            st.rerun()

    if c3.button("Teach (word: definition)"):
        ui = user_input.strip()
        m = re.match(r'\s*([^\:]+)\s*[:\-]\s*(.+)', ui)
        if m:
            w = normalize_key(m.group(1)); d = m.group(2).strip()
            ai_state.setdefault("learned", {})[w] = {"definition": d, "type":"learned", "examples": []}
            save_json(STATE_FILE, ai_state)
            ai_state["model_dirty"] = True; save_json(STATE_FILE, ai_state)
            threading.Thread(target=rebuild_semantic_index).start()
            st.success(f"Learned '{w}'.")
            st.rerun()
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
st.caption(f"Device session id: {DEVICE_ID} (this device's conversations are private).")
