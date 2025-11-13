# jack_offline_saved_memory_with_corpus.py
# Jack — Offline AI with persistent memory, large KB, grammar-aware Markov, and a 500-sentence corpus
# Run:
#   pip install streamlit
#   streamlit run jack_offline_saved_memory_with_corpus.py

import streamlit as st
import json
import os
import re
import math
import random
from datetime import datetime
from typing import List, Dict, Tuple, Any, Optional

# -------------------------
# Files & Persistence
# -------------------------
STATE_FILE = "ai_state.json"
DICT_FILE = "dictionary.json"
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

# load or init state
ai_state = load_json(STATE_FILE, {"conversations": [], "learned": {}, "settings": {}, "model_meta": {}, "model_dirty": False})

# -------------------------
# Tokenization: include punctuation tokens as separate tokens
# -------------------------
WORD_RE = re.compile(r"[A-Za-z']+|[.,!?:;]")

def tokenize(text: str) -> List[str]:
    return WORD_RE.findall((text or "").lower())

# -------------------------
# Generate a 500-sentence curated corpus (programmatic, varied)
# -------------------------
def generate_corpus() -> List[str]:
    nouns = ["cat","dog","child","city","teacher","student","computer","book","car","river","mountain","friend","idea","moment","problem","day","night","house","garden","sky"]
    verbs = ["saw","likes","loves","found","reads","writes","drives","walks","jumps","learns","teaches","knows","understands","creates","builds","says","thinks","wonders","watches","helps"]
    adjectives = ["quick","slow","bright","dark","happy","sad","quiet","loud","old","young","strong","weak","big","small","warm","cold","sharp","soft","clear","noisy"]
    adverbs = ["quickly","slowly","carefully","easily","often","always","sometimes","rarely","never","happily","sadly","quietly","loudly","brightly","calmly"]
    preps = ["in","on","at","with","for","about","under","over","between","near","behind","across","through"]
    starters = ["in the morning","at night","yesterday","today","tomorrow","last week","this week","every day","sometimes","usually"]
    puncts = [".", ".", ".", "!", "?", ";", ":" ]  # make period more likely

    corpus = []
    i = 1
    # mix of statement, question, exclamation, clause sentences
    while len(corpus) < 500:
        t = i % 10
        if t == 1:
            # simple declarative: "the ADJ NOUN VERB the NOUN."
            s = f"the {random.choice(adjectives)} {random.choice(nouns)} {random.choice(verbs)} the {random.choice(nouns)}{random.choice(puncts)}"
        elif t == 2:
            # temporal clause + statement
            s = f"{random.choice(starters)}, the {random.choice(nouns)} {random.choice(verbs)} {random.choice(adverbs)}{random.choice(puncts)}"
        elif t == 3:
            # question: "do you VERB the NOUN?"
            s = f"do you {random.choice(verbs)} the {random.choice(nouns)}?"
        elif t == 4:
            # exclamation
            s = f"what a {random.choice(adjectives)} {random.choice(nouns)}!"
        elif t == 5:
            # prepositional phrase + verb
            s = f"{random.choice(preps).capitalize()} the {random.choice(nouns)}, she {random.choice(verbs)}{random.choice(puncts)}"
        elif t == 6:
            # compound: "The NOUN is ADJ, and the NOUN is ADJ."
            s = f"the {random.choice(nouns)} is {random.choice(adjectives)}, and the {random.choice(nouns)} is {random.choice(adjectives)}{random.choice(puncts)}"
        elif t == 7:
            # imperative: "Please VERB the NOUN."
            s = f"please {random.choice(verbs)} the {random.choice(nouns)}{random.choice(puncts)}"
        elif t == 8:
            # comparative phrase
            s = f"the {random.choice(nouns)} is more {random.choice(adjectives)} than the {random.choice(nouns)}{random.choice(puncts)}"
        elif t == 9:
            # colon / list style
            s = f"remember: {random.choice(nouns)}, {random.choice(nouns)}, and {random.choice(nouns)}."
        else:
            # balanced sentence with adverb
            s = f"the {random.choice(nouns)} {random.choice(verbs)} {random.choice(adverbs)}{random.choice(puncts)}"
        # small cleanups: ensure single spaces, lowercase, strip
        s = re.sub(r"\s+", " ", s).strip().lower()
        # ensure punctuation present at end (most templates include)
        if not re.search(r"[.!?;:]$", s):
            s = s + "."
        corpus.append(s)
        i += 1
    return corpus

# -------------------------
# Expanded embedded dictionary (examples & types help grammar)
# -------------------------
BASE_DICT = {
    "i": {"definition":"First-person pronoun.","type":"pronoun","examples":["i went home.","i think that's correct."]},
    "you": {"definition":"Second-person pronoun.","type":"pronoun","examples":["you are kind.","can you help me?"]},
    "we": {"definition":"First-person plural pronoun.","type":"pronoun","examples":["we agree.","we will go tomorrow."]},
    "the": {"definition":"Definite article.","type":"article","examples":["the book is on the table.","the sky is blue."]},
    "a": {"definition":"Indefinite article.","type":"article","examples":["a dog barked.","a good idea."]},
    "be": {"definition":"Exist, occur, or have a specified quality.","type":"verb","examples":["i want to be helpful.","there will be a meeting."]},
    "have": {"definition":"Possess or own.","type":"verb","examples":["i have a plan.","they have several options."]},
    "do": {"definition":"Perform an action.","type":"verb","examples":["do your best.","what did you do?"]},
    "say": {"definition":"Utter words.","type":"verb","examples":["please say it clearly.","they say it's fine."]},
    "go": {"definition":"Move from one place to another.","type":"verb","examples":["let's go now.","she goes to work."]},
    "get": {"definition":"Obtain, receive.","type":"verb","examples":["get some rest.","i got your message."]},
    "make": {"definition":"Create or form.","type":"verb","examples":["make a list.","we make progress."]},
    "know": {"definition":"Have knowledge or information.","type":"verb","examples":["i know the answer.","do you know him?"]},
    "think": {"definition":"Use reasoning or intuition.","type":"verb","examples":["i think it's right.","she thinks often."]},
    "time": {"definition":"A continuous quantity in which events occur.","type":"noun","examples":["time flies.","what time is it?"]},
    "day": {"definition":"A 24-hour period.","type":"noun","examples":["today is a good day.","we worked all day."]},
    "world": {"definition":"The earth and its inhabitants.","type":"noun","examples":["the world is changing.","she traveled the world."]},
    "life": {"definition":"Existence of living beings.","type":"noun","examples":["life is precious.","he enjoys his life."]},
    "idea": {"definition":"A thought or suggestion for possible action.","type":"noun","examples":["that's a good idea.","she shared an idea."]},
    "problem": {"definition":"A matter needing solution.","type":"noun","examples":["we solved the problem.","this is a tricky problem."]},
    "good": {"definition":"Having positive qualities.","type":"adj","examples":["a good idea.","she is good at it."]},
    "new": {"definition":"Not existing before.","type":"adj","examples":["a new book.","this is new."]},
    "first": {"definition":"Coming before others.","type":"adj","examples":["the first step.","he was first in line."]},
    "other": {"definition":"Different or distinct from the one mentioned.","type":"adj","examples":["the other person.","on the other hand."]},
    "important": {"definition":"Of great significance.","type":"adj","examples":["important work.","this is important."]},
    "very": {"definition":"To a high degree.","type":"adv","examples":["very good.","very quickly."]},
    "often": {"definition":"Frequently.","type":"adv","examples":["we often meet.","he often calls."]},
    "always": {"definition":"At all times.","type":"adv","examples":["she always smiles.","always check facts."]},
    "sometimes": {"definition":"Occasionally.","type":"adv","examples":["sometimes i read.","we sometimes travel."]},
    "however": {"definition":"Used to introduce a contrast.","type":"adv","examples":["however, it may fail.","we tried; however it didn't work."]},
    "then": {"definition":"At that time; next.","type":"adv","examples":["then we left.","finish, then rest."]},
    "in": {"definition":"Expressing location or position.","type":"prep","examples":["in the room.","living in the city."]},
    "on": {"definition":"Positioned above and in contact with.","type":"prep","examples":["on the table.","on monday."]},
    "at": {"definition":"Used for specific times/places.","type":"prep","examples":["at noon.","meet at the park."]},
    "with": {"definition":"Accompanied by.","type":"prep","examples":["with a friend.","cut with a knife."]},
    "for": {"definition":"With the purpose of.","type":"prep","examples":["for example.","i did it for you."]},
    "and": {"definition":"Conjunction joining words or phrases.","type":"conj","examples":["bread and butter.","he and she."]},
    "but": {"definition":"Conjunction showing contrast.","type":"conj","examples":["i like it but...","it was small but useful."]},
    "or": {"definition":"Conjunction indicating alternatives.","type":"conj","examples":["tea or coffee?","now or later."]},
    "if": {"definition":"Introducing a conditional clause.","type":"conj","examples":["if it rains, we'll stay.","ask if needed."]},
    "will": {"definition":"Modal verb indicating future.","type":"modal","examples":["i will go.","she will join."]},
    "can": {"definition":"Modal verb indicating ability or possibility.","type":"modal","examples":["can you help?","it can work."]},
    "there is": {"definition":"Used to state existence.","type":"phrase","examples":["there is a way to solve it.","there is a message for you."]},
    "i think": {"definition":"Phrase expressing opinion.","type":"phrase","examples":["i think we should go.","i think it's correct."]},
    "for example": {"definition":"Used to give an example.","type":"phrase","examples":["many fruits, for example apples, are healthy."]},
    "in order to": {"definition":"With the purpose to.","type":"phrase","examples":["in order to learn, practice is required."]},
    "paris": {"definition":"Capital city of France.","type":"place","examples":["paris is beautiful in spring.","i visited paris."]},
    "python": {"definition":"A high-level programming language.","type":"noun","examples":["i wrote the script in python.","python is popular."]},
    "gravity": {"definition":"Force attracting objects to one another.","type":"noun","examples":["gravity keeps us grounded.","gravity affects motion."]},
    "pi": {"definition":"Mathematical constant ≈ 3.14159.","type":"number","examples":["pi is used to compute circumference."]},
    "learn": {"definition":"Gain knowledge by study.","type":"verb","examples":["we learn from mistakes.","i want to learn more."]},
    "help": {"definition":"Provide assistance.","type":"verb","examples":["can you help me?","thanks for the help."]},
    "start": {"definition":"Begin doing something.","type":"verb","examples":["start now.","we start at nine."]},
    "finish": {"definition":"Bring to an end.","type":"verb","examples":["finish the task.","he finished quickly."]},
    "read": {"definition":"Look at and understand written words.","type":"verb","examples":["read the book.","i like to read."]},
    "write": {"definition":"Mark letters to form words.","type":"verb","examples":["write a note.","she writes code."]},
    "play": {"definition":"Take part in activity for enjoyment.","type":"verb","examples":["let's play a game.","they play music."]},
    "quick": {"definition":"Moving fast.","type":"adj","examples":["a quick reply.","quick and simple."]},
    "slow": {"definition":"Moving at a low speed.","type":"adj","examples":["a slow process.","don't be slow."]},
    "happy": {"definition":"Feeling or showing pleasure.","type":"adj","examples":["she felt happy.","a happy ending."]},
    "sad": {"definition":"Feeling sorrow.","type":"adj","examples":["a sad story.","don't be sad."]},
    "carefully": {"definition":"With care or attention.","type":"adv","examples":["read carefully.","drive carefully."]},
    "easily": {"definition":"Without difficulty.","type":"adv","examples":["it can be done easily.","she solved it easily."]},
    "man": {"definition":"An adult male human.","type":"noun","examples":["the man waved.","he was a kind man."]},
    "woman": {"definition":"An adult female human.","type":"noun","examples":["the woman smiled.","she is a strong woman."]},
    "child": {"definition":"A young person.","type":"noun","examples":["the child laughed.","children play often."]},
    "place": {"definition":"A particular position or point in space.","type":"noun","examples":["this is a good place.","where is the place?"]},
    "person": {"definition":"A human being.","type":"noun","examples":["she is a kind person.","every person matters."]},
}

# add generated corpus as a single entry so examples are picked up by Markov
BASE_DICT["__corpus__"] = {"definition": "500-sentence curated corpus to improve Markov grammar and transitions", "type": "corpus", "examples": generate_corpus()}

# merge external dictionary file if present at startup
if os.path.exists(DICT_FILE):
    ext = load_json(DICT_FILE, {})
    if isinstance(ext, dict):
        for k,v in ext.items():
            BASE_DICT[k.lower()] = v

def merged_dictionary() -> Dict[str, Dict[str,Any]]:
    d = {k.lower(): dict(v) for k,v in BASE_DICT.items()}
    for k,v in ai_state.get("learned", {}).items():
        d[k.lower()] = {"definition": v.get("definition",""), "type": v.get("type","learned"), "examples": v.get("examples",[])}
    return d

# -------------------------
# Large KB (many Q->A pairs)
# -------------------------
KB = {
    "who was the first president of the united states": "George Washington (1789–1797).",
    "who was the first us president": "George Washington (1789–1797).",
    "who was the 16th president of the united states": "Abraham Lincoln (1861–1865).",
    "when was the declaration of independence signed": "The U.S. Declaration of Independence was adopted on July 4, 1776.",
    "when did world war i start": "World War I began in 1914.",
    "when did world war i end": "World War I ended in 1918.",
    "when did world war ii start": "World War II began in 1939.",
    "when did world war ii end": "World War II ended in 1945.",
    "who discovered america": "Christopher Columbus's 1492 voyage reached the Americas for Europe; indigenous peoples lived there long before.",
    "who discovered penicillin": "Alexander Fleming is credited with discovering penicillin in 1928.",
    "capital of france": "Paris.",
    "capital of germany": "Berlin.",
    "capital of spain": "Madrid.",
    "capital of italy": "Rome.",
    "capital of united kingdom": "London.",
    "capital of the united states": "Washington, D.C.",
    "capital of canada": "Ottawa.",
    "capital of australia": "Canberra.",
    "capital of russia": "Moscow.",
    "capital of china": "Beijing.",
    "capital of japan": "Tokyo.",
    "capital of india": "New Delhi.",
    "what is gravity": "Gravity is the force by which objects with mass attract each other (≈9.81 m/s² near Earth's surface).",
    "what is photosynthesis": "A process by which plants convert light energy into chemical energy, producing oxygen and glucose from CO₂ and water.",
    "what is the largest planet": "Jupiter.",
    "what is the smallest planet": "Mercury (excluding dwarf planets).",
    "what is the sun": "The Sun is a star at the center of the Solar System that supplies light and heat to Earth.",
    "how far is the earth from the sun": "About 1 astronomical unit ≈ 149.6 million kilometers (≈93 million miles).",
    "what is dna": "DNA (deoxyribonucleic acid) stores genetic information in living organisms.",
    "what is rna": "RNA (ribonucleic acid) is involved in coding, decoding, regulation, and expression of genes.",
    "what is pi": "Pi (π) ≈ 3.141592653589793 — the ratio of a circle's circumference to its diameter.",
    "what is e": "Euler's number e ≈ 2.718281828 — the base of natural logarithms.",
    "what is the speed of light": "Approximately 299,792,458 meters per second in vacuum.",
    "what is avogadros number": "Avogadro's number ≈ 6.02214076 × 10^23 (particles per mole).",
    "what is 2 plus 2": "2 + 2 = 4.",
    "how many centimeters in a meter": "100 centimeters in 1 meter.",
    "how many meters in a kilometer": "1000 meters in 1 kilometer.",
    "how many inches in a foot": "12 inches in 1 foot.",
    "how many ounces in a pound": "16 ounces in 1 pound (avoirdupois).",
    "convert celsius to fahrenheit": "°F = °C × 9/5 + 32.",
    "what is a fever": "A fever is a raised body temperature, often a sign of infection. Adults: temps above ~38°C (100.4°F).",
    "what is dehydration": "A condition when the body loses more fluids than it takes in; symptoms include thirst and dizziness.",
    "what is python": "Python is a high-level programming language known for readability and wide use in scripting and data science.",
    "what is an api": "An API (Application Programming Interface) allows software systems to communicate and exchange data.",
    "what is machine learning": "A field of AI where models learn patterns from data to make predictions or decisions.",
    "who wrote hamlet": "William Shakespeare.",
    "who painted the mona lisa": "Leonardo da Vinci.",
    "what is the highest mountain": "Mount Everest is the highest mountain above sea level (≈8,848 m).",
    "what is the longest river": "The Nile and Amazon are both contenders depending on measurement method; commonly the Nile is cited as the longest.",
    "what currency does the united states use": "United States dollar (USD).",
    "what language is spoken in brazil": "Portuguese is the official language of Brazil.",
    "how do i boil an egg": "Place eggs in boiling water and cook 6–8 minutes for medium, 9–12 minutes for hard; cool in cold water to stop cooking.",
    "definition of computer": "An electronic device that processes data according to programmed instructions.",
    "definition of algorithm": "A step-by-step procedure for solving a problem or performing a task.",
    "what is the boiling point of water": "100 °C (212 °F) at standard atmospheric pressure (sea level).",
    "what is the freezing point of water": "0 °C (32 °F) at standard atmospheric pressure.",
    "how many seconds in a minute": "60 seconds.",
    "how many minutes in an hour": "60 minutes.",
    "how many hours in a day": "24 hours.",
    "how many days in a year": "365 days (366 in a leap year).",
    "who is elon musk": "Entrepreneur (SpaceX, Tesla, and more).",
    "who is barack obama": "44th President of the United States (2009–2017).",
    "who is albert einstein": "Physicist known for the theory of relativity.",
}

# -------------------------
# Vocab caching & vectorization
# -------------------------
_cached_vocab: List[str] = []
_cached_vocab_key = None

def build_vocab(force: bool=False) -> List[str]:
    global _cached_vocab, _cached_vocab_key
    md = merged_dictionary()
    key = (len(md), len(ai_state.get("learned",{})), len(ai_state.get("conversations",[])))
    if not force and _cached_vocab and key == _cached_vocab_key:
        return _cached_vocab
    vocab = set()
    for k,v in md.items():
        vocab.update(tokenize(k))
        vocab.update(tokenize(v.get("definition","")))
        for ex in v.get("examples",[]):
            vocab.update(tokenize(ex))
    for c in ai_state.get("conversations", [])[-200:]:
        vocab.update(tokenize(c.get("text","")))
    vocab.update(["what","who","when","where","why","how","define","means","calculate","time","date"])
    _cached_vocab = sorted(vocab)
    _cached_vocab_key = key
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
# TinyNN classifier (small & fast)
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

    def train(self, dataset: List[Tuple[List[float], int]], epochs:int=20, lr:float=0.06):
        if not dataset:
            return
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

# -------------------------
# Intents & seeds
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
    for text,intent in SEED_EXAMPLES:
        data.append((text_to_vector(text, vocab), INTENTS.index(intent)))
    for k,v in ai_state.get("learned", {}).items():
        phrase = f"{k} means {v.get('definition','')}"
        data.append((text_to_vector(phrase, vocab), INTENTS.index("teach")))
    return data

# ---------- Replace current Markov class with this improved, seed-aware version ----------
class Markov:
    def __init__(self):
        self.map = {}
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

    def _valid_tokens_set(self):
        try:
            source_vocab = set(VOCAB) if VOCAB else set(build_vocab())
        except Exception:
            source_vocab = set(build_vocab())
        md = merged_dictionary()
        dict_tokens = set()
        for k,v in md.items():
            dict_tokens.update(tokenize(k))
            dict_tokens.update(tokenize(v.get("definition","")))
            for ex in v.get("examples",[]):
                dict_tokens.update(tokenize(ex))
        punct = {".", ",", "!", "?", ":", ";"}
        return source_vocab | dict_tokens | punct

    def _is_real_word(self, tok, allowed_set):
        if tok in {".", ",", "!", "?", ":", ";"}:
            return True
        if not re.fullmatch(r"[a-zA-Z']+", tok):
            return False
        if len(tok) == 1 and tok.lower() not in ("a","i"):
            return False
        if allowed_set and tok not in allowed_set:
            return False
        return True

    def _token_type(self, tok):
        md = merged_dictionary()
        tok_l = tok.lower()
        if tok_l in md:
            return md[tok_l].get("type","")
        for k,v in md.items():
            if tok_l in tokenize(k):
                return v.get("type","")
        if tok_l in ("a","an","the"):
            return "article"
        if tok_l in ("and","or","but"):
            return "conj"
        if tok_l in ("in","on","at","with","for","to","from","by","about"):
            return "prep"
        if tok_l in ("is","are","was","were","be","am","been","being","have","has","had","do","does","did","will","can","may","should","would","could"):
            return "verb"
        if tok_l == "i":
            return "pronoun"
        if tok_l in {".", "!", "?"}:
            return "punct_end"
        if tok_l in {",", ";", ":"}:
            return "punct_mid"
        return ""

    def _pos_score(self, prev_tok, prev_prev_tok, candidate):
        # same lightweight POS bonuses as before
        cand_type = self._token_type(candidate)
        prev_type = self._token_type(prev_tok) if prev_tok else ""
        prevprev_type = self._token_type(prev_prev_tok) if prev_prev_tok else ""
        bonus = 0.0
        if prev_type in ("pronoun","noun","place","person") or prevprev_type in ("pronoun","noun"):
            if cand_type in ("verb","modal","prep"):
                bonus += 2.0
        if prev_type == "article":
            if cand_type in ("adj","noun","place","person","phrase"):
                bonus += 2.5
        if prev_type == "adj":
            if cand_type in ("noun","place","person","adv"):
                bonus += 1.8
        if prev_type == "verb" or prev_type in ("modal","aux"):
            if cand_type in ("noun","pronoun","adv","adj","phrase"):
                bonus += 1.5
        if prev_type == "prep":
            if cand_type in ("article","pronoun","noun","adj","phrase"):
                bonus += 2.2
        if prev_type == "conj":
            if cand_type in ("pronoun","noun","verb","phrase"):
                bonus += 1.6
        if prev_type == "phrase":
            if cand_type in ("verb","noun","adj","adv","phrase"):
                bonus += 1.6
        if cand_type:
            bonus += 0.1
        return bonus

    def _rank_candidates(self, key, seed_tokens=None):
        """
        Rank next-word candidates for a given bigram key.
        If seed_tokens provided, give extra score to candidates that overlap with seed.
        """
        choices = self.map.get(key, {})
        if not choices:
            return []
        allowed = self._valid_tokens_set()
        scored = []
        for w,count in choices.items():
            if not self._is_real_word(w, allowed):
                continue
            prev_tok = key[1]
            prev_prev_tok = key[0]
            bonus = self._pos_score(prev_tok, prev_prev_tok, w)
            # seed-match bonus: prefer candidate tokens that appear in the user's seed
            if seed_tokens and w in seed_tokens:
                bonus += 3.0
            score = count + bonus
            scored.append((w, score))
        if not scored:
            # fallback to raw counts if everything filtered
            for w,count in choices.items():
                scored.append((w, float(count)))
        scored.sort(key=lambda kv: kv[1], reverse=True)
        return scored

    def _find_best_backoff_key(self, seed_tokens):
        """
        If seed bigram isn't present, search the map for the best key by token overlap:
        For each candidate key, score = overlap(seed_tokens, {key0,key1} U top_followers).
        Return the key with highest score (ties resolved by random choice among top scorers).
        This biases the backoff to keys that are topically related to the seed.
        """
        if not self.map:
            return None
        best_score = -1
        best_keys = []
        seed_set = set(seed_tokens or [])
        # Precompute top followers for each key to make scoring faster (limit to top 5)
        for key, nxts in self.map.items():
            key_tokens = {key[0], key[1]}
            # pick top 5 follower tokens by count
            top_followers = sorted(nxts.items(), key=lambda kv: kv[1], reverse=True)[:5]
            follower_tokens = {w for w,_ in top_followers}
            pool = key_tokens | follower_tokens
            score = len(seed_set & pool)
            # small tie-breaker using sum of counts (prefer denser bigrams)
            total_count = sum(nxts.values())
            # combined score: overlap * 10 + log(total_count+1)
            combined = score * 10 + math.log(total_count + 1)
            if combined > best_score:
                best_score = combined
                best_keys = [key]
            elif combined == best_score:
                best_keys.append(key)
        if not best_keys:
            return None
        return random.choice(best_keys)

    def _finalize_sentence(self, words, capitalize_if=True):
        if not words:
            return ""
        s = " ".join(words)
        s = re.sub(r"\s+([,\.\?!;:])", r"\1", s)
        s = re.sub(r"\s{2,}", " ", s).strip()
        if capitalize_if and re.match(r"[a-zA-Z]", s):
            s = s[0].upper() + s[1:]
        if not re.search(r"[\.!\?]$", s):
            if not re.search(r"[,:;]$", s):
                s = s + "."
        return s

    def generate(self, seed=None, max_words=40, capitalize_if=True):
        """
        Seed-aware generation:
         - If seed has >=2 tokens and key exists -> continue from it (as before).
         - If seed key missing -> find best backoff key by token overlap with seed.
         - While ranking candidates, prefer tokens that occur in the seed (seed-match bonus).
        """
        seed_tokens = set(tokenize(seed)) if seed else set()
        # continuation mode when seed given
        if seed:
            toks = tokenize(seed)
            if len(toks) >= 2:
                key = (toks[-2].lower(), toks[-1].lower())
                # if exact key not present, try to find a related backoff key
                if key not in self.map:
                    backoff = self._find_best_backoff_key(seed_tokens)
                    if backoff:
                        key = backoff
                continuation = []
                for _ in range(max_words):
                    ranked = self._rank_candidates(key, seed_tokens=seed_tokens)
                    if not ranked:
                        # backoff: aggregate followers where first part == key[1]
                        candidates = {}
                        for (a,b), nxts in self.map.items():
                            if a == key[1]:
                                for w,cnt in nxts.items():
                                    candidates[w] = candidates.get(w,0) + cnt
                        if not candidates:
                            break
                        temp_key = (key[1], "__BACKOFF__")
                        self.map[temp_key] = candidates
                        ranked = self._rank_candidates(temp_key, seed_tokens=seed_tokens)
                        self.map.pop(temp_key, None)
                        if not ranked:
                            break
                    nxt = ranked[0][0]
                    # avoid short-token loops
                    if len(continuation) >= 2 and continuation[-1] == nxt and len(nxt) <= 2:
                        break
                    continuation.append(nxt)
                    key = (key[1], nxt)
                    # stop if we generated punctuation end
                    if re.fullmatch(r"[\.!\?]", nxt):
                        break
                if continuation:
                    return self._finalize_sentence(continuation, capitalize_if=capitalize_if)
        # No seed or couldn't continue: fall back to random-start generation (unchanged)
        if not self.starts:
            return ""
        key = random.choice(self.starts)
        out = [key[0], key[1]]
        for _ in range(max_words-2):
            ranked = self._rank_candidates((out[-2], out[-1]), seed_tokens=None)
            if not ranked:
                break
            nxt = ranked[0][0]
            out.append(nxt)
            if len(out) >= 5 and out[-1] == out[-2] == out[-3]:
                break
            if re.fullmatch(r"[\.!\?;,:]", nxt):
                break
        return self._finalize_sentence(out, capitalize_if=True)

# instantiate improved Markov
MARKOV = Markov()


def markov_serialize(m):
    out = {}
    for (a,b), nxts in m.items():
        out[f"{a}||{b}"] = nxts
    return out

def markov_deserialize(serial):
    out = {}
    for k,v in serial.items():
        a,b = k.split("||")
        out[(a,b)] = v
    return out

def train_markov_full():
    MARKOV.map.clear(); MARKOV.starts.clear()
    md = merged_dictionary()
    for k,v in md.items():
        for ex in v.get("examples", []):
            MARKOV.train(ex)
        MARKOV.train(k + " " + v.get("definition",""))
    for c in ai_state.get("conversations", []):
        MARKOV.train(c.get("text",""))
    try:
        serial = {"starts": MARKOV.starts, "map": markov_serialize(MARKOV.map)}
        save_json(MARKOV_FILE, serial)
    except Exception:
        pass

# try load persisted markov on startup (fast)
try:
    mser = load_json(MARKOV_FILE, None)
    if mser and isinstance(mser, dict) and "map" in mser:
        MARKOV.starts = mser.get("starts", [])
        MARKOV.map = markov_deserialize(mser.get("map", {}))
    else:
        train_markov_full()
except Exception:
    train_markov_full()

# -------------------------
# Retrieval helpers
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
# Build & train TinyNN model (on-demand)
# -------------------------
VOCAB: List[str] = []
NN_MODEL: Optional[TinyNN] = None

def build_and_train_model(force: bool=False):
    global VOCAB, NN_MODEL
    VOCAB = build_vocab(force=force)
    hidden_dim = max(24, len(VOCAB)//12 or 16)
    NN_MODEL = TinyNN(len(VOCAB), hidden_dim, len(INTENTS))
    dataset = build_training(VOCAB)
    if dataset:
        NN_MODEL.train(dataset, epochs=20, lr=0.06)
    ai_state["model_dirty"] = False
    save_json(STATE_FILE, ai_state)

build_and_train_model(force=False)

def incremental_model_mark_dirty():
    ai_state["model_dirty"] = True
    save_json(STATE_FILE, ai_state)

# -------------------------
# Compose reply core
# -------------------------
def format_definition(key: str, entry: Dict[str,Any]) -> str:
    ex = entry.get("examples", [])
    ex_text = ("\nExamples:\n - " + "\n - ".join(ex)) if ex else ""
    return f"{key} ({entry.get('type','')}): {entry.get('definition','')}{ex_text}"

def safe_eval_math(expr: str):
    try:
        filtered = re.sub(r"[^0-9\.\+\-\*\/\%\(\)\s\^]", "", expr)
        if not re.search(r"\d", filtered): return None
        filtered = filtered.replace("^", "**")
        result = eval(filtered, {"__builtins__": None}, {"math": math, **{k:getattr(math,k) for k in dir(math) if not k.startswith("_")}})
        return result
    except Exception:
        return None

def compose_reply(user_text: str) -> Dict[str,Any]:
    user = user_text.strip()
    lower = user.lower()

    # commands
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

    # safe math
    math_res = safe_eval_math(user)
    if math_res is not None:
        return {"reply": f"Math result: {math_res}", "meta":{"intent":"math"}}

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
            w = normalize_key(m.group(1)); d = m.group(2).strip()
            ai_state.setdefault("learned", {})[w] = {"definition": d, "type":"learned", "examples": []}
            save_json(STATE_FILE, ai_state)
            MARKOV.train(f"{w} {d}")
            incremental_model_mark_dirty()
            return {"reply": f"Learned definition for '{w}'. (Model rebuild recommended for best intent recognition.)", "meta":{"intent":"learning"}}
        m2 = re.match(r'\s*([A-Za-z\'\- ]+)\s*$', rest)
        if m2:
            key = normalize_key(m2.group(1))
            defs = merged_dictionary()
            if key in defs:
                return {"reply": format_definition(key, defs[key]), "meta":{"intent":"definition"}}
            else:
                return {"reply": f"No definition for '{key}'. Use '/define {key}: <meaning>' to teach me.", "meta":{"intent":"definition"}}
        return {"reply":"Usage: /define word: definition", "meta":{"intent":"define"}}

    # natural teaching patterns like "X means Y"
    w,d = try_extract_definition(user)
    if w and d:
        ai_state.setdefault("learned", {})[w] = {"definition": d, "type":"learned", "examples": []}
        save_json(STATE_FILE, ai_state)
        MARKOV.train(f"{w} {d}")
        incremental_model_mark_dirty()
        return {"reply": f"Saved learned definition: '{w}' = {d} (Model rebuild recommended.)", "meta":{"intent":"learning"}}

    # classification via TinyNN (if available)
    if VOCAB and NN_MODEL and not ai_state.get("model_dirty", False):
        xvec = text_to_vector(user, VOCAB)
        intent_idx = NN_MODEL.predict(xvec)
        intent = INTENTS[intent_idx]
    else:
        intent = "chat"

    # intent-driven replies
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
        return {"reply": "I don't have that definition yet. Teach me with '/define word: definition' or 'X means Y'.", "meta":{"intent":"definition"}}
    if intent == "time":
        return {"reply": f"The current time is {datetime.now().strftime('%H:%M:%S')}", "meta":{"intent":"time"}}
    if intent == "date":
        return {"reply": f"Today's date is {datetime.now().strftime('%Y-%m-%d')}", "meta":{"intent":"date"}}
    if intent == "math":
        if math_res is not None:
            return {"reply": f"Math result: {math_res}", "meta":{"intent":"math"}}

    # retrieval from memories or learned
    mem = retrieve_from_memory_or_learned(user)
    if mem:
        return {"reply": mem, "meta":{"intent":"memory"}}

    # Markov generative fallback with controlled capitalization
    prev_ends_sentence = bool(re.search(r"[\.!\?]\s*$", user))
    gen = MARKOV.generate(seed=user, max_words=50, capitalize_if=prev_ends_sentence)
    if gen:
        if gen.strip():
            if prev_ends_sentence:
                reply_text = gen
            else:
                reply_text = (user.rstrip() + " " + gen).strip()
            return {"reply": reply_text, "meta":{"intent":"gen"}}

    return {"reply": "I don't know that yet. Teach me with 'X means Y' or '/define X: Y'.", "meta":{"intent":"unknown"}}

# -------------------------
# File ingestion (txt/json)
# -------------------------
def ingest_text_content(name: str, text: str, save_as_memory: bool=True):
    if not text:
        return "No content."
    parts = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    added = 0
    for p in parts:
        key = f"ingest_{name}_{added}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        if save_as_memory:
            ai_state.setdefault("learned", {})[key] = {"definition": p[:200], "type":"ingest", "examples":[p[:200]]}
            MARKOV.train(p)
        else:
            ai_state.setdefault("conversations", []).append({"role":"assistant","text":p,"time":datetime.now().isoformat()})
            MARKOV.train(p)
        added += 1
    save_json(STATE_FILE, ai_state)
    ai_state["model_dirty"] = True
    save_json(STATE_FILE, ai_state)
    try:
        serial = {"starts": MARKOV.starts, "map": markov_serialize(MARKOV.map)}
        save_json(MARKOV_FILE, serial)
    except Exception:
        pass
    return f"Ingested {added} blocks from {name}."

# -------------------------
# UI: Streamlit
# -------------------------
st.set_page_config(page_title="Omega-B", layout="wide", page_icon="✨")
st.title("Omega-B (V2.2)")

left, right = st.columns([3,1])

with right:
    st.header("Memory & Model Controls")
    if st.button("Clear Conversation"):
        ai_state["conversations"].clear(); save_json(STATE_FILE, ai_state); st.success("Conversation cleared."); st.rerun()
    if st.button("Forget Learned Memories"):
        ai_state["learned"].clear(); save_json(STATE_FILE, ai_state); ai_state["model_dirty"] = True; save_json(STATE_FILE, ai_state); st.success("All learned memories forgotten."); st.rerun()

    st.markdown("---")
    st.write("Model status:")
    if ai_state.get("model_dirty", False):
        st.warning("Model marked DIRTY (rebuild recommended).")
    else:
        st.success("Model up-to-date.")

    if st.button("Rebuild Model (retrain TinyNN + rebuild Markov)"):
        with st.spinner("Rebuilding model — this may take a few seconds..."):
            build_and_train_model(force=True)
            train_markov_full()
            st.success("Model rebuilt.")
            st.rerun()

    st.markdown("---")
    st.markdown("**Manage Learned**")
    learned_keys = list(ai_state.get("learned", {}).keys())
    if learned_keys:
        for k in learned_keys:
            colk1, colk2 = st.columns([5,1])
            with colk1:
                st.write(f"• **{k}** — {ai_state['learned'][k].get('definition','')[:180]}")
            with colk2:
                if st.button(f"Delete {k}", key=f"del_{k}"):
                    ai_state["learned"].pop(k, None); save_json(STATE_FILE, ai_state); ai_state["model_dirty"] = True; save_json(STATE_FILE, ai_state); st.rerun()
    else:
        st.write("_No learned items yet._")

    st.markdown("---")
    st.write("Upload text or JSON to ingest (encyclopedia articles, notes).")
    uploaded = st.file_uploader("Upload .txt or .json", type=["txt","json"])
    if uploaded:
        try:
            raw = uploaded.read().decode("utf-8")
            if uploaded.name.lower().endswith(".json"):
                data = json.loads(raw)
                if isinstance(data, dict):
                    text = "\n\n".join(str(v) for v in data.values())
                elif isinstance(data, list):
                    text = "\n\n".join(str(i) for i in data)
                else:
                    text = str(data)
            else:
                text = raw
            save_as_memory = st.checkbox("Save as learned memory (recommended)", value=True)
            if st.button("Ingest file"):
                msg = ingest_text_content(uploaded.name, text, save_as_memory=save_as_memory)
                st.success(msg)
                st.rerun()
        except Exception as e:
            st.error(f"Failed to read uploaded file: {e}")

    st.markdown("---")
    st.write("State export / import")
    if st.button("Export ai_state.json"):
        st.download_button("Download ai_state.json", data=json.dumps(ai_state, ensure_ascii=False, indent=2), file_name="ai_state.json")
    import_file = st.file_uploader("Import ai_state.json (merge)", type=["json"], key="import_state")
    if import_file:
        try:
            payload = json.loads(import_file.read().decode("utf-8"))
            if isinstance(payload, dict):
                for k,v in payload.get("learned", {}).items():
                    ai_state.setdefault("learned", {})[k] = v
                for c in payload.get("conversations", []):
                    ai_state.setdefault("conversations", []).append(c)
                save_json(STATE_FILE, ai_state)
                ai_state["model_dirty"] = True; save_json(STATE_FILE, ai_state)
                st.success("Merged imported state. Model marked dirty.")
                st.rerun()
            else:
                st.error("Imported file not in expected format.")
        except Exception as e:
            st.error(f"Import failed: {e}")

with left:
    st.subheader("Conversation")
    history = ai_state.get("conversations", [])[-500:]
    if "chat_index" not in st.session_state:
        st.session_state.chat_index = len(history)
    for m in history:
        who = "You" if m.get("role","user")=="user" else "Jack"
        t = m.get("time","")
        st.markdown(f"**{who}**  <span style='color:gray;font-size:12px'>{t}</span>", unsafe_allow_html=True)
        st.write(m.get("text",""))

    st.markdown("---")
    user_input = st.text_area("Message (Shift+Enter = newline)", height=120)
    c1,c2,c3 = st.columns([1,1,1])
    if c1.button("Send"):
        ui = user_input.strip()
        if ui:
            out = compose_reply(ui)
            reply = out.get("reply","")
            ai_state.setdefault("conversations", []).append({"role":"user","text":ui,"time":datetime.now().isoformat()})
            ai_state.setdefault("conversations", []).append({"role":"assistant","text":reply,"time":datetime.now().isoformat()})
            save_json(STATE_FILE, ai_state)
            MARKOV.train(ui); MARKOV.train(reply)
            ai_state["model_dirty"] = True; save_json(STATE_FILE, ai_state)
            st.rerun()
    if c2.button("Complete"):
        ui = user_input.rstrip()
        if ui:
            prev_ends_sentence = bool(re.search(r"[\.!\?]\s*$", ui))
            cont = MARKOV.generate(seed=ui, max_words=40, capitalize_if=prev_ends_sentence)
            if cont:
                if prev_ends_sentence:
                    final = cont
                else:
                    final = (ui + " " + cont).strip()
                ai_state.setdefault("conversations", []).append({"role":"user","text":ui,"time":datetime.now().isoformat()})
                ai_state.setdefault("conversations", []).append({"role":"assistant","text":final,"time":datetime.now().isoformat()})
                MARKOV.train(ui); MARKOV.train(final)
                save_json(STATE_FILE, ai_state)
                ai_state["model_dirty"] = True; save_json(STATE_FILE, ai_state)
            else:
                gen = MARKOV.generate(max_words=40)
                ai_state.setdefault("conversations", []).append({"role":"user","text":ui,"time":datetime.now().isoformat()})
                ai_state.setdefault("conversations", []).append({"role":"assistant","text":gen,"time":datetime.now().isoformat()})
                save_json(STATE_FILE, ai_state)
            st.rerun()
    if c3.button("Teach (word: definition)"):
        ui = user_input.strip()
        m = re.match(r'\s*([^\:]+)\s*[:\-]\s*(.+)', ui)
        if m:
            w = normalize_key(m.group(1)); d = m.group(2).strip()
            ai_state.setdefault("learned", {})[w] = {"definition": d, "type":"learned", "examples": []}
            save_json(STATE_FILE, ai_state)
            MARKOV.train(f"{w} {d}")
            ai_state["model_dirty"] = True; save_json(STATE_FILE, ai_state)
            st.success(f"Learned '{w}'. (Model rebuild recommended.)")
            st.rerun()
        else:
            st.warning("To teach: enter `word: definition` (e.g. gravity: a force that pulls)")

st.markdown("---")
st.markdown("**Examples / Commands**")
st.markdown("""
- Ask a fact: `Who was the first president of the U.S.?`  
- Teach: `gravity means a force that pulls` or `/define gravity: a force that pulls`  
- Math: `12 * (3 + 4)`  
- Time/Date: `what is the time?` or `what is the date?`  
- Commands: `/clear` (clear conversation), `/forget` (clear learned memories), `/delete N` (delete conversation #N)
""")
