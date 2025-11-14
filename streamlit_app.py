# jack_offline_saved_memory_enhanced.py
# Jack — Offline AI with persistent memory, file ingest, Streamlit UI
# Restored from user's original version, with manual dictionary + KB expansion
# Improvements: Markov prefers seed-related words; capitalization/punctuation fixed.
# Run:
#   pip install streamlit
#   streamlit run jack_offline_saved_memory_enhanced.py

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
STATE_FILE = "ai_state.json"   # persistent state (conversations, learned, settings)
DICT_FILE = "dictionary.json"  # optional external dictionary to merge

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
ai_state = load_json(STATE_FILE, {"conversations": [], "learned": {}, "settings": {}, "model_meta": {}})

# -------------------------
# Small embedded dictionary (manually expanded)
# -------------------------
BASE_DICT = {
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

# merge external dictionary file if present at startup
if os.path.exists(DICT_FILE):
    ext = load_json(DICT_FILE, {})
    if isinstance(ext, dict):
        for k,v in ext.items():
            BASE_DICT[k.lower()] = v

def merged_dictionary() -> Dict[str, Dict[str,Any]]:
    """Combined base dictionary + learned items."""
    d = {k.lower(): dict(v) for k,v in BASE_DICT.items()}
    for k,v in ai_state.get("learned", {}).items():
        d[k.lower()] = {"definition": v.get("definition",""), "type": v.get("type","learned"), "examples": v.get("examples",[])}
    return d

# -------------------------
# Knowledge base for quick facts (manually expanded)
# -------------------------
KB: Dict[str,str] = {
    # capitals & geography (manual entries)
    "capital of france": "Paris",
    "capital of germany": "Berlin",
    "capital of spain": "Madrid",
    "capital of italy": "Rome",
    "capital of united states": "Washington, D.C.",
    "capital of canada": "Ottawa",
    "capital of japan": "Tokyo",
    "capital of china": "Beijing",
    "capital of india": "New Delhi",
    "capital of australia": "Canberra",
    "capital of russia": "Moscow",
    "capital of brazil": "Brasília",
    "capital of mexico": "Mexico City",
    "capital of egypt": "Cairo",
    "capital of south africa": "Pretoria (administrative); Cape Town (legislative)",

    # planets & basic astronomy
    "what is earth": "Earth is the third planet from the Sun and the only known planet to support life.",
    "what is mars": "Mars is the fourth planet, often called the red planet; it has the largest volcano in the solar system (Olympus Mons).",
    "what is jupiter": "Jupiter is the largest planet, a gas giant with a storm known as the Great Red Spot.",
    "what is saturn": "Saturn is famous for its rings formed by ice and rock particles.",
    "is pluto a planet": "Pluto is classified as a dwarf planet (reclassified in 2006).",

    # math & conversions
    "what is pi": "Pi (π) ≈ 3.14159",
    "how many centimeters in a meter": "100 centimeters in a meter.",
    "how many seconds in a minute": "60 seconds in a minute.",

    # people & culture
    "who wrote hamlet": "William Shakespeare",
    "who discovered gravity": "Isaac Newton is credited for laws of motion and universal gravitation (classical formulation).",
    "who is marie curie": "Marie Curie was a physicist/chemist who researched radioactivity and won two Nobel Prizes.",

    # cooking / food quick answers
    "how to boil an egg": "Place egg in water and bring to a gentle boil; simmer 6-12 minutes depending on desired firmness; cool in cold water.",
    "how to cook rice": "Rinse rice, use ~1.5–2 parts water to 1 part rice, bring to simmer, cover and cook until tender.",
    "how to make pasta": "Boil water with salt, add pasta, cook until al dente according to package instructions, drain and serve with sauce.",

    # common factual questions
    "what is python": "Python is a high-level programming language used for scripting, automation, and data analysis.",
    "what is gravity": "Gravity is a natural force by which all things with mass or energy are brought toward one another.",
    "what is photosynthesis": "Photosynthesis is the process plants use to convert sunlight into chemical energy (glucose).",

    # miscellaneous
    "who was the first president of the united states": "George Washington",
    "who was the first president": "George Washington",
    "largest planet": "Jupiter"
}

# -------------------------
# Tokenization & vocab
# -------------------------
WORD_RE = re.compile(r"[a-zA-Z']+|[.,!?:;]")

def tokenize(text: str) -> List[str]:
    return WORD_RE.findall((text or "").lower())

def build_vocab() -> List[str]:
    vocab = set()
    md = merged_dictionary()
    for k,v in md.items():
        vocab.update(tokenize(k))
        vocab.update(tokenize(v.get("definition","")))
        for ex in v.get("examples",[]):
            vocab.update(tokenize(ex))
    for c in ai_state.get("conversations", [])[-500:]:
        vocab.update(tokenize(c.get("text","")))
    vocab.update(["what","who","when","where","why","how","define","means","calculate","time","date"])
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
# TinyNN (from-scratch) for intent classification
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

    def train(self, dataset: List[Tuple[List[float], int]], epochs:int=40, lr:float=0.05):
        if not dataset: return
        for _ in range(epochs):
            random.shuffle(dataset)
            for x_vec, label in dataset:
                h_in = add_vec(matvec(self.W1, x_vec), self.b1)
                h = tanh_vec(h_in)
                o_in = add_vec(matvec(self.W2, h), self.b2)
                out = softmax(o_in)
                # one-hot target
                y = [0.0]*len(out); y[label] = 1.0
                err_out = [out[i] - y[i] for i in range(len(out))]
                # W2, b2 update
                for i in range(len(self.W2)):
                    for j in range(len(self.W2[0])):
                        self.W2[i][j] -= lr * err_out[i] * h[j]
                    self.b2[i] -= lr * err_out[i]
                # hidden error
                err_hidden = [0.0]*len(h)
                for j in range(len(h)):
                    s = 0.0
                    for i in range(len(err_out)):
                        s += self.W2[i][j] * err_out[i]
                    err_hidden[j] = s * (1.0 - h[j]*h[j])
                # W1, b1 update
                for j in range(len(self.W1)):
                    for k in range(len(self.W1[0])):
                        self.W1[j][k] -= lr * err_hidden[j] * x_vec[k]
                    self.b1[j] -= lr * err_hidden[j]

# -------------------------
# Intents / seed examples
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

# -------------------------
# Markov fallback generator (improved)
# -------------------------
class Markov:
    def __init__(self):
        self.map = {}
        self.starts = []

    def train(self, text: str):
        toks = tokenize(text)
        if len(toks) < 3: return
        # store starts as original-case tokens list for variety, but all keys stored lowercase
        self.starts.append((toks[0].lower(), toks[1].lower()))
        for i in range(len(toks)-2):
            key = (toks[i].lower(), toks[i+1].lower())
            nxt = toks[i+2].lower()
            self.map.setdefault(key, {})
            self.map[key][nxt] = self.map[key].get(nxt, 0) + 1

    def _score_choice(self, token: str, freq: int, seed_tokens: set) -> float:
        score = float(freq)
        if token in seed_tokens:
            score += freq * 0.6
        # substring or morphological match small bonus
        for s in seed_tokens:
            if s and (s in token or token in s):
                score += freq * 0.25
        # penalize punctuation-only tokens unless it's expected
        if re.fullmatch(r"[.,!?:;]", token):
            score *= 0.8
        return score

    def generate(self, seed: str=None, max_words:int=40) -> str:
        seed_tokens = set(tokenize(seed)) if seed else set()
        # if seed provided and has 2+ tokens, try to continue from its last bigram or find a related start
        if seed:
            toks = tokenize(seed)
            if len(toks) >= 2:
                key = (toks[-2].lower(), toks[-1].lower())
                if key not in self.map:
                    # try to find a key that shares tokens with seed
                    candidates = [k for k in self.map.keys() if (k[0] in seed_tokens or k[1] in seed_tokens)]
                    key = random.choice(candidates) if candidates else (random.choice(self.starts) if self.starts else None)
                if key:
                    out = [key[0], key[1]]
                    for _ in range(max_words-2):
                        choices = self.map.get((out[-2], out[-1]), {})
                        if not choices: break
                        # score choices by freq + seed overlap
                        best = None; best_score = -1.0
                        for token, freq in choices.items():
                            sc = self._score_choice(token, freq, seed_tokens)
                            if sc > best_score:
                                best_score = sc; best = token
                        if not best:
                            break
                        out.append(best)
                        if re.fullmatch(r"[.!?;,:]", best):
                            break
                    s = " ".join(out)
                    s = re.sub(r"\s+([,\.!\?:;])", r"\1", s)
                    # capitalization: **do not** force-capitalize the user's seed; only capitalize sentence starts
                    # if seed ends with sentence terminator, capitalize generated sentence; else keep generated capitalization natural
                    if seed and re.search(r"[.!?]\s*$", seed):
                        if s and s[0].isalpha():
                            s = s[0].upper() + s[1:]
                    else:
                        # lowercase first char to avoid double capitalization unless it's proper noun
                        if s and s[0].isalpha():
                            s = s[0].lower() + s[1:]
                    if not re.search(r"[.!?]$", s):
                        s = s + "."
                    return s
        # fallback: random start
        if not self.starts:
            return ""
        key = random.choice(self.starts)
        out = [key[0], key[1]]
        for _ in range(max_words-2):
            choices = self.map.get((out[-2], out[-1]), {})
            if not choices: break
            # pick highest freq
            nxt = max(choices.items(), key=lambda kv: kv[1])[0]
            out.append(nxt)
            if re.fullmatch(r"[.!?;,:]", nxt):
                break
        s = " ".join(out)
        s = re.sub(r"\s+([,\.!\?:;])", r"\1", s)
        if s and s[0].isalpha():
            s = s[0].upper() + s[1:]
        if not re.search(r"[.!?]$", s):
            s = s + "."
        return s

MARKOV = Markov()

def train_markov():
    MARKOV.map.clear(); MARKOV.starts.clear()
    md = merged_dictionary()
    for k,v in md.items():
        for ex in v.get("examples", []): MARKOV.train(ex)
        MARKOV.train(k + " " + v.get("definition",""))
    for c in ai_state.get("conversations", []):
        MARKOV.train(c.get("text",""))

# initial markov content
train_markov()

# -------------------------
# Retrieval and helpers
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
# Build & train TinyNN model (initial heavy train)
# -------------------------
def build_and_train_model():
    global VOCAB, NN_MODEL
    VOCAB = build_vocab()
    NN_MODEL = TinyNN(len(VOCAB), max(32, len(VOCAB)//8 or 16), len(INTENTS))
    dataset = build_training(VOCAB)
    if dataset:
        NN_MODEL.train(dataset, epochs=100, lr=0.06)

VOCAB: List[str] = []
NN_MODEL: Optional[TinyNN] = None
build_and_train_model()

def incremental_retrain():
    build_and_train_model()

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
        ai_state["conversations"].clear(); save_json(STATE_FILE, ai_state); train_markov(); return {"reply":"Chat cleared.","meta":{"intent":"memory"}}
    if lower in ("/forget", "forget"):
        ai_state["learned"].clear(); save_json(STATE_FILE, ai_state); incremental_retrain(); train_markov(); return {"reply":"Learned memory cleared.","meta":{"intent":"memory"}}
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
                ai_state["learned"].pop(key); save_json(STATE_FILE, ai_state); incremental_retrain(); train_markov(); return {"reply": f"Removed learned definition for '{key}'.", "meta":{"intent":"memory"}}
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

    # natural teaching patterns like "X means Y"
    w,d = try_extract_definition(user)
    if w and d:
        ai_state.setdefault("learned", {})[w] = {"definition": d, "type":"learned", "examples": []}
        save_json(STATE_FILE, ai_state)
        incremental_retrain(); train_markov()
        return {"reply": f"Saved learned definition: '{w}' = {d}", "meta":{"intent":"learning"}}

    # classification via TinyNN (if available)
    if VOCAB and NN_MODEL:
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

    # Markov generative fallback
    gen = MARKOV.generate(seed=user, max_words=50)
    if gen:
        # if user ends with punctuation, return generated sentence capitalized as full sentence
        if re.search(r"[\.!\?]\s*$", user):
            reply = gen.capitalize() if gen and gen[0].isalpha() else gen
        else:
            # join without forcing capitalization of user's input
            # if generated starts with lowercase letter, keep it lowercase to avoid double capitalization
            if gen and gen[0].isalpha():
                # ensure spacing is correct and punctuation aligned
                reply = user.rstrip() + (" " if not user.endswith(" ") else "") + gen
            else:
                reply = user.rstrip() + " " + gen
        return {"reply": reply, "meta":{"intent":"gen"}}

    return {"reply": "I don't know that yet. Teach me with 'X means Y' or '/define X: Y'.", "meta":{"intent":"unknown"}}

# -------------------------
# File ingestion (txt/json)
# -------------------------
def ingest_text_content(name: str, text: str, save_as_memory: bool=True):
    """Add uploaded text into learned memory or conversations. If save_as_memory True, store under learned."""
    if not text:
        return "No content."
    # simple split into paragraphs and add each as a memory entry
    parts = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    added = 0
    for p in parts:
        key = f"ingest_{name}_{added}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        if save_as_memory:
            ai_state.setdefault("learned", {})[key] = {"definition": p[:200], "type":"ingest", "examples":[p[:200]]}
        else:
            ai_state.setdefault("conversations", []).append({"role":"assistant","text":p,"time":datetime.now().isoformat()})
        added += 1
    save_json(STATE_FILE, ai_state)
    incremental_retrain(); train_markov()
    return f"Ingested {added} blocks from {name}."

# -------------------------
# UI: Streamlit Chat & Controls
# -------------------------
st.set_page_config(page_title="Jack — Offline AI (Saved Memory)", layout="wide")
st.title("Jack — Offline AI (Persistent Memory)")

left, right = st.columns([3,1])

with right:
    st.header("Memory Controls")
    if st.button("Clear Conversation"):
        ai_state["conversations"].clear()
        save_json(STATE_FILE, ai_state)
        st.success("Conversation cleared.")
        st.rerun()
    if st.button("Forget Learned Memories"):
        ai_state["learned"].clear()
        save_json(STATE_FILE, ai_state)
        incremental_retrain(); train_markov()
        st.success("All learned memories forgotten.")
        st.rerun()
    st.markdown("---")
    st.markdown("**Manage Learned**")
    # list learned items with delete buttons
    learned_keys = list(ai_state.get("learned", {}).keys())
    if learned_keys:
        for k in learned_keys:
            colk1, colk2 = st.columns([5,1])
            with colk1:
                st.write(f"• **{k}** — {ai_state['learned'][k].get('definition','')[:180]}")
            with colk2:
                if st.button(f"Delete {k}", key=f"del_{k}"):
                    ai_state["learned"].pop(k, None)
                    save_json(STATE_FILE, ai_state)
                    incremental_retrain(); train_markov()
                    st.rerun()
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
                # if it's an object with text fields, flatten: else store raw
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
                #st.rerun()
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
                # merge learned and conversations
                for k,v in payload.get("learned", {}).items():
                    ai_state.setdefault("learned", {})[k] = v
                for c in payload.get("conversations", []):
                    ai_state.setdefault("conversations", []).append(c)
                save_json(STATE_FILE, ai_state)
                incremental_retrain(); train_markov()
                st.success("Merged imported state.")
               # st.rerun()
            else:
                st.error("Imported file not in expected format.")
        except Exception as e:
            st.error(f"Import failed: {e}")

with left:
    st.subheader("Conversation")
    # show last messages
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
            incremental_retrain(); train_markov()
            st.rerun()
    if c2.button("Complete"):
        ui = user_input.strip()
        if ui:
            comp = MARKOV.generate(seed=ui, max_words=40) or (ui + " ...")
            ai_state.setdefault("conversations", []).append({"role":"user","text":ui,"time":datetime.now().isoformat()})
            ai_state.setdefault("conversations", []).append({"role":"assistant","text":comp,"time":datetime.now().isoformat()})
            save_json(STATE_FILE, ai_state)
            st.rerun()
    if c3.button("Teach (word: definition)"):
        ui = user_input.strip()
        m = re.match(r'\s*([^\:]+)\s*[:\-]\s*(.+)', ui)
        if m:
            w = normalize_key(m.group(1)); d = m.group(2).strip()
            ai_state.setdefault("learned", {})[w] = {"definition": d, "type":"learned", "examples": []}
            save_json(STATE_FILE, ai_state)
            incremental_retrain(); train_markov()
            st.success(f"Learned '{w}'.")
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
