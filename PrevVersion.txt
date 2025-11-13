# jack_offline_saved_memory_combined.py
# Jack — Offline AI with persistent memory, file ingest, and Streamlit UI
# Features: expanded dictionary, fast incremental Markov, real-word filtering, persisted Markov, on-demand rebuild.
#
# Run:
#   pip install streamlit
#   streamlit run jack_offline_saved_memory_combined.py

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
STATE_FILE = "ai_state.json"      # persistent state (conversations, learned, settings)
DICT_FILE = "dictionary.json"     # optional external dictionary to merge
MARKOV_FILE = "markov_state.json" # persisted markov map for fast startup

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
# Expanded embedded dictionary
# (keep this reasonably sized here — you can upload larger via dictionary.json)
# -------------------------
BASE_DICT = {
    # pronouns & articles
    "i": {"definition":"First-person pronoun.","type":"pronoun","examples":["I went home.","I think that's correct."]},
    "you": {"definition":"Second-person pronoun.","type":"pronoun","examples":["You are kind.","Can you help me?"]},
    "we": {"definition":"First-person plural pronoun.","type":"pronoun","examples":["We agree.","We will go tomorrow."]},
    "the": {"definition":"Definite article.","type":"article","examples":["The book is on the table.","The sky is blue."]},
    "a": {"definition":"Indefinite article.","type":"article","examples":["A dog barked.","A good idea."]},

    # verbs
    "be": {"definition":"Exist, occur, or have a specified quality.","type":"verb","examples":["I want to be helpful.","There will be a meeting."]},
    "have": {"definition":"Possess or own.","type":"verb","examples":["I have a plan.","They have several options."]},
    "do": {"definition":"Perform an action.","type":"verb","examples":["Do your best.","What did you do?"]},
    "say": {"definition":"Utter words.","type":"verb","examples":["Please say it clearly.","They say it's fine."]},
    "go": {"definition":"Move from one place to another.","type":"verb","examples":["Let's go now.","She goes to work."]},
    "get": {"definition":"Obtain, receive.","type":"verb","examples":["Get some rest.","I got your message."]},
    "make": {"definition":"Create or form.","type":"verb","examples":["Make a list.","We make progress."]},
    "know": {"definition":"Have knowledge or information.","type":"verb","examples":["I know the answer.","Do you know him?"]},
    "think": {"definition":"Use reasoning or intuition.","type":"verb","examples":["I think it's right.","She thinks often."]},
    "take": {"definition":"Lay hold of or carry.","type":"verb","examples":["Take an umbrella.","He took the train."]},
    "see": {"definition":"Perceive with the eyes.","type":"verb","examples":["I see the point.","Can you see it?"]},
    "come": {"definition":"Move towards or arrive.","type":"verb","examples":["Come here.","He came late."]},
    "look": {"definition":"Direct one's gaze.","type":"verb","examples":["Look at this.","She looked surprised."]},

    # nouns
    "time": {"definition":"A continuous quantity in which events occur.","type":"noun","examples":["Time flies.","What time is it?"]},
    "day": {"definition":"A 24-hour period.","type":"noun","examples":["Today is a good day.","We worked all day."]},
    "world": {"definition":"The earth and its inhabitants.","type":"noun","examples":["The world is changing.","She traveled the world."]},
    "life": {"definition":"Existence of living beings.","type":"noun","examples":["Life is precious.","He enjoys his life."]},
    "idea": {"definition":"A thought or suggestion for possible action.","type":"noun","examples":["That's a good idea.","She shared an idea."]},
    "problem": {"definition":"A matter needing solution.","type":"noun","examples":["We solved the problem.","This is a tricky problem."]},

    # adjectives
    "good": {"definition":"Having positive qualities.","type":"adj","examples":["A good idea.","She is good at it."]},
    "new": {"definition":"Not existing before.","type":"adj","examples":["A new book.","This is new."]},
    "first": {"definition":"Coming before others.","type":"adj","examples":["The first step.","He was first in line."]},
    "other": {"definition":"Different or distinct from the one mentioned.","type":"adj","examples":["The other person.","On the other hand."]},
    "important": {"definition":"Of great significance.","type":"adj","examples":["Important work.","This is important."]},

    # adverbs/connectors
    "very": {"definition":"To a high degree.","type":"adv","examples":["Very good.","Very quickly."]},
    "often": {"definition":"Frequently.","type":"adv","examples":["We often meet.","He often calls."]},
    "always": {"definition":"At all times.","type":"adv","examples":["She always smiles.","Always check facts."]},
    "sometimes": {"definition":"Occasionally.","type":"adv","examples":["Sometimes I read.","We sometimes travel."]},
    "however": {"definition":"Used to introduce a contrast.","type":"adv","examples":["However, it may fail.","We tried; however it didn't work."]},
    "then": {"definition":"At that time; next.","type":"adv","examples":["Then we left.","Finish, then rest."]},

    # prepositions & conjunctions
    "in": {"definition":"Expressing location or position.","type":"prep","examples":["In the room.","Living in the city."]},
    "on": {"definition":"Positioned above and in contact with.","type":"prep","examples":["On the table.","On Monday."]},
    "at": {"definition":"Used for specific times/places.","type":"prep","examples":["At noon.","Meet at the park."]},
    "with": {"definition":"Accompanied by.","type":"prep","examples":["With a friend.","Cut with a knife."]},
    "for": {"definition":"With the purpose of.","type":"prep","examples":["For example.","I did it for you."]},
    "and": {"definition":"Conjunction joining words or phrases.","type":"conj","examples":["Bread and butter.","He and she."]},
    "but": {"definition":"Conjunction showing contrast.","type":"conj","examples":["I like it but...","It was small but useful."]},
    "or": {"definition":"Conjunction indicating alternatives.","type":"conj","examples":["Tea or coffee?","Now or later."]},
    "if": {"definition":"Introducing a conditional clause.","type":"conj","examples":["If it rains, we'll stay.","Ask if needed."]},
    "will": {"definition":"Modal verb indicating future.","type":"modal","examples":["I will go.","She will join."]},
    "can": {"definition":"Modal verb indicating ability or possibility.","type":"modal","examples":["Can you help?","It can work."]},

    # phrases / multiword
    "there is": {"definition":"Used to state existence.","type":"phrase","examples":["There is a way to solve it.","There is a message for you."]},
    "i think": {"definition":"Phrase expressing opinion.","type":"phrase","examples":["I think we should go.","I think it's correct."]},
    "for example": {"definition":"Used to give an example.","type":"phrase","examples":["Many fruits, for example apples, are healthy."]},
    "in order to": {"definition":"With the purpose to.","type":"phrase","examples":["In order to learn, practice is required."]},

    # domain words
    "paris": {"definition":"Capital city of France.","type":"place","examples":["Paris is beautiful in spring.","I visited Paris."]},
    "python": {"definition":"A high-level programming language.","type":"noun","examples":["I wrote the script in Python.","Python is popular."]},
    "gravity": {"definition":"Force attracting objects to one another.","type":"noun","examples":["Gravity keeps us grounded.","Gravity affects motion."]},
    "pi": {"definition":"Mathematical constant ≈ 3.14159.","type":"number","examples":["Pi is used to compute circumference."]},

    # small extra verbs/nouns for transitions
    "learn": {"definition":"Gain knowledge by study.","type":"verb","examples":["We learn from mistakes.","I want to learn more."]},
    "help": {"definition":"Provide assistance.","type":"verb","examples":["Can you help me?","Thanks for the help."]},
    "start": {"definition":"Begin doing something.","type":"verb","examples":["Start now.","We start at nine."]},
    "finish": {"definition":"Bring to an end.","type":"verb","examples":["Finish the task.","He finished quickly."]},
    "read": {"definition":"Look at and understand written words.","type":"verb","examples":["Read the book.","I like to read."]},
    "write": {"definition":"Mark letters to form words.","type":"verb","examples":["Write a note.","She writes code."]},
    "play": {"definition":"Take part in activity for enjoyment.","type":"verb","examples":["Let's play a game.","They play music."]},
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
# Knowledge base for quick facts
# -------------------------
# -------------------------
# Big Knowledge base for quick facts (large)
# Keys are lowercase normalized question phrases
# -------------------------
KB = {
    # History & dates
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
    "when was napoleon defeated": "Napoleon was finally defeated at the Battle of Waterloo in 1815.",

    # Capitals & countries
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
    "which country has the largest population": "China (followed closely by India).",
    "which is the largest country by area": "Russia is the largest country by area.",
    "which continent is brazil in": "Brazil is in South America.",
    "what is the largest ocean": "The Pacific Ocean.",

    # Science & nature
    "what is gravity": "Gravity is the force by which objects with mass attract each other (≈9.81 m/s² near Earth's surface).",
    "what is photosynthesis": "A process by which plants convert light energy into chemical energy, producing oxygen and glucose from CO₂ and water.",
    "what is the largest planet": "Jupiter.",
    "what is the smallest planet": "Mercury (excluding dwarf planets).",
    "what is the sun": "The Sun is a star at the center of the Solar System that supplies light and heat to Earth.",
    "how far is the earth from the sun": "About 1 astronomical unit ≈ 149.6 million kilometers (≈93 million miles).",
    "what is dna": "DNA (deoxyribonucleic acid) stores genetic information in living organisms.",
    "what is rna": "RNA (ribonucleic acid) is a molecule involved in coding, decoding, regulation, and expression of genes.",
    "what is a gene": "A gene is a unit of heredity made of DNA that codes for a protein or functional product.",
    "what is cellular respiration": "A process cells use to convert nutrients into energy (ATP), producing CO₂ and water.",
    "what is an atom": "The smallest unit of ordinary matter, made of protons, neutrons, and electrons.",
    "what is a molecule": "Two or more atoms chemically bonded together.",

    # Math & constants
    "what is pi": "Pi (π) ≈ 3.141592653589793 — the ratio of a circle's circumference to its diameter.",
    "what is e": "Euler's number e ≈ 2.718281828 — the base of natural logarithms.",
    "what is the speed of light": "Approximately 299,792,458 meters per second in vacuum.",
    "what is avogadros number": "Avogadro's number ≈ 6.02214076 × 10^23 (particles per mole).",
    "what is 2 plus 2": "2 + 2 = 4.",
    "what is the square root of 9": "The square root of 9 is 3.",
    "how many degrees in a circle": "360 degrees.",
    "what is a prime number": "A number greater than 1 with no positive divisors other than 1 and itself.",

    # Units & conversions
    "how many centimeters in a meter": "100 centimeters in 1 meter.",
    "how many meters in a kilometer": "1000 meters in 1 kilometer.",
    "how many inches in a foot": "12 inches in 1 foot.",
    "how many feet in a yard": "3 feet in 1 yard.",
    "how many ounces in a pound": "16 ounces in 1 pound (avoirdupois).",
    "convert celsius to fahrenheit": "°F = °C × 9/5 + 32.",
    "convert fahrenheit to celsius": "°C = (°F − 32) × 5/9.",

    # Biology & health (general info, not medical advice)
    "what is a fever": "A fever is a raised body temperature, often a sign of infection. Adults: temps above ~38°C (100.4°F).",
    "what is dehydration": "A condition when the body loses more fluids than it takes in; symptoms include thirst, low urine output, dizziness.",
    "what is a vaccination": "A vaccine stimulates the immune system to develop protection against a disease-causing organism.",
    "what is an antibiotic": "A medicine that destroys or inhibits the growth of bacteria; not effective against viruses.",
    "what is a virus": "A tiny infectious agent that needs a host cell to replicate.",

    # Computers & technology
    "what is python": "Python is a high-level programming language known for readability and wide use in scripting and data science.",
    "what is an api": "An API (Application Programming Interface) allows software systems to communicate and exchange data.",
    "what is machine learning": "A field of AI where models learn patterns from data to make predictions or decisions.",
    "what is a database": "A structured collection of data stored and accessed electronically.",
    "what is cloud computing": "Delivery of computing services (servers, storage, databases, networking) over the internet.",
    "what is github": "A web-based platform for hosting and collaborating on Git repositories (code).",
    "what is linux": "An open-source family of Unix-like operating systems based on the Linux kernel.",

    # Web / internet basics
    "what is http": "HTTP is the Hypertext Transfer Protocol used for transferring web pages on the internet.",
    "what is https": "HTTPS is HTTP over TLS/SSL — it encrypts web traffic for security.",
    "what is a url": "A Uniform Resource Locator — the address used to access resources on the web.",
    "what is dns": "Domain Name System — translates human-readable domain names to IP addresses.",

    # Programming basics
    "what is a variable": "A named storage location that holds a value in programming.",
    "what is a function": "A reusable block of code that performs a specific task.",
    "what is a loop": "A programming construct that repeats a block of code while a condition holds true.",
    "what is recursion": "When a function calls itself to solve smaller instances of a problem.",

    # Literature & arts
    "who wrote hamlet": "William Shakespeare.",
    "who wrote pride and prejudice": "Jane Austen.",
    "who painted the mona lisa": "Leonardo da Vinci.",
    "what is a sonnet": "A 14-line poem with a specific rhyme scheme, often about love or philosophy.",

    # People & biographies (short)
    "who is elon musk": "Entrepreneur: SpaceX, Tesla, Neuralink, among others (roles may change over time).",
    "who is barack obama": "44th President of the United States (2009–2017).",
    "who is albert einstein": "Physicist known for the theory of relativity and contributions to quantum mechanics.",
    "who discovered america": "Christopher Columbus's 1492 voyage reached the Americas for Europe; indigenous peoples inhabited the continents long before.",

    # Geography & travel
    "what is the highest mountain": "Mount Everest is the highest mountain above sea level (≈8,848 m).",
    "what is the longest river": "The Nile and Amazon are both contenders depending on measurement method; commonly the Nile is cited as the longest.",
    "what currency does the united states use": "United States dollar (USD).",
    "what currency does japan use": "Japanese yen (JPY).",
    "what language is spoken in brazil": "Portuguese is the official language of Brazil.",

    # Food & cooking
    "how do i boil an egg": "Place eggs in boiling water and cook 6–8 minutes for medium, 9–12 minutes for hard; cool in cold water to stop cooking.",
    "how to make coffee": "Brew ground coffee with hot water using your preferred method (drip, French press, espresso), adjusting coffee-to-water ratio to taste.",
    "what is baking soda": "Sodium bicarbonate, a leavening agent used in baking.",
    "what is baking powder": "A mixture that contains baking soda and acids to create chemical leavening when moistened and heated.",

    # Practical 'how-to' & everyday tasks
    "how to tie a tie": "A common method is the four-in-hand knot: wrap the wide end over the narrow, loop, bring through and tighten. Many illustrated tutorials online.",
    "how to reset a password": "Use the service's 'forgot password' link to receive reset instructions via email or SMS; follow provider-specific steps.",
    "how to take a screenshot": "On Windows: PrtScn or Win+Shift+S; macOS: Cmd+Shift+3 or Cmd+Shift+4; many phones use power+volume buttons.",
    "how to unzip a file": "On many systems you can right-click → Extract; or use command-line tools like unzip (Linux/macOS) or Expand-Archive (PowerShell).",

    # Sports & entertainment
    "how many players in a soccer team": "11 players on the field per team in association football (soccer).",
    "how many players in basketball": "5 players on the court per team in basketball.",
    "what is the super bowl": "The NFL's annual championship game in American football (United States).",

    # Law & civics (general)
    "how to register to vote": "Voter registration procedures vary by country and state — check your local election authority for requirements and deadlines.",
    "what is a constitution": "A set of fundamental principles or established precedents according to which a state is governed.",

    # Business & finance
    "what is inflation": "A general rise in prices and fall in purchasing power of money.",
    "what is a stock": "A share of ownership in a company.",
    "what is interest rate": "The percentage charged on a loan or paid on savings, typically expressed annually.",

    # Short definitions
    "definition of algorithm": "A step-by-step procedure for solving a problem or performing a task.",
    "definition of computer": "An electronic device that processes data according to programmed instructions.",
    "definition of democracy": "A system of government by the whole population, typically through elected representatives.",

    # Quick facts & trivia
    "what is the boiling point of water": "100 °C (212 °F) at standard atmospheric pressure (sea level).",
    "what is the freezing point of water": "0 °C (32 °F) at standard atmospheric pressure.",
    "how many seconds in a minute": "60 seconds.",
    "how many minutes in an hour": "60 minutes.",
    "how many hours in a day": "24 hours.",
    "how many days in a year": "365 days (366 in a leap year).",

    # Science history & discoveries
    "who proposed the theory of relativity": "Albert Einstein proposed the theory of relativity.",
    "who discovered electricity": "Electric phenomena were studied over centuries; Benjamin Franklin, Alessandro Volta, Michael Faraday, and others made key discoveries.",

    # Computing concepts
    "what is encryption": "The process of encoding information so only authorized parties can read it.",
    "what is a firewall": "A system that monitors and controls incoming and outgoing network traffic based on security rules.",
    "what is a virus (computer)": "Malicious software that can replicate and damage systems or data.",

    # Education & institutions
    "what is mit": "Massachusetts Institute of Technology (MIT), a research university in Cambridge, Massachusetts.",
    "what is harvard": "Harvard University, an Ivy League research university in Cambridge, Massachusetts.",

    # Weather & climate
    "what causes rain": "Condensed water vapor in clouds forms droplets that fall as precipitation when heavy enough.",
    "what is climate change": "Long-term changes in average weather patterns, including global warming driven largely by greenhouse gas emissions.",

    # Astronomy
    "what is a black hole": "A region of spacetime with gravity so strong that nothing, not even light, can escape from it.",
    "what is a galaxy": "A system of stars, gas, dust, and dark matter bound together by gravity (e.g., the Milky Way).",
    "how many planets in the solar system": "There are eight planets: Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, Neptune.",

    # Language & grammar
    "what is a noun": "A word that names a person, place, thing, or idea.",
    "what is a verb": "A word that expresses an action, occurrence, or state of being.",
    "what is an adjective": "A word that modifies or describes a noun.",

    # Travel & transport
    "what is an airport": "A facility where aircraft take off and land; includes runways, terminals, and support services.",
    "how to get a passport": "Apply through your country's passport authority; requirements typically include ID, photos, and fees.",

    # Safety & emergencies
    "what to do in a fire": "Get out quickly, stay low to avoid smoke, call emergency services when safe, and follow evacuation plans.",
    "what to do for a heart attack": "Call emergency services immediately; if trained, begin CPR if the person is unresponsive and not breathing normally.",

    # Consumer & shopping
    "how to return an item": "Check the seller's return policy, keep the receipt, and follow the provider's return/exchange instructions.",
    "how to check a warranty": "Review the product documentation or the manufacturer's website for warranty terms and claim process.",

    # Arts & culture
    "what is jazz": "A music genre that originated in African-American communities, known for improvisation and syncopated rhythms.",
    "what is classical music": "Art music rooted in Western traditions, spanning from the medieval era to the contemporary period.",

    # Misc short useful facts
    "what is the internet": "A global network connecting computers and other devices for data exchange.",
    "what is email": "Electronic mail — messages sent over the internet between users.",
    "how to send an email": "Use an email client or webmail, compose a message, enter recipient address, subject, body, and press send.",

    # Localized or alternate phrasings to help matching
    "who was the first president of the u s a": "George Washington (1789–1797).",
    "who wrote hamlet play": "William Shakespeare.",
    "what is the capital of france": "Paris.",
    "what is the capital of the united states": "Washington, D.C.",

    # Short fallback hints
    "who invented the telephone": "Alexander Graham Bell is often credited, though others contributed to telephone-like inventions.",
    "who invented the light bulb": "Thomas Edison improved and commercialized electric light; others like Humphry Davy and Joseph Swan made earlier contributions.",

    # Add more if needed...
}


# -------------------------
# Tokenization & vocab (cached)
# -------------------------
WORD_RE = re.compile(r"[a-zA-Z']+")

def tokenize(text: str) -> List[str]:
    return WORD_RE.findall((text or "").lower())

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
# TinyNN (small & fast)
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
# Markov generator with light grammar preferences (drop-in replacement)
# -------------------------
class Markov:
    def __init__(self):
        # map: (w1,w2) -> { next_word: count, ... }
        self.map = {}
        # list of starting bigrams observed
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

    def _best_choice(self, choices):
        if not choices:
            return None
        best = max(sorted(choices.items()), key=lambda kv: kv[1])
        return best[0]

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
        return source_vocab | dict_tokens

    def _is_real_word(self, tok, allowed_set):
        if not re.fullmatch(r"[a-zA-Z']+", tok):
            return False
        if len(tok) == 1 and tok.lower() not in ("a","i"):
            return False
        if allowed_set and tok not in allowed_set:
            return False
        return True

    def _token_type(self, tok):
        """Return a best-guess 'type' for token from merged_dictionary or simple heuristics."""
        md = merged_dictionary()
        # direct token match in multi-word keys or single-word keys
        tok_l = tok.lower()
        # check single-word dictionary entries first
        for k,v in md.items():
            if k == tok_l:
                return v.get("type","")
        # token may appear inside multiword key (prefer that)
        for k,v in md.items():
            if tok_l in tokenize(k):
                return v.get("type","")
        # heuristics: articles/prepositions/conjunctions/modals
        if tok_l in ("a","an","the"):
            return "article"
        if tok_l in ("and","or","but"):
            return "conj"
        if tok_l in ("in","on","at","with","for","to","from","by","about"):
            return "prep"
        if tok_l in ("is","are","was","were","be","am","been","being","have","has","had","do","does","did","will","can","may","should","would","could"):
            return "verb"
        # short-word guess: single-letter i -> pronoun
        if tok_l == "i":
            return "pronoun"
        return ""

    def _pos_score(self, prev_tok, prev_prev_tok, candidate):
        """
        Lightweight POS preference:
         - pronoun/noun -> prefer verb or modal next
         - article -> prefer adjective or noun
         - adjective -> prefer noun or adverb
         - verb -> prefer adverb or noun or pronoun (object)
         - preposition -> prefer determiner/adjective/noun/pronoun
         - conjunction -> prefer noun/pronoun/verb
        Returns a small bonus to add to raw counts.
        """
        cand_type = self._token_type(candidate)
        prev_type = self._token_type(prev_tok) if prev_tok else ""
        prevprev_type = self._token_type(prev_prev_tok) if prev_prev_tok else ""

        # default no bonus
        bonus = 0.0

        # pronoun or noun tends to be followed by verb/modal or preposition
        if prev_type in ("pronoun","noun","place","person") or prevprev_type in ("pronoun","noun"):
            if cand_type in ("verb","modal","prep"):
                bonus += 2.0
        # articles prefer adjectives and nouns
        if prev_type == "article":
            if cand_type in ("adj","noun","place","person","phrase"):
                bonus += 2.5
        # adjectives prefer nouns or adverbs
        if prev_type == "adj":
            if cand_type in ("noun","place","person","adv"):
                bonus += 1.8
        # verbs often followed by nouns/adverbs/pronouns
        if prev_type == "verb" or prev_type in ("modal","aux"):
            if cand_type in ("noun","pronoun","adv","adj","phrase"):
                bonus += 1.5
        # prepositions prefer determiners/pronouns/nouns/adjectives
        if prev_type == "prep":
            if cand_type in ("article","pronoun","noun","adj","phrase"):
                bonus += 2.2
        # conjunctions often connect clauses: prefer pronoun/noun/verb
        if prev_type == "conj":
            if cand_type in ("pronoun","noun","verb","phrase"):
                bonus += 1.6
        # phrase tokens (like "i think", "there is") often followed by phrase continuation
        if prev_type == "phrase":
            if cand_type in ("verb","noun","adj","adv","phrase"):
                bonus += 1.6

        # small preference for known dictionary types (non-empty)
        if cand_type:
            bonus += 0.1

        return bonus

    def _rank_candidates(self, key, prev_key):
        """Return list of (candidate,score) sorted by score, using counts + pos preference + real-word filter."""
        choices = self.map.get(key, {})
        if not choices:
            return []
        allowed = self._valid_tokens_set()
        scored = []
        for w,count in choices.items():
            # prefer real words; if not real, skip normally
            if not self._is_real_word(w, allowed):
                continue
            # pos bonus uses last token and the previous token from key
            prev_tok = key[1]
            prev_prev_tok = key[0]
            bonus = self._pos_score(prev_tok, prev_prev_tok, w)
            score = count + bonus
            scored.append((w, score))
        # fallback: if we filtered everything out, allow unfiltered best choices but still rank by raw count
        if not scored:
            for w,count in choices.items():
                score = count
                scored.append((w, score))
        scored.sort(key=lambda kv: kv[1], reverse=True)
        return scored

    def _finalize_sentence(self, words):
        """Post-process: capitalize first word, fix spacing/punctuation, append period if needed."""
        if not words:
            return ""
        s = " ".join(words)
        # remove space before punctuation
        s = re.sub(r"\s+([,\.\?!;:])", r"\1", s)
        # collapse multiple spaces
        s = re.sub(r"\s{2,}", " ", s).strip()
        # capitalize first character
        s = s[0].upper() + s[1:] if s else s
        # ensure sentence ends with punctuation
        if not re.search(r"[\.!\?]$", s):
            s = s + "."
        return s

    def generate(self, seed=None, max_words=40):
        """
        Grammar-aware deterministic greedy generator:
         - If seed has >=2 tokens and we have a matching bigram, greedily pick the top-ranked candidate
           by (count + POS preference) and continue.
         - Otherwise generate from a random start, but still apply POS preferences.
         - Post-process to fix capitalization and punctuation.
        """
        # continuation mode when seed given
        if seed:
            toks = tokenize(seed)
            if len(toks) >= 2:
                key = (toks[-2].lower(), toks[-1].lower())
                continuation = []
                for _ in range(max_words):
                    ranked = self._rank_candidates(key, None)
                    if not ranked:
                        # backoff: try any unigram following prev token
                        # build synthetic choices from keys that start with key[1]
                        candidates = {}
                        for (a,b), nxts in self.map.items():
                            if a == key[1]:
                                for w,cnt in nxts.items():
                                    candidates[w] = candidates.get(w,0) + cnt
                        if not candidates:
                            break
                        # convert to ranked list with pos preference
                        temp_key = (key[1], "__BACKOFF__")
                        # monkeypatch map temporarily to reuse ranking function logic:
                        self.map[temp_key] = candidates
                        ranked = self._rank_candidates(temp_key, None)
                        self.map.pop(temp_key, None)
                        if not ranked:
                            break
                    nxt = ranked[0][0]
                    # safety: avoid short-token loops
                    if len(continuation) >= 2 and continuation[-1] == nxt and len(nxt) <= 2:
                        break
                    continuation.append(nxt)
                    key = (key[1], nxt)
                if continuation:
                    # finalize grammar for continuation alone (capitalize first word if it's a sentence)
                    return self._finalize_sentence(continuation)
        # No seed or can't continue: build full sentence from a start bigram
        if not self.starts:
            return ""
        key = random.choice(self.starts)
        out = [key[0], key[1]]
        for _ in range(max_words-2):
            ranked = self._rank_candidates((out[-2], out[-1]), None)
            if not ranked:
                break
            nxt = ranked[0][0]
            out.append(nxt)
            # avoid trivial repeating loops
            if len(out) >= 5 and out[-1] == out[-2] == out[-3]:
                break
            # stop early if punctuation token appended (rare, but helps)
            if re.fullmatch(r"[\.!\?;,]", nxt):
                break
        return self._finalize_sentence(out)

# instantiate
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
    """Full rebuild of Markov from dictionary + conversations (expensive-ish)."""
    MARKOV.map.clear(); MARKOV.starts.clear()
    md = merged_dictionary()
    for k,v in md.items():
        for ex in v.get("examples", []):
            MARKOV.train(ex)
        MARKOV.train(k + " " + v.get("definition",""))
    for c in ai_state.get("conversations", []):
        MARKOV.train(c.get("text",""))
    # persist to disk
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
# Build & train TinyNN model (on-demand)
# -------------------------
VOCAB: List[str] = []
NN_MODEL: Optional[TinyNN] = None

def build_and_train_model(force: bool=False):
    """Expensive: build vocab and train TinyNN. Call only when needed."""
    global VOCAB, NN_MODEL
    VOCAB = build_vocab(force=force)
    hidden_dim = max(24, len(VOCAB)//12 or 16)
    NN_MODEL = TinyNN(len(VOCAB), hidden_dim, len(INTENTS))
    dataset = build_training(VOCAB)
    if dataset:
        NN_MODEL.train(dataset, epochs=20, lr=0.06)
    ai_state["model_dirty"] = False
    save_json(STATE_FILE, ai_state)

# build a tiny model at startup but let user rebuild fully if desired
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

    # Markov generative fallback
    gen = MARKOV.generate(seed=user, max_words=50)
    if gen:
        if gen.strip():
            if user and user.strip()[-1] in ".!?":
                reply_text = gen.capitalize() + "."
            else:
                reply_text = (user.rstrip() + " " + gen).strip()
            return {"reply": reply_text, "meta":{"intent":"gen"}}

    return {"reply": "I don't know that yet. Teach me with 'X means Y' or '/define X: Y'.", "meta":{"intent":"unknown"}}

# -------------------------
# File ingestion (txt/json) with incremental Markov update
# -------------------------
def ingest_text_content(name: str, text: str, save_as_memory: bool=True):
    """Add uploaded text into learned memory or conversations. If save_as_memory True, store under learned."""
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
    # persist markov quickly
    try:
        serial = {"starts": MARKOV.starts, "map": markov_serialize(MARKOV.map)}
        save_json(MARKOV_FILE, serial)
    except Exception:
        pass
    return f"Ingested {added} blocks from {name}."

# -------------------------
# UI: Streamlit Chat & Controls
# -------------------------
st.set_page_config(page_title="Jack — Offline AI (Persistent Memory)", layout="wide")
st.title("Jack — Offline AI (Persistent Memory) — Combined")

left, right = st.columns([3,1])

with right:
    st.header("Memory & Model Controls")
    if st.button("Clear Conversation"):
        ai_state["conversations"].clear()
        save_json(STATE_FILE, ai_state)
        st.success("Conversation cleared.")
        st.rerun()
    if st.button("Forget Learned Memories"):
        ai_state["learned"].clear()
        save_json(STATE_FILE, ai_state)
        ai_state["model_dirty"] = True
        save_json(STATE_FILE, ai_state)
        st.success("All learned memories forgotten.")
        st.rerun()

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
                    ai_state["learned"].pop(k, None)
                    save_json(STATE_FILE, ai_state)
                    ai_state["model_dirty"] = True
                    save_json(STATE_FILE, ai_state)
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
                ai_state["model_dirty"] = True
                save_json(STATE_FILE, ai_state)
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
            ai_state["model_dirty"] = True
            save_json(STATE_FILE, ai_state)
            st.rerun()
    if c2.button("Complete"):
        ui = user_input.rstrip()
        if ui:
            cont = MARKOV.generate(seed=ui, max_words=40)
            if cont:
                if ui and ui.strip()[-1] in ".!?":
                    final = ui.rstrip() + " " + cont.capitalize()
                else:
                    final = (ui + " " + cont).strip()
                ai_state.setdefault("conversations", []).append({"role":"user","text":ui,"time":datetime.now().isoformat()})
                ai_state.setdefault("conversations", []).append({"role":"assistant","text":final,"time":datetime.now().isoformat()})
                MARKOV.train(ui); MARKOV.train(final)
                save_json(STATE_FILE, ai_state)
                ai_state["model_dirty"] = True
                save_json(STATE_FILE, ai_state)
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
            ai_state["model_dirty"] = True
            save_json(STATE_FILE, ai_state)
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
