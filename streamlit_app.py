# jack_offline_fast_start.py
# Faster startup: lazy-load big dictionary and on-demand model rebuild.
# Run:
#   pip install streamlit
#   streamlit run jack_offline_fast_start.py

import streamlit as st
import json, os, re, math, random, uuid, threading
from datetime import datetime
from typing import List, Dict, Tuple, Any, Optional

# Cache of already-fetched topics to avoid re-downloading repeatedly
_fetched_topics_cache = set()

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
     # --------------------
    # Countries & Capitals (common set)
    # --------------------
    "capital of afghanistan": "Kabul",
    "capital of albania": "Tirana",
    "capital of algeria": "Algiers",
    "capital of andorra": "Andorra la Vella",
    "capital of angola": "Luanda",
    "capital of argentina": "Buenos Aires",
    "capital of armenia": "Yerevan",
    "capital of australia": "Canberra",
    "capital of austria": "Vienna",
    "capital of azerbaijan": "Baku",
    "capital of bahamas": "Nassau",
    "capital of bahrain": "Manama",
    "capital of bangladesh": "Dhaka",
    "capital of barbados": "Bridgetown",
    "capital of belarus": "Minsk",
    "capital of belgium": "Brussels",
    "capital of belize": "Belmopan",
    "capital of benin": "Porto-Novo",
    "capital of bhutan": "Thimphu",
    "capital of bolivia": "Sucre (constitutional), La Paz (administrative)",
    "capital of bosnia and herzegovina": "Sarajevo",
    "capital of botswana": "Gaborone",
    "capital of brazil": "Brasília",
    "capital of brunei": "Bandar Seri Begawan",
    "capital of bulgaria": "Sofia",
    "capital of burkina faso": "Ouagadougou",
    "capital of burundi": "Gitega",
    "capital of cambodia": "Phnom Penh",
    "capital of cameroon": "Yaoundé",
    "capital of canada": "Ottawa",
    "capital of cape verde": "Praia",
    "capital of central african republic": "Bangui",
    "capital of chad": "N'Djamena",
    "capital of chile": "Santiago",
    "capital of china": "Beijing",
    "capital of colombia": "Bogotá",
    "capital of comoros": "Moroni",
    "capital of costa rica": "San José",
    "capital of cote d'ivoire": "Yamoussoukro (official), Abidjan (economic)",
    "capital of croatia": "Zagreb",
    "capital of cuba": "Havana",
    "capital of cyprus": "Nicosia",
    "capital of czech republic": "Prague",
    "capital of democratic republic of the congo": "Kinshasa",
    "capital of denmark": "Copenhagen",
    "capital of djibouti": "Djibouti",
    "capital of dominica": "Roseau",
    "capital of dominican republic": "Santo Domingo",
    "capital of ecuador": "Quito",
    "capital of egypt": "Cairo",
    "capital of el salvador": "San Salvador",
    "capital of equatorial guinea": "Malabo (planned: Oyala / Ciudad de la Paz)",
    "capital of eritrea": "Asmara",
    "capital of estonia": "Tallinn",
    "capital of eswatini": "Mbabane (administrative), Lobamba (royal/legislative)",
    "capital of ethiopia": "Addis Ababa",
    "capital of fiji": "Suva",
    "capital of finland": "Helsinki",
    "capital of france": "Paris",
    "capital of gabon": "Libreville",
    "capital of gambia": "Banjul",
    "capital of georgia": "Tbilisi",
    "capital of germany": "Berlin",
    "capital of ghana": "Accra",
    "capital of greece": "Athens",
    "capital of grenada": "St. George's",
    "capital of guatemala": "Guatemala City",
    "capital of guinea": "Conakry",
    "capital of guinea-bissau": "Bissau",
    "capital of guyana": "Georgetown",
    "capital of haiti": "Port-au-Prince",
    "capital of honduras": "Tegucigalpa",
    "capital of hungary": "Budapest",
    "capital of iceland": "Reykjavík",
    "capital of india": "New Delhi",
    "capital of indonesia": "Jakarta",
    "capital of iran": "Tehran",
    "capital of iraq": "Baghdad",
    "capital of ireland": "Dublin",
    "capital of israel": "Jerusalem (disputed; many embassies in Tel Aviv)",
    "capital of italy": "Rome",
    "capital of jamaica": "Kingston",
    "capital of japan": "Tokyo",
    "capital of jordan": "Amman",
    "capital of kazakhstan": "Astana (Nur-Sultan formerly)",
    "capital of kenya": "Nairobi",
    "capital of kiribati": "South Tarawa",
    "capital of kosovo": "Pristina",
    "capital of kuwait": "Kuwait City",
    "capital of kyrgyzstan": "Bishkek",
    "capital of laos": "Vientiane",
    "capital of latvia": "Riga",
    "capital of lebanon": "Beirut",
    "capital of lesotho": "Maseru",
    "capital of liberia": "Monrovia",
    "capital of libya": "Tripoli",
    "capital of liechtenstein": "Vaduz",
    "capital of lithuania": "Vilnius",
    "capital of luxembourg": "Luxembourg (city)",
    "capital of madagascar": "Antananarivo",
    "capital of malawi": "Lilongwe",
    "capital of malaysia": "Kuala Lumpur (official), Putrajaya (administrative)",
    "capital of maldives": "Malé",
    "capital of mali": "Bamako",
    "capital of malta": "Valletta",
    "capital of marshall islands": "Majuro",
    "capital of mauritania": "Nouakchott",
    "capital of mauritius": "Port Louis",
    "capital of mexico": "Mexico City",
    "capital of micronesia": "Palikir",
    "capital of moldova": "Chișinău",
    "capital of monaco": "Monaco",
    "capital of mongolia": "Ulaanbaatar",
    "capital of montenegro": "Podgorica",
    "capital of morocco": "Rabat",
    "capital of mozambique": "Maputo",
    "capital of myanmar": "Naypyidaw",
    "capital of namibia": "Windhoek",
    "capital of nauru": "Yaren (de facto)",
    "capital of nepal": "Kathmandu",
    "capital of netherlands": "Amsterdam (constitutional), The Hague (seat of government)",
    "capital of new zealand": "Wellington",
    "capital of nicaragua": "Managua",
    "capital of niger": "Niamey",
    "capital of nigeria": "Abuja",
    "capital of north korea": "Pyongyang",
    "capital of north macedonia": "Skopje",
    "capital of norway": "Oslo",
    "capital of oman": "Muscat",
    "capital of pakistan": "Islamabad",
    "capital of palau": "Ngerulmud",
    "capital of panama": "Panama City",
    "capital of papua new guinea": "Port Moresby",
    "capital of paraguay": "Asunción",
    "capital of peru": "Lima",
    "capital of philippines": "Manila",
    "capital of poland": "Warsaw",
    "capital of portugal": "Lisbon",
    "capital of qatar": "Doha",
    "capital of romania": "Bucharest",
    "capital of russia": "Moscow",
    "capital of rwanda": "Kigali",
    "capital of saint lucia": "Castries",
    "capital of samoa": "Apia",
    "capital of san marino": "San Marino",
    "capital of sao tome and principe": "São Tomé",
    "capital of saudi arabia": "Riyadh",
    "capital of senegal": "Dakar",
    "capital of serbia": "Belgrade",
    "capital of seychelles": "Victoria",
    "capital of sierra leone": "Freetown",
    "capital of singapore": "Singapore",
    "capital of slovakia": "Bratislava",
    "capital of slovenia": "Ljubljana",
    "capital of solomon islands": "Honiara",
    "capital of somalia": "Mogadishu",
    "capital of south africa": "Pretoria (administrative), Cape Town (legislative), Bloemfontein (judicial)",
    "capital of south korea": "Seoul",
    "capital of south sudan": "Juba",
    "capital of spain": "Madrid",
    "capital of sri lanka": "Sri Jayawardenepura Kotte (official), Colombo (commercial)",
    "capital of sudan": "Khartoum",
    "capital of suriname": "Paramaribo",
    "capital of sweden": "Stockholm",
    "capital of switzerland": "Bern",
    "capital of syria": "Damascus",
    "capital of taiwan": "Taipei",
    "capital of tajikistan": "Dushanbe",
    "capital of tanzania": "Dodoma (official), Dar es Salaam (largest city)",
    "capital of thailand": "Bangkok",
    "capital of togo": "Lomé",
    "capital of tonga": "Nukuʻalofa",
    "capital of trinidad and tobago": "Port of Spain",
    "capital of tunisia": "Tunis",
    "capital of turkey": "Ankara",
    "capital of turkmenistan": "Ashgabat",
    "capital of tuvalu": "Funafuti",
    "capital of uganda": "Kampala",
    "capital of ukraine": "Kyiv",
    "capital of united arab emirates": "Abu Dhabi",
    "capital of united kingdom": "London",
    "capital of united states": "Washington, D.C.",
    "capital of uruguay": "Montevideo",
    "capital of uzbekistan": "Tashkent",
    "capital of vanuatu": "Port Vila",
    "capital of venezuela": "Caracas",
    "capital of vietnam": "Hanoi",
    "capital of yemen": "Sana'a (constitutional), Aden (temporary seat declared)",
    "capital of zambia": "Lusaka",
    "capital of zimbabwe": "Harare",

    # --------------------
    # Planets & solar system
    # --------------------
    "how many planets are in the solar system": "Eight (Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, Neptune).",
    "what are the planets": "Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, Neptune.",
    "largest planet": "Jupiter is the largest planet in the solar system.",
    "smallest planet": "Mercury is the smallest planet in the solar system (among the eight).",
    "what planet is known as the red planet": "Mars is called the Red Planet.",
    "which planet has rings": "Saturn is best known for its rings; Jupiter, Uranus, and Neptune also have ring systems.",
    "what planet is closest to the sun": "Mercury is the closest planet to the Sun.",
    "what planet is farthest from the sun": "Neptune is the farthest recognized planet from the Sun (Pluto is a dwarf planet).",
    "what is earth's moon called": "The Moon (sometimes called Luna).",
    "what is the axis tilt of earth": "Earth's axial tilt is about 23.5 degrees.",

    # --------------------
    # Foods (common items)
    # --------------------
    "apple": "a sweet, edible fruit from the apple tree.",
    "banana": "a long curved fruit with soft pulpy flesh.",
    "orange": "a citrus fruit rich in vitamin C.",
    "bread": "a baked food made from flour and water.",
    "rice": "a staple grain consumed worldwide.",
    "pasta": "an Italian noodle made from durum wheat.",
    "pizza": "a flatbread topped with tomato sauce, cheese, and various toppings.",
    "cheese": "a dairy product made by coagulating milk proteins.",
    "milk": "a nutrient-rich liquid produced by mammals.",
    "egg": "an oval reproductive body laid by birds used as food.",
    "chicken": "poultry meat commonly eaten worldwide.",
    "beef": "meat from cattle.",
    "fish": "edible aquatic vertebrates.",
    "salad": "a dish of mixed vegetables (sometimes fruits) often served cold.",
    "sandwich": "food consisting of fillings between slices of bread.",
    "tomato": "a red edible fruit used as a vegetable in cooking.",
    "potato": "a starchy tuber eaten worldwide.",
    "onion": "a pungent bulbous vegetable used for flavoring.",
    "garlic": "a pungent aromatic used widely in cooking.",
    "butter": "a dairy product made by churning cream.",
    "yogurt": "a fermented milk product.",
    "coffee": "a brewed drink prepared from roasted coffee beans.",
    "tea": "a beverage made by steeping cured leaves.",
    "sugar": "a sweet crystalline carbohydrate used as sweetener.",
    "salt": "a mineral (sodium chloride) used for seasoning and preservation.",
    "pepper": "a spice (black pepper) used for seasoning.",
    "lettuce": "a leafy vegetable commonly used in salads.",
    "carrot": "a root vegetable, usually orange.",
    "onigiri": "Japanese rice ball, often wrapped in seaweed with a filling.",
    "avocado": "a creamy green fruit often used in salads and spreads.",
    "mango": "a tropical stone fruit with sweet flesh.",
    "strawberry": "a red aggregate fruit with seeds on its exterior.",
    "blueberry": "a small sweet blue-purple berry.",
    "grape": "a small round fruit used for eating and wine-making.",
    "olive": "a small fruit used for oil and as table olives.",
    "salmon": "a fatty fish often eaten grilled or smoked.",
    "tuna": "a large saltwater fish commonly used in sushi and cans.",
    "shrimp": "small crustacean eaten worldwide.",
    "lobster": "a large marine crustacean served as seafood.",
    "tofu": "a soy-based protein, common in East Asian cuisines.",
    "noodles": "long, thin strands of dough used in many cuisines.",
    "cereal": "processed grain usually eaten for breakfast.",
    "oats": "a cereal grain used for porridge and baking.",
    "almond": "an edible tree nut.",
    "walnut": "a tree nut with brain-like kernel.",
    "peanut": "a legume often called a nut; used for butter and snacks.",
    "honey": "a sweet viscous food made by bees.",
    "olive oil": "oil extracted from olives used in cooking and dressings.",
    "vinegar": "an acidic liquid used as condiment and preservative.",
    "mayonnaise": "emulsion of oil, egg yolk and acid used as a spread.",
    "ketchup": "a sweet-savory tomato-based condiment.",
    "mustard": "a condiment made from mustard seeds.",
    "soy sauce": "a salty fermented sauce used in Asian cooking.",
    "salsa": "a spicy sauce used as a dip or condiment.",
    "hummus": "a spread made from mashed chickpeas and tahini.",
    "pancake": "a flat cake cooked on a hot surface, often eaten for breakfast.",
    "waffle": "a leavened batter cooked between two hot plates to form a patterned surface.",
    "croissant": "a buttery, flaky, viennoiserie pastry.",
    "bagel": "a dense bread roll formed into a ring.",
    "brownie": "a dense chocolate baked dessert.",
    "cake": "a sweet baked dessert, often frosted for celebrations.",
    "pie": "a baked dish with filling enclosed in pastry.",
    "soup": "a primarily liquid food, typically served hot.",
    "stew": "a dish cooked slowly in liquid, often with meat and vegetables.",
    "curry": "a dish with a spiced sauce common in South Asian cuisines.",
    "barbecue": "food cooked or flavored by smoke and grilling.",
    "burger": "a ground meat patty served in a bun with toppings.",
    "fries": "deep-fried strips of potato.",
    "ice cream": "a frozen sweet dairy dessert.",
    "gelato": "an Italian style ice cream, denser and often lower in fat.",
    "smoothie": "blended drink made from fruits and often yogurt or milk.",

    # --------------------
    # Simple recipes / how-to (short)
    # --------------------
    "how to boil an egg": "Place egg in boiling water for 6–10 minutes depending on desired firmness; cool in cold water.",
    "how to cook rice": "Rinse rice, add water (approx. 1:1.5 rice:water by volume), simmer covered until water absorbed (~15–20 min), fluff with fork.",
    "how to make scrambled eggs": "Whisk eggs, salt, butter in a pan over medium heat, stir until softly set.",
    "how to make coffee": "Brew ground coffee with hot water using drip, pour-over, French press, or espresso method.",
    "how to make tea": "Steep tea leaves in near-boiling water 2–5 minutes depending on type, strain and serve.",
    "how to make salad": "Chop greens and vegetables, toss with dressing, season to taste.",
    "how to grill a steak": "Season steak, sear over high heat to desired doneness, rest 5 minutes before slicing.",
    "how to bake bread": "Mix flour, water, yeast, salt, knead, let rise, shape, proof, and bake until golden.",
    "how to make pancakes": "Mix flour, milk, eggs; pour batter on hot griddle, flip when bubbles form, cook until done.",
    "how to make sandwich": "Place fillings between two slices of bread, cut and serve.",
    "how to make pasta": "Boil pasta in salted water until al dente; drain and combine with sauce.",
    "how to make soup": "Sauté aromatics, add stock and main ingredients, simmer until flavors meld.",
    "how to roast vegetables": "Toss vegetables with oil and salt, roast at 200°C (400°F) until tender and caramelized.",
    "how to make guacamole": "Mash avocados, mix with lime juice, salt, chopped onion, cilantro, and tomato.",
    "how to make hummus": "Blend cooked chickpeas, tahini, lemon juice, garlic, olive oil until smooth.",
    "how to make omelette": "Beat eggs, pour into hot buttered pan, cook until set, fold with fillings if desired.",
    "how to make pancakes from scratch": "Combine flour, sugar, baking powder, milk, eggs, melted butter; cook on griddle.",

    # --------------------
    # Famous people (short descriptors)
    # --------------------
    "who is albert einstein": "A theoretical physicist known for the theory of relativity (1879–1955).",
    "who is isaac newton": "A mathematician and physicist who formulated classical mechanics and gravity (1643–1727).",
    "who is marie curie": "A physicist and chemist who pioneered research on radioactivity; two-time Nobel laureate.",
    "who is william shakespeare": "An English playwright and poet from the late 16th/early 17th century.",
    "who is leonardo da vinci": "An Italian polymath: painter, scientist, and engineer (1452–1519).",
    "who is nelson mandela": "South African anti-apartheid leader and president (1918–2013).",
    "who is martin luther king jr": "A leader of the American civil rights movement (1929–1968).",
    "who is mother teresa": "A Catholic nun and missionary known for charitable work in Kolkata (1910–1997).",
    "who is mahatma gandhi": "Indian independence leader who promoted nonviolent resistance (1869–1948).",
    "who is ada lovelace": "Often considered the first computer programmer (1815–1852).",
    "who is alan turing": "A mathematician and computer science pioneer; codebreaker in WWII (1912–1954).",
    "who is rosa parks": "Civil rights activist whose refusal to give up her bus seat became a symbol of the movement.",
    "who is mother jones": "A labor and community organizer known as Mother Jones (1837–1930).",
    "who is malala yousafzai": "Education activist and youngest Nobel Prize laureate (born 1997).",
    "who is steve jobs": "Co-founder of Apple Inc., technology entrepreneur (1955–2011).",
    "who is bill gates": "Co-founder of Microsoft and philanthropist (born 1955).",
    "who is elon musk": "Entrepreneur behind Tesla, SpaceX, and other ventures (born 1971).",
    "who is oprah winfrey": "Talk show host, media executive, and philanthropist.",
    "who is beyonce": "A popular American singer, songwriter, and performer.",
    "who is taylor swift": "An American singer-songwriter known for pop and country-influenced music.",

    # --------------------
    # General facts & units
    # --------------------
    "what is pi": "Pi (π) ≈ 3.141592653589793, the ratio of a circle's circumference to its diameter.",
    "how many seconds in a minute": "60 seconds.",
    "how many minutes in an hour": "60 minutes.",
    "how many hours in a day": "24 hours.",
    "how many days in a year": "365 days in a common year; 366 in a leap year.",
    "what is the speed of light": "Approximately 299,792,458 meters per second in vacuum.",
    "what is avogadro's number": "Approximately 6.02214076 × 10^23 (particles per mole).",
    "what is the boiling point of water celsius": "100°C at standard atmospheric pressure (1 atm).",
    "what is absolute zero": "0 Kelvin (−273.15°C), theoretically the lowest possible temperature.",
    "what is dna": "Deoxyribonucleic acid, molecule carrying genetic instructions in living organisms.",
    "what is an atom": "The smallest unit of a chemical element that retains its chemical properties.",
    "what is photosynthesis": "A process where plants convert sunlight, CO₂ and water into glucose and oxygen.",
    "what is climate change": "Long-term shifts in temperatures and weather patterns, often driven by human activity.",
    "what is renewable energy": "Energy from sources that are naturally replenished, like solar, wind, hydro.",
    "what is socialism": "An economic system where the means of production are owned or regulated by the community.",
    "what is democracy": "A system of government by the whole population or eligible members, typically via elected representatives.",
    "what is the human heart": "A muscular organ that pumps blood through the circulatory system.",
    "what is the largest ocean": "The Pacific Ocean is the largest ocean on Earth.",
    "what is the tallest mountain": "Mount Everest is the Earth's highest mountain above sea level (~8,848 m).",
    "what is the largest desert": "The Sahara is the largest hot desert; Antarctica is the largest desert overall.",

    # --------------------
    # Dates & history (short)
    # --------------------
    "when did world war i start": "1914",
    "when did world war i end": "1918",
    "when did world war ii start": "1939",
    "when did world war ii end": "1945",
    "when was the united nations founded": "1945",
    "when did man first walk on the moon": "1969 (Apollo 11).",

    # --------------------
    # Technology & computing
    # --------------------
    "what is an api": "Application Programming Interface — a way for programs to communicate.",
    "what is html": "HyperText Markup Language, used for structuring web pages.",
    "what is css": "Cascading Style Sheets, used for styling web pages.",
    "what is javascript": "A programming language commonly used to build interactive websites.",
    "what is machine learning": "Algorithms that enable systems to learn patterns from data.",
    "what is artificial intelligence": "Simulation of human intelligence by machines.",
    "what does cpu stand for": "Central Processing Unit — the primary component that executes instructions in a computer.",
    "what does gpu stand for": "Graphics Processing Unit — specialized processor for graphics and parallel tasks.",

    # --------------------
    # Health & biology
    # --------------------
    "what is an antibody": "A protein produced by the immune system to neutralize pathogens.",
    "what is a vaccine": "A biological preparation that provides immunity to a specific disease.",
    "what is diabetes": "A group of diseases that affect how the body uses blood sugar (glucose).",
    "what is hypertension": "High blood pressure; a chronic medical condition.",
    "what is cholesterol": "A waxy substance in blood; high levels can increase heart disease risk.",

    # --------------------
    # Economics & finance (basic)
    # --------------------
    "what is inflation": "A general increase in prices and fall in the purchasing power of money.",
    "what is gdp": "Gross Domestic Product — the total value of goods and services produced in a country.",
    "what is a stock": "A share of ownership in a company.",

    # --------------------
    # Misc short Q/A (useful triggers)
    # --------------------
    "how to tie a tie": "Cross the wide end over the narrow, loop, and knot (Windsor/Knot styles vary).",
    "how to change a tire": "Loosen lug nuts, lift vehicle with jack, remove wheel, replace with spare, tighten nuts.",
    "how to reset a router": "Unplug power for 10–30 seconds, plug back in, wait for reboot.",
    "how to take a screenshot": "Use OS-specific shortcuts (e.g., Print Screen on Windows, Cmd+Shift+4 on macOS).",
    "how to measure a room": "Measure length and width with a tape measure; multiply for area.",
    "what is the best way to learn a language": "Practice regularly, study vocabulary and grammar, speak with native speakers.",
    "what is the largest mammal": "The blue whale is the largest mammal.",
    "what is the fastest land animal": "The cheetah is the fastest land animal over short distances.",
    "what is renewable energy example": "Solar panels convert sunlight into electricity.",
    "what is the unit of force": "Newton (N).",
    "what is the unit of energy": "Joule (J).",
    "what is the unit of power": "Watt (W).",
    "what does html stand for": "HyperText Markup Language.",
    "what does url stand for": "Uniform Resource Locator.",
    "what is bitcoin": "A decentralized digital currency based on blockchain technology.",
    "what is a blockchain": "A distributed ledger that records transactions in cryptographic blocks.",
    "what is the ozone layer": "A region of Earth's stratosphere containing ozone that absorbs much of the sun's UV radiation.",

    # --------------------
    # Short FAQ-like entries
    # --------------------
    "what should i do if my computer is slow": "Restart, check for malware, remove unused programs, free disk space, consider hardware upgrades.",
    "how to backup data": "Use external drives, cloud storage, or automated backup software to copy important files regularly.",
    "how to improve sleep": "Keep a consistent schedule, reduce screen time before bed, create a dark comfortable environment.",
    "what is a balanced diet": "A variety of foods providing adequate macro- and micronutrients: vegetables, fruits, grains, protein, dairy or alternatives.",
    "how to stay hydrated": "Drink water regularly, especially when active or in hot weather; monitor urine color as rough guide.",

    # --------------------
    # Travel & geography
    # --------------------
    "what is the largest city in the united states": "New York City (by population).",
    "what is europe's largest country by area": "Russia (partly in Europe), if excluding Russia then Ukraine (or France depending on measure).",
    "what is the longest river": "The Nile and Amazon are among the longest; sources and measurement methods vary.",

    # --------------------
    # Education & study
    # --------------------
    "how to write an essay": "Plan a structure (intro, body, conclusion), draft, revise, and proofread.",
    "how to prepare for an exam": "Review notes, practice problems, make a study schedule, get rest before the test.",

    # --------------------
    # Short troubleshooting/help
    # --------------------
    "why won't my phone charge": "Check cable and charger, try different outlet, inspect charging port for debris, try different charger.",
    "why is my internet slow": "Check router, restart devices, test speed, consider congestion or ISP issues.",

    # --------------------
    # Placeholder facts (expandable)
    # --------------------
    "who was the first president of the united states": "George Washington (1789–1797).",
    "what is the capital of texas": "Austin.",
    "what is the capital of california": "Sacramento.",
    "how many continents are there": "Seven commonly listed continents: Africa, Antarctica, Asia, Europe, North America, Oceania (Australia), South America.",
    "what is the currency of the united kingdom": "Pound sterling (GBP).",
    "what is the currency of the european union": "The euro (EUR) for most member states; some member states use other currencies.",
    "what is the tallest building": "As of many recent records, the Burj Khalifa in Dubai is the tallest building (828 m).",

    # --------------------
    # Short math helpers
    # --------------------
    "what is gcd": "Greatest Common Divisor — the largest integer dividing two numbers without remainder.",
    "what is lcm": "Least Common Multiple — the smallest positive integer that is a multiple of two numbers.",
    "what is factorial": "Product of all positive integers up to n, written n! (e.g., 5! = 120).",

    # --------------------
    # Short language / dictionary-style entries
    # --------------------
    "define gravity": "Gravity: a natural phenomenon by which objects with mass attract one another.",
    "define democracy": "Democracy: government by the people, typically via elected representatives.",
    "define algorithm": "A step-by-step procedure for calculations or problem-solving.",

    # --------------------
    # Safety / health quick pointers
    # --------------------
    "what to do for minor burn": "Cool under running water for 10–20 minutes, cover with sterile dressing; seek care if severe.",
    "what to do for sprain": "RICE: Rest, Ice, Compression, Elevation; seek medical attention if severe.",

    # --------------------
    # Short entertainment / culture facts
    # --------------------
    "who wrote 'to kill a mockingbird'": "Harper Lee.",
    "who painted the mona lisa": "Leonardo da Vinci.",

    # --------------------
    # End of KB_BIG (add more entries as needed)
    # --------------------
}

# ---------- Manual-style big dictionary generator ----------
def generate_manual_dictionary(min_entries: int = 2000) -> Dict[str, Dict[str,Any]]:
    """
    Build a large, high-quality dictionary from curated seeds + safe morphological variants.
    Returns a dict suitable to merge into BASE_DICT.
    Use only when user requests (to avoid startup lag).
    """
    def examples_for(word, typ):
        w = word.replace("_", " ")
        if typ == "noun":
            return [f"the {w} is on the table.", f"i saw a {w} yesterday."]
        if typ == "verb":
            return [f"please {w} carefully.", f"i {w} every day."]
        if typ == "adj":
            return [f"that is very {w}.", f"the {w} example."]
        if typ == "food":
            return [f"i like {w}.", f"{w} is delicious."]
        if typ == "place":
            return [f"i visited {w}.", f"{w} is beautiful."]
        return [f"{w} example."]

    def make_plural(w):
        if w.endswith(("s","x","z","ch","sh")): return w + "es"
        if w.endswith("y") and len(w)>1 and w[-2] not in "aeiou": return w[:-1] + "ies"
        return w + "s"

    def make_ing(w):
        if w.endswith("ie"): return w[:-2] + "ying"
        if w.endswith("e") and not w.endswith("ee"): return w[:-1] + "ing"
        if len(w)>=3 and (w[-1] not in "aeiou" and w[-2] in "aeiou" and w[-3] not in "aeiou"): return w + w[-1] + "ing"
        return w + "ing"

    # curated seed lists (real, meaningful words)
    foods = """apple banana orange bread rice pasta pizza cheese milk egg chicken beef fish shrimp salmon tuna pork lamb bacon tofu yogurt butter honey avocado mango strawberry blueberry grape pear peach plum coconut cherry watermelon cantaloupe kiwi lemon lime grapefruit onion garlic potato tomato carrot cucumber lettuce spinach kale broccoli cauliflower mushroom zucchini eggplant pepper chili corn beans lentils oats quinoa barley rye cereal bagel pancake waffle croissant donut brownie cake pie icecream gelato sorbet toast sandwich burger fries steak tacos burrito sushi sashimi ramen udon pho curry stew soup salad""".split()
    verbs = """eat drink cook bake boil fry chop slice mix stir serve taste walk run jump drive read write play learn teach think make get take give find see watch listen create build open close start stop continue choose buy sell order deliver order prepare slice dice grill roast sear simmer blend whisk beat fold knead pour measure mix toss marinate""".split()
    adjectives = """good new old big small large tiny quick slow happy sad bright dark warm cold spicy sweet sour bitter fresh frozen ripe raw cooked crunchy creamy soft hard tender juicy moist dry salty oily""".split()
    household = """table chair bed sofa lamp fridge stove oven microwave toaster kettle sink bathtub shower mirror curtain blanket pillow towel rug shelf closet drawer""".split()
    body = """head face eye ear nose mouth neck shoulder arm hand finger thumb wrist elbow chest stomach back hip leg knee ankle foot toe""".split()
    tech = """computer laptop server router modem smartphone tablet api html css javascript python java c++ rust go typescript node git docker kubernetes linux windows macos database sql nosql json xml api endpoint encryption ssl ai machine learning neural network model algorithm data training inference""".split()
    places = """paris london rome berlin madrid tokyo beijing washington ottawa canberra moscow delhi beirut cairo bangkok singapore dubai zurich amsterdam lisbon brussels vienna athens prague budapest oslo stockholm helsinki dublin""".split()
    people = """einstein newton curie shakespeare da_vinci mandela gandhi rosa_parks martin_luther_king ada_lovelace alan_turing jobs gates musk oprah beyonce taylor_swift madonna bob_dylan""" .split()
    numbers = """one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty thirty forty fifty sixty seventy eighty ninety hundred thousand million billion""".split()
    daily = """morning afternoon evening night today tomorrow yesterday week month year monday tuesday wednesday thursday friday saturday sunday""".split()
    connectors = """and or but because however therefore although since while if when where why how also moreover instead besides finally""".split()

    # start building dictionary
    D = {}
    def add_word(w, typ="noun", definition=None, examples=None):
        key = w.lower().replace(" ", "_")
        if key in D: return
        if not definition:
            # short human-like definitions by type
            if typ == "food":
                definition = f"a food item: {w.replace('_',' ')}"
            elif typ == "verb":
                definition = f"to {w.replace('_',' ')}"
            elif typ == "adj":
                definition = f"describes something that is {w.replace('_',' ')}"
            elif typ == "place":
                definition = f"a place: {w.replace('_',' ')}"
            elif typ == "person":
                definition = f"{w.replace('_',' ').title()}, a notable person (placeholder)."
            else:
                definition = f"{w.replace('_',' ')} (noun)"
        if examples is None:
            examples = examples_for(w.replace("_"," "), typ)
        D[key] = {"definition": definition, "type": typ, "examples": examples}

    # add curated sets
    for f in foods:
        add_word(f, typ="food", definition=f"A food commonly called {f}.", examples=[f"I like {f}.", f"{f} is delicious."])
    for v in verbs:
        add_word(v, typ="verb", definition=f"to {v.replace('_',' ')}", examples=[f"please {v}.", f"i {v} every day."])
        # add -ing and -ed forms
        ing = make_ing(v)
        add_word(ing, typ="verb", definition=f"present participle of {v}", examples=[f"i am {ing}."])
        ed = v + "ed" if not v.endswith("e") else v + "d"
        add_word(ed, typ="verb", definition=f"past tense of {v}", examples=[f"i {ed} yesterday."])
    for a in adjectives:
        add_word(a, typ="adj", definition=f"{a} (adjective).", examples=[f"that is very {a}."])
    for h in household:
        add_word(h, typ="noun", definition=f"{h} (household item).", examples=[f"the {h} is in the room."])
    for b in body:
        add_word(b, typ="noun", definition=f"{b} (body part).", examples=[f"my {b} hurts."])
    for t in tech:
        add_word(t, typ="tech", definition=f"{t} (technology/term).", examples=[f"i used {t} today."])
    for p in places:
        add_word(p, typ="place", definition=f"{p.replace('_',' ').title()}, a city/place.", examples=[f"I visited {p}."])
    for n in numbers:
        add_word(n, typ="number", definition=f"{n} (number word).", examples=[f"{n} is a number."])
    for c in connectors:
        add_word(c, typ="conjunction", definition=f"{c} (connector word).", examples=[f"this is {c} an example."])
    for pe in people:
        add_word(pe, typ="person", definition=f"{pe.replace('_',' ').title()}, notable person (short).", examples=[f"{pe.replace('_',' ').title()} was notable."])
    for d in daily:
        add_word(d, typ="time", definition=f"{d} (time-related).", examples=[f"{d} is a day name."])

    # add capital cities (select larger list manually)
    capitals = {
        "afghanistan":"kabul","albania":"tirana","algeria":"algiers","andorra":"andorra_la_vella","angola":"luanda",
        "argentina":"buenos_aires","armenia":"yerevan","australia":"canberra","austria":"vienna","azerbaijan":"baku",
        "bahamas":"nassau","bahrain":"manama","bangladesh":"dhaka","barbados":"bridgetown","belarus":"minsk",
        "belgium":"brussels","belize":"belmopan","benin":"porto-novo","bhutan":"thimphu","bolivia":"sucre",
        "bosnia_and_herzegovina":"sarajevo","botswana":"gaborone","brazil":"brasilia","brunei":"bandar_seri_begawan",
        "bulgaria":"sofia","burkina_faso":"ouagadougou","burundi":"gitega","cambodia":"phnom_penh","cameroon":"yaounde",
        "canada":"ottawa","cape_verde":"praia","central_african_republic":"bangui","chad":"n_djamena","chile":"santiago",
        "china":"beijing","colombia":"bogota","comoros":"moroni","costa_rica":"san_jose","croatia":"zagreb","cuba":"havana",
        "cyprus":"nicosia","czech_republic":"prague","democratic_republic_of_the_congo":"kinshasa","denmark":"copenhagen",
        "djibouti":"djibouti","dominica":"roseau","dominican_republic":"santo_domingo","ecuador":"quito","egypt":"cairo",
        "el_salvador":"san_salvador","equatorial_guinea":"malabo","eritrea":"asmara","estonia":"tallinn","eswatini":"mbabane",
        "ethiopia":"addis_ababa","fiji":"suva","finland":"helsinki","france":"paris","gabon":"libreville","gambia":"banjul",
        "georgia":"tbilisi","germany":"berlin","ghana":"accra","greece":"athens","grenada":"st_georges","guatemala":"guatemala_city",
        "guinea":"conakry","guinea_bissau":"bissau","guyana":"georgetown","haiti":"port_au_prince","honduras":"tegucigalpa",
        "hungary":"budapest","iceland":"reykjavik","india":"new_delhi","indonesia":"jakarta","iran":"tehran","iraq":"baghdad",
        "ireland":"dublin","israel":"jerusalem","italy":"rome","jamaica":"kingston","japan":"tokyo","jordan":"amman",
        "kazakhstan":"astana","kenya":"nairobi","kiribati":"south_tarawa","kosovo":"pristina","kuwait":"kuwait_city",
        "kyrgyzstan":"bishkek","laos":"vientiane","latvia":"riga","lebanon":"beirut","lesotho":"maseru","liberia":"monrovia",
        "libya":"tripoli","liechtenstein":"vaduz","lithuania":"vilnius","luxembourg":"luxembourg","madagascar":"antananarivo",
        "malawi":"lilongwe","malaysia":"kuala_lumpur","maldives":"male","mali":"bamako","malta":"valletta","marshall_islands":"majuro",
        "mauritania":"nouakchott","mauritius":"port_louis","mexico":"mexico_city","micronesia":"palikir","moldova":"chisinau",
        "monaco":"monaco","mongolia":"ulaanbaatar","montenegro":"podgorica","morocco":"rabat","mozambique":"maputo",
        "myanmar":"naypyidaw","namibia":"windhoek","nauru":"yaren","nepal":"kathmandu","netherlands":"amsterdam",
        "new_zealand":"wellington","nicaragua":"managua","niger":"niamey","nigeria":"abuja","north_korea":"pyongyang",
        "north_macedonia":"skopje","norway":"oslo","oman":"muscat","pakistan":"islamabad","palau":"ngerulmud",
        "panama":"panama_city","papua_new_guinea":"port_moresby","paraguay":"asuncion","peru":"lima","philippines":"manila",
        "poland":"warsaw","portugal":"lisbon","qatar":"doha","romania":"bucharest","russia":"moscow","rwanda":"kigali",
        "saint_lucia":"castries","samoa":"apia","san_marino":"san_marino","sao_tome_and_principe":"sao_tome","saudi_arabia":"riyadh",
        "senegal":"dakar","serbia":"belgrade","seychelles":"victoria","sierra_leone":"freetown","singapore":"singapore_city",
        "slovakia":"bratislava","slovenia":"ljubljana","solomon_islands":"honiara","somalia":"mogadishu","south_africa":"pretoria",
        "south_korea":"seoul","south_sudan":"juba","spain":"madrid","sri_lanka":"colombo","sudan":"khartoum","suriname":"paramaribo",
        "sweden":"stockholm","switzerland":"bern","syria":"damascus","taiwan":"taipei","tajikistan":"dushanbe","tanzania":"dodoma",
        "thailand":"bangkok","togo":"lome","tonga":"nukualofa","trinidad_and_tobago":"port_of_spain","tunisia":"tunis","turkey":"ankara",
        "turkmenistan":"ashgabat","tuvalu":"funafuti","uganda":"kampala","ukraine":"kyiv","united_arab_emirates":"abu_dhabi",
        "united_kingdom":"london","united_states":"washington_dc","uruguay":"montevideo","uzbekistan":"tashkent","vanuatu":"port_vila",
        "venezuela":"caracas","vietnam":"hanoi","yemen":"sana_a","zambia":"lusaka","zimbabwe":"harare"
    }
    for country,cap in capitals.items():
        add_word(country, typ="place", definition=f"{country.replace('_',' ').title()} (country).", examples=[f"I traveled to {country.replace('_',' ')}."])
        add_word(cap, typ="place", definition=f"{cap.replace('_',' ').title()} (capital city).", examples=[f"{cap.replace('_',' ').title()} is the capital."])

    # combine seeds to reach size by adding morphological variants of curated words
    keys_list = list(D.keys())
    idx = 0
    while len(D) < min_entries and idx < len(keys_list)*10:
        base = keys_list[idx % len(keys_list)]
        idx += 1
        # try to add plural or -ing or small derived forms
        if D[base]["type"] in ("noun","food","place","person"):
            p = make_plural(base)
            add_word(p, typ=D[base]["type"], definition=f"plural of {base.replace('_',' ')}", examples=[f"the {p.replace('_',' ')} are here."])
        if D[base]["type"] == "verb":
            ing = make_ing(base)
            add_word(ing, typ="verb", definition=f"{ing} (present participle)", examples=[f"i am {ing.replace('_',' ')}."])
            ed = base + "ed" if not base.endswith("e") else base + "d"
            add_word(ed, typ="verb", definition=f"{ed} (past tense)", examples=[f"i {ed.replace('_',' ')} yesterday."])
        # also add some compound nouns from adjective+noun
        if D[base]["type"] in ("adj","noun"):
            # choose a random noun to pair
            other = random.choice(list(D.keys()))
            if D[other]["type"] in ("noun","food","place"):
                comp = base + "_" + other
                add_word(comp, typ="noun", definition=f"a {base.replace('_',' ')} {other.replace('_',' ')}", examples=[f"the {comp.replace('_',' ')} was useful."])
        # refresh keys list occasionally
        if idx % 500 == 0:
            keys_list = list(D.keys())
    # final safety: ensure at least min_entries by adding synthetic readable fillers
    cntr = 1
    while len(D) < min_entries:
        key = f"term_manual_{cntr}"
        if key not in D:
            D[key] = {"definition": f"manual filler term {cntr}", "type":"auto", "examples":[f"{key.replace('_',' ')} example."]}
        cntr += 1

    return D

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

# -------------------------
# Fetch & train Markov from the web using sklearn TF-IDF
# -------------------------
import requests
from bs4 import BeautifulSoup

def fetch_and_train_markov(topic: str, limit: int = 5, type):
    """
    Retrieve text snippets about a topic from the web and train the Markov model.
    Uses Wikipedia as the data source and sklearn for sentence ranking.

    Auto-skips topics already fetched (cached in _fetched_topics_cache).
    """
    try:
        global _fetched_topics_cache
        topic = topic.strip().lower()
        if not topic:
            return "⚠️ Empty topic provided."
        if topic in _fetched_topics_cache:
            return f"(cached) Markov already trained on '{topic}'."

        if TfidfVectorizer is None:
            return "❌ scikit-learn not available. Install it with 'pip install scikit-learn'."

        # Wikipedia fetch with browser headers
        if type == "wiki":
             url = f"https://en.wikipedia.org/wiki/{topic.replace(' ', '_')}"
        else:
             url = f"https://reddit.com/r/{topic.replace(' ', '_')}/"
       
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0 Safari/537.36"
            )
        }
        r = requests.get(url, headers=headers, timeout=10)
        if r.status_code != 200:
            return f"⚠️ Couldn't retrieve content for '{topic}' (HTTP {r.status_code})."

        # Extract visible text
        soup = BeautifulSoup(r.text, "html.parser")
        paragraphs = [p.get_text() for p in soup.find_all("p")]
        text = " ".join(paragraphs)
        text = re.sub(r"\s+", " ", text).strip()

        sentences = re.split(r"(?<=[.!?])\s+", text)
        if not sentences or len(sentences) < 3:
            return f"No usable text found for '{topic}'."

        vect = TfidfVectorizer(stop_words="english", max_features=5000)
        X = vect.fit_transform(sentences)
        scores = X.sum(axis=1).A.flatten()
        top_indices = scores.argsort()[::-1][:limit]
        top_sentences = [sentences[i] for i in top_indices if len(sentences[i].split()) > 4]

        for s in top_sentences:
            MARKOV.train(s)

        # Remember we’ve seen this topic
        _fetched_topics_cache.add(topic)

        try:
            ser = {"starts": MARKOV.starts,
                   "map": {f"{a}||{b}": nxts for (a,b), nxts in MARKOV.map.items()}}
            save_json(MARKOV_FILE, ser)
        except Exception:
            pass

        return f"✅ Markov trained with {len(top_sentences)} sentences about '{topic}'."

    except Exception as e:
        return f"⚠️ Error during fetch_and_train_markov: {e}"

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
    ("what is photosynthesis", "fact"),
    ("who wrote hamlet", "fact"),
    ("what is pi", "fact"),
    ("what is the capital of france", "fact"),
    ("how many centimeters in a meter", "fact"),
    ("convert 5 miles to kilometers", "math"),
    ("what's the weather like", "chat"),
    ("say hello", "chat"),
    ("define entropy", "define"),
    ("explain photosynthesis", "fact"),
    ("what does dna stand for", "fact"),
    ("calculate 3^4", "math"),
    ("what's 45 divided by 9", "math"),
    ("how many minutes in a day", "fact"),
    ("what year did world war two end", "fact"),
    ("define democracy", "define"),
    ("who is marie curie", "fact"),
    ("what is the speed of light", "fact"),
    ("tell me a joke", "chat"),
    ("how to boil an egg", "fact"),
    ("teach me 'ubiquitous' meaning", "teach"),
    ("ubiquitous means present everywhere", "teach"),
    ("what time is it in tokyo", "time"),
    ("what date is it tomorrow", "date"),
    ("what is the capital of japan", "fact"),
    ("how many ounces in a pound", "fact"),
    ("calculate the area of a circle with radius 3", "math"),
    ("define algorithm", "define"),
    ("explain machine learning simply", "fact"),
    ("who discovered penicillin", "fact"),
    ("what is a black hole", "fact"),
    ("how many days until new year", "date"),
    ("what is the meaning of life", "chat"),
    ("define 'heuristic'", "define"),
    ("42 * 7", "math"),
    ("what's the time in new york", "time"),
    ("when is the next solar eclipse", "fact"),
    ("what is the capital of canada", "fact"),
    ("teach pasta: a type of noodle", "teach"),
    ("pasta means an Italian noodle", "teach"),
    ("what's 2+2", "math"),
    ("calculate 100 / 25", "math"),
    ("define photosynthesis", "define"),
    ("what causes tides", "fact"),
    ("who wrote pride and prejudice", "fact"),
    ("explain relativity in a sentence", "fact"),
    ("what is an atom", "fact"),
    ("how to make a sandwich", "fact"),
    ("teach 'serendipity' means fortunate discovery", "teach"),
    ("what is the capital of italy", "fact"),
    ("what time zone is london in", "fact"),
    ("what's today's date in yyyy-mm-dd", "date"),
    ("define 'metaphor'", "define"),
    ("how many weeks in a year", "fact"),
    ("multiply 17 by 3", "math"),
    ("what is quantum mechanics", "fact"),
    ("who painted the mona lisa", "fact"),
    ("explain blockchain briefly", "fact"),
    ("what is photosynthesis used for", "fact"),
    ("teach carrot means an orange vegetable", "teach"),
    ("carrot means a root vegetable", "teach"),
    ("what's 9 to the power of 3", "math"),
    ("convert 100 celsius to fahrenheit", "math"),
    ("what's the capital of australia", "fact"),
    ("what time is it in sydney", "time"),
    ("define 'symbiosis'", "define"),
    ("tell me a short poem", "chat"),
    ("who invented the telephone", "fact"),
    ("how many planets are in the solar system", "fact"),
    ("is pluto a planet", "fact"),
    ("what is a prime number", "fact"),
    ("calculate gcd of 24 and 36", "math"),
    ("what's the date of independence day in the usa", "date"),
    ("define 'ecosystem'", "define"),
    ("what is the capital of india", "fact"),
    ("teach 'ramen means Japanese noodle soup'", "teach"),
    ("ramen means Japanese noodle soup", "teach"),
    ("how long does it take to boil potatoes", "fact"),
    ("what is an ecosystem", "fact"),
    ("who was nelson mandela", "fact"),
    ("what does 'CPU' stand for", "fact"),
    ("calculate square root of 256", "math"),
    ("what time zone is tokyo", "fact"),
    ("what day of week is 2025-01-01", "date"),
    ("define 'metabolism'", "define"),
    ("how to cook rice", "fact"),
    ("who invented electricity", "fact"),
    ("tell me something interesting", "chat"),
    ("teach 'sushi means vinegared rice with fish'", "teach"),
    ("sushi means vinegared rice with fish", "teach"),
    ("what is the capital of russia", "fact"),
    ("what causes seasons", "fact"),
    ("what is the largest ocean", "fact"),
    ("how many bones in the human body", "fact"),
    ("calculate 7 factorial", "math"),
    ("what is the boiling point of water", "fact"),
    ("define 'gravity'", "define"),
    ("what is an API", "fact"),
    ("who is the current president of the united states", "fact"),
    ("when was the internet invented", "fact"),
    ("how many bytes in a kilobyte", "fact"),
    ("what is 0 divided by 0", "math"),
    ("teach 'avocado means creamy green fruit'", "teach"),
    ("avocado means a creamy green fruit", "teach"),
    ("how to make coffee", "fact"),
    ("what is the capital of brazil", "fact"),
    ("what is photosynthesis equation", "fact"),
    ("who discovered america", "fact"),
    ("tell me a fun fact about space", "chat"),
    ("define 'photosynthesis' in one line", "define"),
    ("what's 123 * 456", "math"),
    ("what time is it in UTC", "time"),
    ("when is daylight saving time", "fact"),
    ("what's the capital of germany", "fact"),
    ("teach 'tofu means soy curd product'", "teach"),
    ("tofu means soy curd product", "teach"),
    ("how to bake bread", "fact"),
    ("what is a continent", "fact"),
    ("who was albert einstein", "fact"),
    ("explain evolution", "fact"),
    ("what's the population of china", "fact"),
    ("what is machine learning", "fact"),
    ("calculate 3.14159 * 2", "math"),
    ("what is the capital of spain", "fact"),
    ("define 'molecule'", "define"),
    ("how to tie a tie", "fact"),
    ("teach 'omelette means beaten eggs cooked in a pan'", "teach"),
    ("omelette means beaten eggs cooked in a pan", "teach"),
    ("what is the capital of mexico", "fact"),
    ("how many teeth does an adult human have", "fact"),
    ("what is the currency of japan", "fact"),
    ("convert 10 usd to eur", "math"),
    ("what is a black hole in simple terms", "fact"),
    ("who wrote the odyssey", "fact"),
    ("define 'photosphere'", "define"),
    ("how far is the moon", "fact"),
    ("what is the capital of china", "fact"),
    ("teach 'bagel means a dense bread ring'", "teach"),
    ("bagel means a dense bread ring", "teach"),
    ("how to make pancakes", "fact"),
    ("what is the difference between a comet and an asteroid", "fact"),
    ("what is an electron", "fact"),
    ("calculate 2^10", "math"),
    ("what date is easter next year", "date"),
    ("who discovered penicillin", "fact"),
    ("what's the capital of south africa", "fact"),
    ("define 'neuron'", "define"),
    ("how many calories in an apple", "fact"),
    ("how to make a salad", "fact"),
    ("teach 'quinoa means a seed used as a grain'", "teach"),
    ("quinoa means a seed used as a grain", "teach"),
    ("what's the square of 15", "math"),
    ("what time does the stock market open", "fact"),
    ("who painted starry night", "fact"),
    ("what is the tallest mountain", "fact"),
    ("calculate the hypotenuse of a right triangle with legs 3 and 4", "math"),
    ("what is the capital of egypt", "fact"),
    ("define 'antibody'", "define"),
    ("how to make tea", "fact"),
    ("who is the author of '1984'", "fact"),
    ("what is the nearest star to earth", "fact"),
    ("teach 'latte means espresso with steamed milk'", "teach"),
    ("latte means espresso with steamed milk", "teach"),
    ("how many liters in a gallon", "math"),
    ("what is the capital of argentina", "fact"),
    ("what's the meaning of 'ephemeral'", "define"),
    ("tell me an inspirational quote", "chat"),
    ("who invented the lightbulb", "fact"),
    ("what is an algorithm", "fact"),
    ("calculate 2 + 2 * 2", "math"),
    ("what day of week is 2025-12-25", "date"),
    ("define 'isotope'", "define"),
    ("how to make soup", "fact"),
    ("what is the capital of portugal", "fact"),
    ("who wrote 'to kill a mockingbird'", "fact"),
    ("what does 'HTTP' stand for", "fact"),
    ("teach 'gazpacho means cold Spanish soup'", "teach"),
    ("gazpacho means cold Spanish soup", "teach"),
    ("how many states are in the usa", "fact"),
    ("what's the escape velocity of earth", "fact"),
    ("define 'covalent bond'", "define"),
    ("calculate 10 factorial", "math"),
    ("what time is sunrise today", "time"),
    ("who discovered electricity", "fact"),
    ("what is the capital of sweden", "fact"),
    ("what causes lightning", "fact"),
    ("teach 'salsa means sauce in Spanish'", "teach"),
    ("salsa means sauce in Spanish", "teach"),
    ("how to roast vegetables", "fact"),
    ("what is the human genome", "fact"),
    ("who wrote 'the great gatsby'", "fact"),
    ("define 'polymer'", "define"),
    ("calculate 0.25 * 4", "math"),
    ("what is the capital of norway", "fact"),
    ("how many moons does jupiter have", "fact"),
    ("what is renewable energy", "fact"),
    ("teach 'espresso means concentrated coffee'", "teach"),
    ("espresso means concentrated coffee", "teach"),
    ("how long to grill chicken", "fact"),
    ("what's the date today", "date"),
    ("what's 5 to the power of 3", "math"),
    ("define 'taxonomy'", "define"),
    ("who is the author of the iliad", "fact"),
    ("what's the capital of poland", "fact"),
    ("how to make guacamole", "fact"),
    ("teach 'sashimi means sliced raw fish'", "teach"),
    ("sashimi means sliced raw fish", "teach"),
    ("what is biodiversity", "fact"),
    ("calculate the perimeter of a rectangle 5 by 8", "math"),
    ("what time is it in berlin", "time"),
    ("who invented the printing press", "fact"),
    ("define 'osmosis'", "define"),
    ("what is the capital of greece", "fact"),
    ("how to ferment yogurt", "fact"),
    ("teach 'bento means Japanese boxed meal'", "teach"),
    ("bento means Japanese boxed meal", "teach")
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
    if not gen or len(gen.split()) < 4:
        # Auto-fetch: if generation failed or seems too short, try web training
        topic_guess = re.sub(r"[^a-zA-Z0-9\s]", "", user).strip().split()
        if topic_guess:
            topic_guess = topic_guess[-1]  # use last word as rough topic
            fetch_result = fetch_and_train_markov(topic_guess, limit=8, "wiki")
            print("[auto-fetch]", fetch_result)
            gen = MARKOV.generate(seed=user, max_words=50)

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
        
        with st.spinner("Generating large dictionary (one-time): this may take a few seconds..."):
             big = generate_manual_dictionary(min_entries=2000)
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
    st.header("Web Markov Trainer")
    topic = st.text_input("Train Markov using Wikipedia topic:", "cats")
    limit = st.slider("Number of sentences to use", 3, 15, 6)
    if st.button("Fetch & Train from Web"):
            with st.spinner(f"Fetching and training on '{topic}'..."):
                result = fetch_and_train_markov(topic, limit=limit, "wiki")
                st.success(result)

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
