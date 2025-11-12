# -*- coding: utf-8 -*-
import streamlit as st
import json
from datetime import datetime
import os

# ---------------------------
# Memory & Data Handling
# ---------------------------
MEMORY_FILE = "ai_memory.json"
if os.path.exists(MEMORY_FILE):
    with open(MEMORY_FILE, "r") as f:
        ai_memory = json.load(f)
else:
    ai_memory = {"conversations": [], "learned_words": {}}

# ---------------------------
# Dictionary (sample, expandable to 10k words)
# ---------------------------
dictionary = {
    "apple": {"definition": "A round fruit with red or green skin and a whitish interior.", "type": "noun", "examples": ["I ate an apple for lunch.", "Apple pie is delicious."]},
    "run": {"definition": "To move swiftly on foot.", "type": "verb", "examples": ["I run every morning.", "She runs faster than me."]},
    "happy": {"definition": "Feeling or showing pleasure or contentment.", "type": "adjective", "examples": ["I am happy today.", "He felt happy after hearing the news."]},
    "python": {"definition": "A high-level programming language.", "type": "noun", "examples": ["Python is easy to learn.", "I wrote a script in Python."]},
    "book": {"definition": "A set of written or printed pages bound together.", "type": "noun", "examples": ["I read a book last night.", "The library has many books."]},
    "jump": {"definition": "Push oneself off a surface into the air by using the muscles in the legs and feet.", "type": "verb", "examples": ["The cat can jump high.", "He jumped over the fence."]},
    "sad": {"definition": "Feeling or showing sorrow; unhappy.", "type": "adjective", "examples": ["She was sad after the movie.", "It made me sad to hear the news."]},
    "computer": {"definition": "An electronic device for storing and processing data.", "type": "noun", "examples": ["I bought a new computer.", "The computer is slow today."]},
    "learn": {"definition": "Gain knowledge or skill by studying, practicing, or being taught.", "type": "verb", "examples": ["I want to learn French.", "She learned to play the piano."]},
    "teacher": {"definition": "A person who teaches, especially in a school.", "type": "noun", "examples": ["The teacher explained the lesson.", "My teacher is very kind."]},
    "beautiful": {"definition": "Pleasing the senses or mind aesthetically.", "type": "adjective", "examples": ["The sunset is beautiful.", "She has a beautiful voice."]},
    "eat": {"definition": "Put food into the mouth and chew and swallow it.", "type": "verb", "examples": ["I eat breakfast every day.", "He eats quickly."]},
    "water": {"definition": "A clear, colorless, odorless, and tasteless liquid essential for life.", "type": "noun", "examples": ["I drink water every day.", "The water is cold."]},
    "swim": {"definition": "Propel the body through water by using the limbs.", "type": "verb", "examples": ["We swim in the pool.", "Fish can swim very fast."]},
    "angry": {"definition": "Feeling or showing strong annoyance or displeasure.", "type": "adjective", "examples": ["I am angry at the traffic.", "He was angry with me."]},
    "music": {"definition": "Vocal or instrumental sounds combined to produce harmony, melody, or rhythm.", "type": "noun", "examples": ["I listen to music every day.", "She loves classical music."]},
    "write": {"definition": "Mark letters, words, or symbols on a surface with a pen, pencil, or other instrument.", "type": "verb", "examples": ["I write in my notebook.", "She writes stories every week."]},
    "fast": {"definition": "Moving or capable of moving at high speed.", "type": "adjective", "examples": ["The car is very fast.", "He runs fast."]},
    "house": {"definition": "A building for human habitation.", "type": "noun", "examples": ["I live in a house.", "The house is big."]},
    "think": {"definition": "Have a particular opinion, belief, or idea about someone or something.", "type": "verb", "examples": ["I think it will rain today.", "She thinks about her future."]},
    "smart": {"definition": "Having or showing a quick-witted intelligence.", "type": "adjective", "examples": ["The AI is very smart.", "He is smart and talented."]},
    "dog": {"definition": "A domesticated carnivorous mammal.", "type": "noun", "examples": ["I have a dog.", "The dog barked loudly."]},
    "run": {"definition": "To move swiftly on foot.", "type": "verb", "examples": ["I run every morning.", "She runs faster than me."]},
    "sleep": {"definition": "A condition of body and mind which typically recurs for several hours every night, in which the nervous system is relatively inactive.", "type": "verb", "examples": ["I sleep eight hours a night.", "She is sleeping now."]},
    "cold": {"definition": "Of or at a low or relatively low temperature.", "type": "adjective", "examples": ["It is cold outside.", "The water is cold."]},
}


# ---------------------------
# Helper Functions
# ---------------------------
def save_memory():
    with open(MEMORY_FILE, "w") as f:
        json.dump(ai_memory, f, indent=2)

def define_word(word):
    word = word.lower()
    if word in dictionary:
        return f"{word.capitalize()} ({dictionary[word]['type']}): {dictionary[word]['definition']}"
    elif word in ai_memory["learned_words"]:
        return f"{word.capitalize()} (learned): {ai_memory['learned_words'][word]}"
    else:
        return "Word not found in dictionary."

def explain_phrase(phrase):
    words = phrase.split()
    explanations = [define_word(w) for w in words]
    return "\n".join(explanations)

def calculate(expr):
    try:
        return eval(expr, {"__builtins__": {}})
    except Exception as e:
        return f"Error: {str(e)}"

def get_time():
    return datetime.now().strftime("%H:%M:%S")

def get_date():
    return datetime.now().strftime("%Y-%m-%d")

def learn_word(word, definition):
    ai_memory["learned_words"][word.lower()] = definition
    save_memory()
    return f"Learned '{word}': {definition}"

def clear_memory():
    ai_memory["conversations"] = []
    ai_memory["learned_words"] = {}
    save_memory()
    return "Memory cleared!"

# ---------------------------
# Streamlit Interface
# ---------------------------
st.title("Offline AI Assistant")
st.write("Type your command below:")

user_input = st.text_input("You:", "")

if user_input:
    user_input_lower = user_input.lower()

    # Commands
    if user_input_lower.startswith("define "):
        word = user_input[7:].strip()
        response = define_word(word)
    elif user_input_lower.startswith("explain "):
        phrase = user_input[8:].strip()
        response = explain_phrase(phrase)
    elif user_input_lower.startswith("calc "):
        expr = user_input[5:].strip()
        response = calculate(expr)
    elif user_input_lower == "time":
        response = get_time()
    elif user_input_lower == "date":
        response = get_date()
    elif user_input_lower.startswith("learn "):
        try:
            _, rest = user_input.split(" ", 1)
            word, definition = rest.split(":", 1)
            response = learn_word(word.strip(), definition.strip())
        except:
            response = "Use format: learn <word>: <definition>"
    elif user_input_lower == "clear memory":
        response = clear_memory()
    else:
        # Fallback: remember conversation
        response = f"I don't know how to do that yet. You said: {user_input}"
        ai_memory["conversations"].append({"user": user_input})
        save_memory()

    st.text_area("AI:", value=response, height=200)
