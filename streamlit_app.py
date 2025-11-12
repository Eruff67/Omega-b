import streamlit as st
import math
import json
from datetime import datetime

# ---------- AI Memory ----------
try:
    with open("ai_memory.json", "r") as f:
        MEMORY = json.load(f)
except:
    MEMORY = {"history": [], "known_words": {}}

# ---------- English Dictionary ----------
ENGLISH_DICT = {
    "greetings": ["hello", "hi", "hey", "greetings", "good morning", "good afternoon", "good evening", "howdy", "yo"],
    "time": ["time", "current time", "what time is it", "tell me the time", "hour", "clock"],
    "date": ["date", "today's date", "what is the date", "day", "month", "year", "calendar"],
    "thanks": ["thank you", "thanks", "thx", "cheers", "much obliged", "thanks a lot", "thank you very much"],
    "math": ["plus", "add", "+", "minus", "subtract", "-", "times", "multiply", "*", "x",
             "divide", "over", "/", "modulus", "%", "power", "^"],
    "question_words": ["who", "what", "where", "when", "why", "how", "which", "whom", "whose"],
    "common_verbs": ["be", "have", "do", "say", "go", "can", "get", "make", "know", "think", "take", "see", 
                     "come", "want", "look", "use", "find", "give", "tell", "work", "call", "try", "ask", 
                     "need", "feel", "become", "leave", "put", "mean", "keep", "let", "begin", "seem", 
                     "help", "talk", "turn", "start", "show", "hear", "play", "run", "move", "live", 
                     "believe", "bring", "happen", "write", "provide", "sit", "stand", "lose", "pay"],
    "common_nouns": ["time", "person", "year", "way", "day", "thing", "man", "world", "life", "hand", "part", 
                     "child", "eye", "woman", "place", "work", "week", "case", "point", "government", 
                     "company", "number", "group", "problem", "fact"],
    "common_adjectives": ["good", "new", "first", "last", "long", "great", "little", "own", "other", "old", 
                          "right", "big", "high", "different", "small", "large", "next", "early", "young", 
                          "important", "few", "public", "bad", "same", "able"],
    "common_adverbs": ["up", "so", "out", "just", "now", "how", "then", "more", "also", "here", "well", 
                       "only", "very", "even", "back", "there", "down", "still", "in", "as", "to", "when", 
                       "never", "really", "most"],
    "prepositions": ["of", "in", "to", "for", "with", "on", "at", "by", "from", "up", "about", "into", 
                     "over", "after", "beneath", "under", "above"],
    "conjunctions": ["and", "but", "or", "nor", "for", "yet", "so", "although", "because", "since", "unless"],
    "pronouns": ["I", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them", 
                 "my", "your", "his", "her", "its", "our", "their", "mine", "yours", "hers", "ours", "theirs"]
}

# Flatten dictionary into a set for fast lookup
FLATTENED_DICT = set(word for category in ENGLISH_DICT.values() for word in category)

# ---------- Helper Functions ----------
def save_memory():
    with open("ai_memory.json", "w") as f:
        json.dump(MEMORY, f)

def add_to_memory(user_input, ai_response):
    MEMORY["history"].append({"user": user_input, "ai": ai_response})
    save_memory()

def learn_new_words(text):
    words = text.lower().split()
    for word in words:
        if word not in FLATTENED_DICT:
            MEMORY["known_words"][word] = MEMORY["known_words"].get(word, 0) + 1
    save_memory()

def calculate_math(expression):
    try:
        # Very simple math evaluation
        expression = expression.replace("^", "**")
        result = eval(expression, {"__builtins__": None}, math.__dict__)
        return f"The result is {result}"
    except:
        return "I couldn't calculate that."

def understand_text(text):
    text_lower = text.lower()
    
    # Commands
    if text_lower in ["clear memory", "delete memory", "reset"]:
        MEMORY["history"] = []
        MEMORY["known_words"] = {}
        save_memory()
        return "Memory cleared."
    
    if any(word in text_lower for word in ENGLISH_DICT["time"]):
        return f"The current time is {datetime.now().strftime('%H:%M:%S')}"
    
    if any(word in text_lower for word in ENGLISH_DICT["date"]):
        return f"Today's date is {datetime.now().strftime('%Y-%m-%d')}"
    
    if any(word in text_lower for word in ENGLISH_DICT["thanks"]):
        return "You're welcome!"
    
    if any(op in text_lower for op in ["+", "-", "*", "/", "^"]):
        return calculate_math(text_lower)
    
    # Learn new words
    learn_new_words(text_lower)
    
    # Generic response
    return "I understood: " + text

# ---------- Streamlit Interface ----------
st.set_page_config(page_title="Advanced Offline AI", page_icon="🤖", layout="wide")
st.title("Advanced Offline AI 🤖")
st.write("Type something below and the AI will respond. Commands: `clear memory` to reset memory.")

user_input = st.text_input("You:", "")

if user_input:
    ai_response = understand_text(user_input)
    add_to_memory(user_input, ai_response)
    st.write(f"AI: {ai_response}")

# Display conversation history
if MEMORY["history"]:
    st.subheader("Conversation History")
    for entry in MEMORY["history"][-20:]:
        st.write(f"**You:** {entry['user']}")
        st.write(f"**AI:** {entry['ai']}")
