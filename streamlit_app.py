# offline_ai.py
import streamlit as st
import datetime

# ----------------------
# Dictionary & Memory
# ----------------------
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
    "sleep": {"definition": "A condition of body and mind which typically recurs for several hours every night, in which the nervous system is relatively inactive.", "type": "verb", "examples": ["I sleep eight hours a night.", "She is sleeping now."]},
    "cold": {"definition": "Of or at a low or relatively low temperature.", "type": "adjective", "examples": ["It is cold outside.", "The water is cold."]},
    # ... continue to add up to 1000 words
}

memory = []

# ----------------------
# Helper Functions
# ----------------------
def define_word(word):
    word_lower = word.lower()
    if word_lower in dictionary:
        entry = dictionary[word_lower]
        return f"{word} ({entry['type']}): {entry['definition']}\nExamples: {', '.join(entry['examples'])}"
    else:
        return f"I don't have a definition for '{word}'."

def math_solver(expr):
    try:
        return eval(expr)
    except:
        return "Invalid math expression."

def sentence_completion(text):
    # Very simple rule-based completion
    return text + " ...and then something happened."

def remember(text):
    memory.append(text)
    return f"Remembered: {text}"

def show_memory():
    if memory:
        return "\n".join([f"{i+1}. {m}" for i, m in enumerate(memory)])
    else:
        return "Memory is empty."

def clear_memory():
    memory.clear()
    return "Memory cleared."

def delete_memory_item(index):
    try:
        removed = memory.pop(index)
        return f"Removed memory item: {removed}"
    except:
        return "Invalid index."

def get_time():
    return datetime.datetime.now().strftime("%H:%M:%S")

def get_date():
    return datetime.datetime.now().strftime("%Y-%m-%d")

# ----------------------
# Streamlit UI
# ----------------------
st.title("Offline AI — Fully Functional")

user_input = st.text_input("Ask me something:")

if user_input:
    response = ""
    if user_input.startswith("/define "):
        word = user_input.replace("/define ", "")
        response = define_word(word)
    elif user_input.startswith("/math "):
        expr = user_input.replace("/math ", "")
        response = math_solver(expr)
    elif user_input.startswith("/remember "):
        text = user_input.replace("/remember ", "")
        response = remember(text)
    elif user_input.startswith("/memory"):
        response = show_memory()
    elif user_input.startswith("/clear"):
        response = clear_memory()
    elif user_input.startswith("/delete "):
        try:
            index = int(user_input.replace("/delete ", "")) - 1
            response = delete_memory_item(index)
        except:
            response = "Provide a valid number after /delete"
    elif user_input.startswith("/time"):
        response = get_time()
    elif user_input.startswith("/date"):
        response = get_date()
    elif user_input.startswith("/complete "):
        text = user_input.replace("/complete ", "")
        response = sentence_completion(text)
    else:
        response = "I can define words (/define), do math (/math), remember things (/remember), show memory (/memory), clear memory (/clear), delete memory (/delete #), give time (/time), date (/date), and complete sentences (/complete)."

    st.text_area("AI Response:", value=response, height=200)

