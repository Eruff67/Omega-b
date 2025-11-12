import os
import json
import re
import math
from datetime import datetime

# ----- CONFIG -----
MEMORY_FILE = "memory.json"
KNOWLEDGE_DIR = "knowledge"  # Folder where text files are stored for AI to learn from
LANGUAGES = ["english"]  # You can expand with "spanish", "french", etc.

# ----- MEMORY HANDLING -----
if os.path.exists(MEMORY_FILE):
    with open(MEMORY_FILE, "r", encoding="utf-8") as f:
        memory = json.load(f)
else:
    memory = {"conversations": [], "knowledge": {}}

def save_memory():
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(memory, f, ensure_ascii=False, indent=2)

# ----- TEXT PROCESSING -----
def tokenize(text):
    """Lowercase and split text into words, remove non-alphanumeric."""
    return re.findall(r'\b\w+\b', text.lower())

def cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two term frequency vectors."""
    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[x] * vec2[x] for x in intersection])
    sum1 = sum([v**2 for v in vec1.values()])
    sum2 = sum([v**2 for v in vec2.values()])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)
    if not denominator:
        return 0.0
    else:
        return numerator / denominator

def text_to_vector(text):
    """Convert text to a frequency dictionary of words."""
    words = tokenize(text)
    vec = {}
    for w in words:
        vec[w] = vec.get(w, 0) + 1
    return vec

# ----- KNOWLEDGE LEARNING -----
def learn_from_files():
    if not os.path.exists(KNOWLEDGE_DIR):
        os.makedirs(KNOWLEDGE_DIR)
    for fname in os.listdir(KNOWLEDGE_DIR):
        if fname.endswith(".txt"):
            path = os.path.join(KNOWLEDGE_DIR, fname)
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
                memory["knowledge"][fname] = text
    save_memory()

# ----- RESPONSE GENERATION -----
def generate_response(user_input):
    input_vec = text_to_vector(user_input)
    best_match = None
    best_score = 0.0

    # Check memory knowledge
    for key, text in memory["knowledge"].items():
        text_vec = text_to_vector(text)
        score = cosine_similarity(input_vec, text_vec)
        if score > best_score:
            best_score = score
            best_match = text

    # Check previous conversations
    for convo in memory["conversations"]:
        text_vec = text_to_vector(convo["user"])
        score = cosine_similarity(input_vec, text_vec)
        if score > best_score:
            best_score = score
            best_match = convo["assistant"]

    if best_match:
        response = f"[Learned Info]: {best_match}"
    else:
        response = "I am learning. Tell me more!"

    # Save to memory
    memory["conversations"].append({"user": user_input, "assistant": response, "timestamp": str(datetime.now())})
    save_memory()
    return response

# ----- MAIN LOOP -----
def main():
    print("=== Offline AI (Lightweight Learning) ===")
    print("Type 'exit' to quit or 'learn' to load knowledge files.")
    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() == "exit":
            print("Goodbye!")
            break
        elif user_input.lower() == "learn":
            learn_from_files()
            print("Knowledge loaded from files.")
        else:
            response = generate_response(user_input)
            print(f"AI: {response}")

if __name__ == "__main__":
    main()
