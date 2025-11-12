# Jack.AI ASI Offline — Full Auto-Learning, Multi-language, Semantic Memory
# Fully offline, single-file, copy/paste ready
# Requirements: pip install scikit-learn nltk pandas numpy langdetect joblib

import os
import json
import uuid
import nltk
import pandas as pd
import numpy as np
from langdetect import detect
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from joblib import dump, load

# ---------------- Setup ----------------
MEMORY_FILE = "jack_memory.json"
KB_FILE = "jack_kb.json"
EMBEDDINGS_FILE = "jack_embeddings.joblib"
LEARNING_FOLDER = "learning_files"

os.makedirs(LEARNING_FOLDER, exist_ok=True)

memory = json.load(open(MEMORY_FILE, "r", encoding="utf-8")) if os.path.exists(MEMORY_FILE) else {"conversations": []}
kb = json.load(open(KB_FILE, "r", encoding="utf-8")) if os.path.exists(KB_FILE) else {"files": {}}
embeddings = load(EMBEDDINGS_FILE) if os.path.exists(EMBEDDINGS_FILE) else {"matrix": None, "keys": []}

nltk.download('punkt', quiet=True)

# ---------------- NLP Utilities ----------------
def tokenize(text):
    return nltk.word_tokenize(text.lower())

def summarize_text(text, max_sentences=3):
    sentences = nltk.sent_tokenize(text)
    if len(sentences) <= max_sentences:
        return text
    vectorizer = TfidfVectorizer().fit_transform(sentences)
    vectors = vectorizer.toarray()
    scores = np.sum(vectors, axis=1)
    ranked_sentences = [sentences[i] for i in np.argsort(scores)[::-1][:max_sentences]]
    return " ".join(ranked_sentences)

def detect_language(text):
    try:
        return detect(text)
    except:
        return "en"

# ---------------- File Analysis ----------------
def read_file(path):
    if not os.path.exists(path):
        return None, "File not found."
    if path.endswith((".txt", ".md")):
        with open(path, "r", encoding="utf-8") as f:
            return f.read(), None
    elif path.endswith(".csv"):
        df = pd.read_csv(path)
        return df.to_string(), None
    elif path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            return json.dumps(json.load(f)), None
    else:
        return None, "Unsupported file type."

def learn_file(path):
    content, error = read_file(path)
    if error: return error
    summary = summarize_text(content)
    kb["files"][path] = {"content": content, "summary": summary}
    update_embeddings()
    save_kb()
    return f"File '{path}' learned successfully."

def auto_learn_folder():
    for file in os.listdir(LEARNING_FOLDER):
        full_path = os.path.join(LEARNING_FOLDER, file)
        if file not in kb["files"]:
            learn_file(full_path)

# ---------------- Embeddings ----------------
def update_embeddings():
    global embeddings
    corpus = [kb[f]["content"] for f in kb["files"]]
    if corpus:
        vectorizer = TfidfVectorizer().fit(corpus)
        matrix = vectorizer.transform(corpus).toarray()
        embeddings = {"matrix": matrix, "keys": list(kb["files"].keys())}
        dump(embeddings, EMBEDDINGS_FILE)

# ---------------- Math & Logic ----------------
def solve_math(question):
    try:
        allowed = {k: v for k, v in __import__("math").__dict__.items() if not k.startswith("__")}
        return str(eval(question, {"__builtins__": None}, allowed))
    except:
        return "Cannot solve that."

def logical_reasoning(statement):
    return f"I analyzed the statement '{statement}', but advanced logic reasoning is under development."

# ---------------- AI Response ----------------
def generate_response(user_input, persona="default"):
    lang = detect_language(user_input)
    auto_learn_folder()
    
    # Check math
    if any(c.isdigit() for c in user_input) and any(op in user_input for op in "+-*/()"):
        return solve_math(user_input), "en"

    # Check logic
    if user_input.lower().startswith("/logic"):
        return logical_reasoning(user_input[len("/logic"):].strip()), "en"

    # Semantic search
    base_response = "I don't know yet. Provide a file or more context."
    if embeddings["matrix"] is not None:
        vectorizer = TfidfVectorizer().fit([kb[f]["content"] for f in kb["files"]])
        query_vec = vectorizer.transform([user_input]).toarray()
        sim = cosine_similarity(query_vec, embeddings["matrix"])
        best_idx = sim.argmax()
        if sim[0][best_idx] > 0.1:
            key = embeddings["keys"][best_idx]
            base_response = kb[key]["summary"]

    if persona != "default":
        base_response = f"[{persona} mode] {base_response}"

    return base_response, lang

# ---------------- Save KB ----------------
def save_kb():
    json.dump(kb, open(KB_FILE, "w", encoding="utf-8"), indent=2)
    json.dump(memory, open(MEMORY_FILE, "w", encoding="utf-8"), indent=2)

# ---------------- Main Loop ----------------
def main():
    print("Jack.AI ASI Offline — Full Auto-Learning Multi-language AI")
    persona = "default"

    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in ["exit", "quit"]:
            save_kb()
            print("Memory saved. Goodbye!")
            break

        # Commands
        if user_input.startswith("/persona"):
            persona = user_input[len("/persona"):].strip() or "default"
            print(f"Jack: Persona set to '{persona}'.")
            continue

        if user_input.startswith("/learn"):
            path = user_input[len("/learn"):].strip()
            print(f"Jack: {learn_file(path)}")
            continue

        # Generate response
        response, lang = generate_response(user_input, persona)
        print(f"Jack ({lang}): {response}")

        # Update memory
        memory["conversations"].append({
            "id": str(uuid.uuid4()),
            "user": user_input,
            "assistant": response
        })

if __name__ == "__main__":
    main()
