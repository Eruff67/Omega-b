import streamlit as st
import json
import os
import datetime

# ---------- Memory ----------
MEMORY_FILE = "memory.json"

def load_memory():
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "r") as f:
            return json.load(f)
    return {"conversations": []}

def save_memory(memory):
    with open(MEMORY_FILE, "w") as f:
        json.dump(memory, f, indent=4)

memory = load_memory()

# ---------- AI Logic ----------
def respond(user_input):
    response = ""

    # Basic math handling
    try:
        # Only evaluate arithmetic expressions
        allowed_chars = "0123456789+-*/(). "
        if all(c in allowed_chars for c in user_input):
            result = eval(user_input)
            response += f"Math result: {result}\n"
    except Exception:
        pass

    # Simple English understanding / keywords
    user_lower = user_input.lower()
    if "hello" in user_lower or "hi" in user_lower:
        response += "Hello! How can I assist you today?\n"
    elif "time" in user_lower:
        response += f"The current time is {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    elif "remember" in user_lower:
        response += "I will remember that.\n"
    else:
        response += "I see. Can you tell me more?\n"

    # Add conversation to memory
    memory["conversations"].append({"user": user_input, "ai": response})
    save_memory(memory)

    return response

# ---------- Streamlit UI ----------
st.title("Offline AI Assistant")

user_input = st.text_input("You:", "")

if st.button("Send") and user_input:
    ai_response = respond(user_input)
    st.text_area("AI:", value=ai_response, height=200)

# Display past conversation
if st.checkbox("Show Conversation Memory"):
    for conv in memory["conversations"]:
        st.markdown(f"**You:** {conv['user']}")
        st.markdown(f"**AI:** {conv['ai']}")
