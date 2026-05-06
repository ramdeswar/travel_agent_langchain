# -------------------------------------------------------------
# TRAVEL AGENT CHATBOT — LANGCHAIN + LANGGRAPH + GEMINI
# -------------------------------------------------------------
# Features implemented:
# 1. Streams responses token-by-token
# 2. Manual memory using a Python list
# 3. Summarizes chat history every 5 messages
# 4. Replaces older messages with summary to reduce context
# 5. Rejects non-travel questions
# 6. Uses LangChain PromptTemplate
# 7. Uses Gemini LLM
# -------------------------------------------------------------

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langgraph.graph import StateGraph, END
from typing import List, Dict
import asyncio
import os
from dotenv import load_dotenv


# -------------------------------------------------------------
# 1. MANUAL MEMORY STRUCTURE
# -------------------------------------------------------------
# We store all messages in a Python list.
# After every 5 user messages, we summarize and compress memory.
# -------------------------------------------------------------

memory: List[Dict] = []      # Stores full conversation
message_counter = 0          # Counts user messages
SUMMARY_THRESHOLD = 5        # Summarize after every 5 messages


# -------------------------------------------------------------
# 2. LLM SETUP (Gemini)
# -------------------------------------------------------------

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("Missing GOOGLE_API_KEY environment variable")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.3,
    streaming=True,  # Enables token-by-token streaming
    google_api_key=GOOGLE_API_KEY
)

# -------------------------------------------------------------
# 3. PROMPT TEMPLATE
# -------------------------------------------------------------
# The agent must ONLY answer travel-related questions.
# Otherwise respond EXACTLY: "I can't help with it."
# -------------------------------------------------------------

prompt = PromptTemplate(
    input_variables=["history", "user_input"],
    template="""
You are a helpful travel agent chatbot.

You MUST follow these rules:
1. Only answer travel-related questions (flights, hotels, destinations, trips, travel tips).
2. If the question is NOT travel-related, respond EXACTLY with:
   I can't help with it.
Conversation history:
{history}

User: {user_input}
Assistant:"""
)


# -------------------------------------------------------------
# 4. SUMMARIZATION FUNCTION
# -------------------------------------------------------------
async def summarize_history(memory_list):
    """Summarizes the entire conversation to reduce context size."""
    summary_prompt = PromptTemplate(
        input_variables=["messages"],
        template="""
Summarize the following conversation in 3 lines, keeping only important travel context:

{messages}

Summary:"""
    )

    chain = summary_prompt | llm
    result = ""
    async for chunk in chain.astream({"messages": str(memory_list)}):
        if chunk.content:
            result += chunk.content
    return result


# -------------------------------------------------------------
# 5. LANGGRAPH STATE
# -------------------------------------------------------------
class ChatState(Dict):
    history: str
    user_input: str
    output: str


# -------------------------------------------------------------
# 6. NODE: PROCESS USER MESSAGE
# -------------------------------------------------------------
async def process_message(state: ChatState):
    global memory, message_counter

    user_input = state["user_input"]

    # Add user message to memory
    memory.append({"role": "user", "content": user_input})
    message_counter += 1

    # Build history string
    history_text = "\n".join([f"{m['role']}: {m['content']}" for m in memory])

    # Build final prompt
    final_prompt = prompt.format(history=history_text, user_input=user_input)

    # STREAMING RESPONSE
    print("\nAssistant:", end=" ", flush=True)
    streamed_output = ""

    async for chunk in llm.astream(final_prompt):
        if chunk.content:
            print(chunk.content, end="", flush=True)
            streamed_output += chunk.content

    print("\n")

    # Save assistant response
    memory.append({"role": "assistant", "content": streamed_output})

    # SUMMARIZE EVERY 5 USER MESSAGES
    if message_counter >= SUMMARY_THRESHOLD:
        print("\n--- Conversation Summary Generated ---\n")

        summary = await summarize_history(memory)

        # Print summary to console
        print(summary)
        print("\n--------------------------------------\n")

        # Replace old memory with summary
        memory = [{"role": "system", "content": summary}]

        # Reset counter
        message_counter = 0

    return {"output": streamed_output}


# -------------------------------------------------------------
# 7. BUILD LANGGRAPH
# -------------------------------------------------------------
graph = StateGraph(ChatState)
graph.add_node("chat", process_message)
graph.set_entry_point("chat")
graph.set_finish_point("chat")

app = graph.compile()


# -------------------------------------------------------------
# 8. MAIN LOOP
# -------------------------------------------------------------
async def main():
    print("Travel Agent Chatbot Ready!\n")

    while True:
        user_input = input("You: ")

        # Exit conditions
        if user_input.strip().lower() in ["bye", "exit", "quit", "goodbye"]:
            print("Assistant: Safe travels! Goodbye.")
            break

        await app.ainvoke({"user_input": user_input})


# Run the chatbot
asyncio.run(main())
