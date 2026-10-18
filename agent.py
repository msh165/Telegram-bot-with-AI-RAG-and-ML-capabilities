from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from llm import groq_llm
from tools import kyphosis_predict

agent = create_agent(
    model=groq_llm,
    tools=[kyphosis_predict],
    system_prompt="You are a helpful assistant, whenever user gives you info about their childs spinal surgery. Use that to predict whether kyphosis is absent or not, DO NOT do on your own, always use the predict_kyphosis functoin",
    # Remembers the conversation per thread_id (used by telegram_bot.py, one thread per chat)
    # checkpointer=InMemorySaver(),
)
