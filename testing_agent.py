from agent import agent

# Replace with whatever question you want to ask
user_input = "My child is 40 months old, had surgery involving 4 vertebrae, and the top vertebra operated on was number 12. Can you predict the kyphosis outcome?"

# Invoke the agent
response = agent.invoke({"messages": [("user", user_input)]})

# Print the agent's final response
print("\nAgent Response:")
print(response["messages"][-1].content)