from agent import agent

# Replace with whatever question you want to ask
user_input = "My child is 40 months old, had surgery involving 4 vertebrae, and the top vertebra operated on was number 12. Can you predict the kyphosis outcome?"
user_input = "Go through the docs and tell me, what website would be used for EMEA market in place of DV360 Reference from Nestle doc and please asnwer in bullet piints, les than 50 words.Simple t ou understad"
# Invoke the agent
response = agent.invoke({"messages": [("user", user_input)]})

# Print the agent's final response
print("\nAgent Response:")
print(response["messages"][-1].content)