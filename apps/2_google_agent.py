from dotenv import load_dotenv
load_dotenv()
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_groq import ChatGroq
from langchain.agents import create_agent

model=ChatGroq(model="openai/gpt-oss-20b")
search=GoogleSerperAPIWrapper()

agent=create_agent(
    model=model,
    tools=[search.run],
    system_prompt="you are a agent and can search for any question on google."
)

while True:
    query=input("user:")
    if query.lower()=="quit":
        print("good_bye")
        break
    response=agent.invoke({"messages":[{"role":"user","content":query}]})
    print("AI:",response["messages"][-1].content)