from dotenv import load_dotenv
load_dotenv()
from pydantic import BaseModel
from langchain_groq import  ChatGroq
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from typing import Annotated
class ChatState(BaseModel):
    message:Annotated[list,add_messages]
llm=ChatGroq(model="opeai/gpt-oss-20b")
def chatBotNode(state:ChatState) -> ChatState:
    res=llm.invoke(state.message)
    state.message=[res]
    return state

memory=InMemorySaver()

graph=StateGraph(ChatState)
graph.add_node("chatBot",chatBotNode)

graph.add.edge(START,"chatBot")
graph.add_edge("chatBot",END)

graph=graph.compile(checkpointer=memory)

config={"configurable":{"thread_id":"my-bot-1"}}

while True:
    qurey=input("You: ")
    if qurey.lower() in ["exit","quit"]:
        print("thanks you later!")
        break
    res=graph.invoke(
        {"message":[{"role":"user","content":qurey}]}
        ,config)
    
    
ans=res["message"][-1].content
print("Bot:",ans)