from dotenv import load_dotenv
load_dotenv()
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
llm=ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# while True:
#     query=input("user:")
#     if query.lower() in ["quit","exit","bye"]:
#         print("goodbye")
#         break
# result=llm.invoke(query)
# print("ai:",result.content,"\n")


#--------------------------------------------------
st.title("Askbuddy - AI Qna Bot")
st.markdown("my Qna bot with langchain and google gemini !")

if "message" not in st.session_state:#message name ki key mere session me store nahi karti to store karwa de 
    st.session_state.messages=[]
for message in st.session_state.messages:
    role=message["role"]
    content=message["content"]
    st.chat_message(role).markdown(content)
    

query=st.chat_input("ask anything")
if query:
    st.session_state.messages.append({"role":"user","content":query})
    st.chat_message("user").markdown(query)
    res=llm.invoke(query)
    st.chat_message("ai").markdown(res.content)
    st.session_state.messages.append({"role":"user","content":res.content})