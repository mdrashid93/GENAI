from dotenv import load_dotenv
load_dotenv()
### DB, LLM, TOOLS, CREATE_AGENT, SYSTEM_PROMPT
from langchain_groq import ChatGroq
from langchain_comunity.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents import create_agent
import streamlit as st

db=SQLDatabase.from_uri("sqlite://my_tasks.db")
db.run(""" 
       CREATE TABLE IF NOT EXISTS tasks(
           id INTEGER PRIMARY KEY AUTOINCREMENT,
           title TEXT NOT NULL,
           description TEXT,
           status TEXT CHECK (status IN ('pending', 'in_progress','completed',)) DEFAULT 'pending',
           created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
       );
       
       """)
print("db TABLE created successfully")

### LLM TOOLS MEMORY SYSTEM_PROMPT
model=ChatGroq(model="openai/gpt-oss-20b")
toolkit=SQLDatabaseToolkit(db=db, llm=model)
tools=toolkit.get_tools()
# memory=InMemorySaver()

#   
system_prompt="""
you are taskmanagement assistent that interacts with a sql database containnig a "tas"

TASK RULES:
1. Limit SELECT queries to 10 results max wiht ORDER BY created_at  DESC
2. After CREATE/UPDATE/DELETE, confirm with SELECT query
3. If the user requests a list of tasks, present the output in structured table format to ensure a clean and organized display in the browser."

CRUD OPERATIONS:
    CREATE: INSERT INTO tasks(title, desccriptionns, status)
    READ: SELECT * FROM tasks WHERE ... LIMIT 10
    UPDATE: UPDATE tasks SET status=? OR title=?

Table schema: id, title, description, status(pending/in_progress/completed), created_at.
"""
@st.cache_resource
def get_agent():
    agent = create_agent(
        model=model,
        tools=tools,
        checkpointer=InMemorySaver,
        system_prompt=system_prompt
    )
    return agent
agent=get_agent()

if "messages" not in st.session_state:
    st.session_state.messages=[]
    
for message in st.session_state.messages:
    st.chat_message(message["role"]).markdown(message["content"])

prompt=st.chat_input("ask me to manage your tasks")
if prompt:
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role":"user","content":prompt})
    with st.chat_message("ai"):
        with st.spinner("processing..."):
    
            response=agent.invoke(
                {"messages":[{"role":"user","content":prompt}]},
                {"configurable":{"thread_id":"1"}}
            )
            result=response["messages"][-1].content
            st.markdown(result)
            st.session_state.messages.append({"role":"ai","content":result})


