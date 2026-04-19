from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import PyPDFLoader,PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import InMemoryVectorStore
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import InMemorySaver
import streamlit as st

# data in st session 
if "documents_uploaded" not in st.session_state:
    st.session_state.documents_uploaded=False
if "agent" not in st.session_state:
    st.session_state.agent=None
if "vector_store" not in st.session_state:
    st.session_state.vector_store=None
if "messages" not in st.session_state:
    st.session_state.messages=[]    


def process_documents(path):
    

    # load the documents
    loader=PyPDFDirectoryLoader("../data/")
    docs=loader.load()

    #split the documents into chunks
    splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
    docs=splitter.split_documents(documents=docs)  

    # print(len(docs),docs[0])
    #embeddings and vector store
    em=OpenAIEmbeddings(model="text-embedding-3-large")
    vector_db=InMemoryVectorStore.from_documents(
        documents=docs,
        embedding=em
    )
    #create the agent - tool ,llm,prompt
    llm=ChatGroq(model="openai/gpt-oss-20b")

    @tool
    def retrieve_context(query:str):
        """retrieve documents relevant to a query form the knowledge based"""
        context=""
        docs=vector_db.similarity_search(query=query k=3)
        for doc in docs:
            context=doc.page_content + "\n\n"
        return context

    system_prompt="""
    you are a helpful assistant that answer the questions using retrieved context.
    my knowledge base cnsists of the details from the uploaded document. 
    always use the retrieved context to answer the question. if the retrieved context is not relevant to the question, say you don't know.
    """

    memory=InMemorySaver()

    agent=create_agent(
        model=llm,
        tools=[retrieve_context],
        system_prompt=system_prompt,
        checkpointer=memory
    )
    st.session_state.agent=agent
    st.session_state.documents_uploaded=True

    # while True:
    #     query=input("user:")
    #     if query.lower()=="quit":
    #         break
    #     res=agent.invoke({"message":[{"role":"user","content":query}]},
    #     {"configurable":{"thread_id":1}}
    #                      )
    #     result=res["message"][-1].content
    #     print("AI:",result)


    ###upload ui
    if not st.session_state.documents_uploaded:
        uploaded=st.file_uploader(label="select a pdf files", type=["pdf"], accept_multiple_files=True):
        if uploaded:
            with st.spinner("processing..."):
                path="./doc_files/"
                for file in uploaded:
                    with open(path+file.name,"wb") as f:
                        f.write(file.getvalue())


#chat ui