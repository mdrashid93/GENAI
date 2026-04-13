from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings,ChatOpenAI
from langchain_vectorstores import inMemoryVectorStore
from langchain.agents import create_agent
from langchian.tools import tool
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import InMemorySaver

## load the documents
loader=PyPDFLoader("state_of_the_union.pdf")
docs=loader.load()

#split the documents into chunks
splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
docs=splitter.split_documents(documents=docs)

# embeddings and vector store
embeddings=OpenAIEmbeddings(model="text-embedding-3-small")
vector_db=inMemoryVectorStore.from_documents(documents=docs,embedding=embeddings)

#create the agent
llm=ChatGroq(model="openai/gpt-oss-20b")

@tool
def retrieve_context(query:str):
    """
    Retrieve relevant context from the vector store based on the query.
    """
    context=""
    print("query",query)
    docs=vector_db.similarity_search(query,k=3)
    for doc in docs:
        context =doc.page_content+"\n\n" 
    print("context:",context)
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

while True:
    query=input("User: ")
    if query.lower()=="quit":
        break
    response=agent.invoke({"message":[{"role":"user","content":query}]},
              {"configurable":{"thread_id":"1"}}            )
    result=response["message"][-1].content
    print("Agent:",result)