#Disable spell checking for document
#cSpell:disable


################################
#          LIBRARIES           #
################################

import os
import csv
from typing import List, Optional

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
from langchain.docstore.document import Document
from langchain.llms.ollama import Ollama

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredMarkdownLoader, UnstructuredExcelLoader

from langchain.vectorstores.pinecone import Pinecone
import pinecone

import chainlit as cl
from chainlit.types import AskFileResponse



################################
#           PINECONE           #
################################

pinecone.init(
    api_key=os.environ.get("PINECONE_API_KEY"),
    environment=os.environ.get("PINECONE_ENV"),
)

index_name = "chainlit-chatbot"
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                    model_kwargs={'device': 'cpu'})
namespaces = set()



################################
#        RAG FUNCTIONS         #
################################

def process_file(file: AskFileResponse):
    if file.type == "text/plain":
        Loader = TextLoader
    elif file.type == "application/pdf":
        Loader = PyPDFLoader

        loader = Loader(file.path)
        documents = loader.load()
        docs = text_splitter.split_documents(documents)
        for i, doc in enumerate(docs):
            doc.metadata["source"] = f"source_{i}"
        return docs


def get_docsearch(file: AskFileResponse):
    docs = process_file(file)

    # Save data in the user session
    cl.user_session.set("docs", docs)

    # Create a unique namespace for the file
    namespace = file.id

    if namespace in namespaces:
        docsearch = Pinecone.from_existing_index(
            index_name=index_name, embedding=embeddings, namespace=namespace
        )
    else:
        docsearch = Pinecone.from_documents(
            docs, embeddings, index_name=index_name, namespace=namespace
        )
        namespaces.add(namespace)

    return docsearch



################################
#           CHAINLIT           #
################################

welcome_message = """Welcome to the PDF/TXT QA demo! To get started:
1. Upload a PDF or text file
2. Ask a question about the file
"""

@cl.password_auth_callback
def auth_callback(username: str, password: str) -> Optional[cl.User]:
    with open("users.csv", "r") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row["username"] == username:
                if row["password"] == password:
                    return cl.User(identifier=username,  metadata={"role": "admin", "provider": "credentials"})
    return None


@cl.on_chat_start
async def start():
    await cl.Avatar(
        name="Chatbot",
        path="public/chatbot-icon.png",
    ).send()
    files = None
    while files is None:
        files = await cl.AskFileMessage(
            content=welcome_message,
            accept=["text/plain", "application/pdf"],
            max_size_mb=20,
            timeout=180,
        ).send()

    file = files[0]

    msg = cl.Message(content=f"Processing `{file.name}`...", disable_feedback=True)
    await msg.send()

    # No async implementation in the Pinecone client, fallback to sync
    docsearch = await cl.make_async(get_docsearch)(file)

    message_history = ChatMessageHistory()

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key="answer",
        chat_memory=message_history,
        return_messages=True,
    )

    chain = ConversationalRetrievalChain.from_llm(
        Ollama(model="mistral:instruct",),
        chain_type="stuff",
        retriever=docsearch.as_retriever(),
        memory=memory,
        return_source_documents=True,
    )

    # Let the user know that the system is ready
    msg.content = f"`{file.name}` processed. You can now ask questions!"
    await msg.update()

    cl.user_session.set("chain", chain)


@cl.on_message
async def main(message: cl.Message):
    chain = cl.user_session.get("chain")  # type: ConversationalRetrievalChain
    cb = cl.AsyncLangchainCallbackHandler()
    res = await chain.ainvoke(message.content, callbacks=[cb])
    answer = res["answer"]
    source_documents = res["source_documents"]  # type: List[Document]

    text_elements = []  # type: List[cl.Text]

    if source_documents:
        for source_idx, source_doc in enumerate(source_documents):
            source_name = f"source_{source_idx}"
            # Create the text element referenced in the message
            text_elements.append(
                cl.Text(content=source_doc.page_content, name=source_name)
            )
        source_names = [text_el.name for text_el in text_elements]

        if source_names:
            answer += f"\nSources: {', '.join(source_names)}"
        else:
            answer += "\nNo sources found"

    await cl.Message(content=answer, elements=text_elements).send()