import streamlit as st
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import CharacterTextSplitter


def process_input(urls, question):
   
    model_local = Ollama(model="mistral")

    
    urls_list = urls.split("\n")
    docs = [WebBaseLoader(url.strip()).load() for url in urls_list if url.strip()]
    docs_list = [item for sublist in docs for item in sublist]

   
    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=7500, chunk_overlap=100
    )
    doc_splits = text_splitter.split_documents(docs_list)

    
    vectorstore = Chroma.from_documents(
        documents=doc_splits,
        collection_name="rag-chroma",
        embedding=OllamaEmbeddings(model="nomic-embed-text"),
    )
    retriever = vectorstore.as_retriever()

    
    after_rag_template = """Answer the question based only on the following context:
    {context}
    Question: {question}
    """
    after_rag_prompt = ChatPromptTemplate.from_template(after_rag_template)

    after_rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | after_rag_prompt
        | model_local
        | StrOutputParser()
    )

    return after_rag_chain.invoke(question)


# Streamlit UI
st.title("Neuro Assistance")
st.write("Enter URLs (one per line) and a question to query the documents.")


urls = st.text_area("Enter URLs separated by new lines", height=150)
question = st.text_input("Question")


if st.button("Query Documents"):
    with st.spinner("Processing..."):
        try:
            answer = process_input(urls, question)
            st.text_area("Answer", value=answer, height=300, disabled=True)
        except Exception as e:
            st.error(f"Error: {str(e)}")