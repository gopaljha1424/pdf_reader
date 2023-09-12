import streamlit as st
from dotenv import load_dotenv
import pickle 
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS 
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
import os  


#sidebar contents
with st.sidebar:
    st.title("💬 Pdf Chat App")
    st.markdown('''
    ## About
    This app is an Pdf Chatbot built using:
    - [Streamlit](https://streamlit.io/generative-ai)
    - [LangChain](https://python.langchain.com/docs/get_started/quickstart)
    - [OpenAI](https://platform.openai.com/docs/models)
    ''')
    add_vertical_space(5)
    st.write('Made by Employee(Times Pro)')
 


def main():
    st.header("Chat With Pdf 💬")

    load_dotenv()

    # upload a pdf file
    pdf =  st.file_uploader("Upload Your Pdf", type ='pdf')
    if pdf is not None:
        pdf_reader = PdfReader(pdf)

    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
        )
    
    chunks = text_splitter.split_text(text=text)

    #embeddings
    
    store_name = pdf.name[:-4]
    
    if os.path.exists(f"{store_name}.pkl"):
        with open(f"{store_name}.pkl", "rb") as f:
            VectorStore = pickle.load(f)
        st.write("Embedding loaded from the Disk")
    else:
        embeddings = OpenAIEmbeddings()
        vectorstores = FAISS.from_texts(chunks, embedding=embeddings)
        with open(f"{store_name}.pkl","wb")as f:
            pickle.dump(vectorstores, f)
    #Accept user question/query
    query = st.text_input("Ask questions about your pdf file:")
    #st.write(query)

    if query:
        docs = VectorStore.similarity_search(query=query, k=3)

        llm = OpenAI()
        chain = load_qa_chain(llm=llm, chain_type="stuff")
        response = chain.run(input_documents=docs, question=query)
        st.write(response)

        
        #st.write(docs)

 
    #st.write(chunks)
    #st.write(text)  



if __name__ == '__main__':
    main()  