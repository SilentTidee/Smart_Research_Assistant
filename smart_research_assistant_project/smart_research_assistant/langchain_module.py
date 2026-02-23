import os
from typing import List,Dict, Any
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain_core.documents import Document
import tempfile

load_dotenv()

class ResearchAssistantLangchain:

    def __init__(self):
        """Initialize the Research Assistant with OpenAI configurations."""
        self.llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
        self.embeddings = OpenAIEmbeddings()
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        self.vector_db = None
        # Create a temporary directory for Chroma vector store
        self.persist_directory = tempfile.mkdtemp()
        print(f"Created temporary Chroma directory: {self.persist_directory}")



    def load_urls(self, urls: List[str]) -> List[Document]:
        all_documents = []
    
        for url in urls:
            try:
                loader = WebBaseLoader(url)
                data = loader.load()
                
                split_documents = self.text_splitter.split_documents(data)
                all_documents.extend(split_documents)

                print(f"Successdully loaded and processed {url}")
            except Exception as e:
                print(f"Error loading {url}: {e}")
        return all_documents
    
    def create_vector_store(self, documents: List[Document]):

        try:
            from langchain_community.vectorstores import FAISS

            self.vector_db = FAISS.from_documents(documents, self.embeddings)
            print(f"Created FAISS vector store with {len(documents)} documents.")
        except Exception as e:
            print(f"Error creating FAISS vector store: {e}")

            try:
                self.vector_db = Chroma.from_documents(documents, self.embeddings, persist_directory = None)
                print(f"Created in-memory Chroma vector store .")
            except Exception as e2:
                print(f"Error creating in-memory Chroma vector store: {e2}")
                raise ValueError


    def query_data(self, query: str, num_results: int = 5) -> Dict[str, Any]:

        if not self.vector_db:
            raise ValueError("Vector store not initialized. Please load URLs and create vector store first.")

        retriever = self.vector_db.as_retriever(search_kwargs={"k": num_results})


        prompt_text = """
        Answer the following question based only on the provided context:
        
        <context>
        {context}
        </context>
        
        Question: {input}
        """

        prompt = ChatPromptTemplate.from_template(prompt_text)

        document_chain = create_stuff_documents_chain(self.llm,prompt)   
        retrieval_chain = create_retrieval_chain(retriever,document_chain)   
        response = retrieval_chain.invoke({"input": query})    

        return {
            "answer": response["answer"],
            "source_documents": response["context"]
        } 
    


    def summarize_document(self, document: str) -> str:

        message = HumanMessage(content=f"Summarize the following document in a concise but comprehensive manner:\n\n{document}")

        # Invoke the model directly 
        response = self.llm.invoke([message])
        return response.content