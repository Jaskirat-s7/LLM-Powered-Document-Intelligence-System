import os
import base64
import io
from typing import List, Any, Dict
from PIL import Image
import pdf2image
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.docstore.document import Document
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv

load_dotenv()

class MultimodalRAG:
    def __init__(self, openai_api_key: str):
        self.openai_api_key = openai_api_key
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vectorstore = None
        self.chain = None
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        self.llm = ChatOpenAI(
            model="gpt-4o",
            openai_api_key=self.openai_api_key,
            max_tokens=1024
        )

    def extract_images_from_pdf(self, pdf_path: str) -> List[Image.Image]:
        """Extracts images from a PDF file using pdf2image."""
        try:
            return pdf2image.convert_from_path(pdf_path)
        except Exception as e:
            print(f"Error extracting images: {e}")
            return []

    def encode_image(self, image: Image.Image) -> str:
        """Encodes a PIL image to base64 string."""
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def summarize_image(self, image: Image.Image) -> str:
        """Generates a summary of the image using OpenAI GPT-4o."""
        from langchain_core.messages import HumanMessage
        
        base64_image = self.encode_image(image)
        
        message = HumanMessage(
            content=[
                {"type": "text", "text": "Describe this image in detail. If it's a chart or graph, explain the data and trends shown."},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                }
            ]
        )
        
        try:
            response = self.llm.invoke([message])
            return response.content
        except Exception as e:
            print(f"Error summarizing image: {e}")
            return "Error analyzing image."

    def process_document(self, file_path: str) -> List[Document]:
        """
        Loads PDF text and extracts/summarizes images.
        Returns a list of Documents (text chunks + image summaries).
        """
        documents = []
        
        # 1. Load Text
        loader = PyPDFLoader(file_path)
        text_docs = loader.load()
        documents.extend(text_docs)
        
        # 2. Extract and Summarize Images
        images = self.extract_images_from_pdf(file_path)
        for i, img in enumerate(images):
            print(f"Analyzing image {i+1}/{len(images)}...")
            summary = self.summarize_image(img)
            
            # Create a Document for the image summary
            # We store the base64 image in metadata to display it later if extracted
            base64_img = self.encode_image(img)
            doc = Document(
                page_content=f"Image Summary (Page {i+1}): {summary}",
                metadata={
                    "source": file_path,
                    "page": i,
                    "type": "image",
                    "image_data": base64_img 
                }
            )
            documents.append(doc)
            
        return documents

    def create_vector_store(self, files: List[str]):
        """
        Processes files, splits text, creates embeddings, and initializes FAISS.
        """
        all_docs = []
        for file in files:
            all_docs.extend(self.process_document(file))
            
        # Split text
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        splits = text_splitter.split_documents(all_docs)
        
        # Create Vector Store
        self.vectorstore = FAISS.from_documents(splits, self.embeddings)
        
        # Create Chain
        self.chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.vectorstore.as_retriever(),
            memory=self.memory
        )

    def query(self, question: str) -> Dict[str, Any]:
        """Queries the RAG system."""
        if not self.chain:
            return {"answer": "Please upload and process documents first."}
        
        response = self.chain.invoke({"question": question})
        return response
