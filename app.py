from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain.pinecone import PineconeVectorStore
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers 
from langchain.chains import RetrievalQA
from typing import TYPE_CHECKING, Any, Dict, List, Optional
from langchain.vectorstores import VectorStore
from langchain_core.documents import Document
from langchain_core.callbacks import CallbackManagerForRetrieverRun, AsyncCallbackManagerForRetrieverRun
from dataclasses import dataclass
from dotenv import load_dotenv 
from src.prompt import *
import os


app = Flask(__name__)

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV')

embeddings = download_hugging_face_embeddings()

index_name = "medical-chatbot"

docsearch = PineconeVectorStore.from_existing_index(index_name,embeddings)

PROMPT=PromptTemplate(template=prompt_template, input_variables=["context", "question"])
chain_type_kwargs = {"prompt": PROMPT}

llm = CTransformers(model='model/llama-2-7b-chat.ggmlv3.q4_0.bin',
                    model_type='llama',
                    config={'max_new_tokens':512,
                            'temperature': 0.8})


'''The .as_retriever method is deprecated so we can't use the below code anymore'''
# qa = RetrievalQA.from_chain_type(
#     llm=llm,
#     chain_type="stuff",
#     retriever=docsearch.as_retriever(search_kwargs={'k':2}),
#     return_source_documents=True,
#     chain_type_kwargs=chain_type_kwargs)

'''We have to create our own CustomRetriever Class for that'''

class CustomRetriever(BaseRetriever):
    
    def __init__(self):
        self.retriever: docsearch.as_retriever(search_kwargs={"k": 2})
        vectorstores: VectorStore

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        
        documents = self.retriever.get_relevant_documents(query, callbacks=run_manager.get_child())
        
        documents = sorted(documents, key=lambda doc: doc.metadata.get('source'))
        
        return documents

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun
    ) -> List[Document]:
        
        documents = self.retriever.get_relevant_documents(query, callbacks=run_manager.get_child())
        
        documents = sorted(documents, key=lambda doc: doc.metadata.get('source'))
        
        return documents

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=CustomRetriever(),
    return_source_documents=True,
    chain_type_kwargs=chain_type_kwargs)

@app.route("/")
def index():
    return render_template('chat.html')


if __name__ == "__main__":
    app.run(host='0.0.0.0',port=8080, debug=True)