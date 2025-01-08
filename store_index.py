from src.helper import load_pdf_file, text_split, download_hugging_face_embeddings
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
import os


load_dotenv()

PINECONE_API_KEY=os.environ.get('pcsk_3hUTT4_8LuHsYDLS3KpVEMXW2Qx6T5yUAxbfX5Jow8zrhVTh8iFLJhrRKcMEEDHgf2J92o')
os.environ["PINECONE_API_KEY"] = 'pcsk_3hUTT4_8LuHsYDLS3KpVEMXW2Qx6T5yUAxbfX5Jow8zrhVTh8iFLJhrRKcMEEDHgf2J92o'


extracted_data=load_pdf_file(data='Data/')
text_chunks=text_split(extracted_data)
embeddings = download_hugging_face_embeddings()


pc = Pinecone(api_key='pcsk_3hUTT4_8LuHsYDLS3KpVEMXW2Qx6T5yUAxbfX5Jow8zrhVTh8iFLJhrRKcMEEDHgf2J92o')

index_name = "medicalbot"


pc.create_index(
    name=index_name,
    dimension=384, 
    metric="cosine", 
    spec=ServerlessSpec(
        cloud="aws", 
        region="us-east-1"
    ) 
) 

# Embed each chunk and upsert the embeddings into your Pinecone index.
docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    index_name=index_name,
    embedding=embeddings, 
)