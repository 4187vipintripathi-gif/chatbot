
from langchain_huggingface import HuggingFaceEmbeddings
import os
from pinecone import Pinecone
from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from groq import Groq
load_dotenv()


pinecone_api_key = os.getenv("PINECONE_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")
client = Groq(api_key=groq_api_key)
pc = Pinecone(api_key=pinecone_api_key)

def generate_questions(text):
    prompt = f"""
    Generate 3 short questions that can be answered from the following text.
    IMPORTANT:
            - Do NOT add numbering
            - Do NOT add bullets
            - Only plain questions

    Text:
    {text}

    Return only questions, each on new line.
    """

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )

    output = response.choices[0].message.content

    # clean questions
    questions = [q.strip("- ").strip() for q in output.split("\n") if q.strip()]
    return questions[:4]

    
embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

index = pc.Index("langchainvector")
def read_doc(directory):
    # file_loader=DirectoryLoader(
    #                             directory,     
    #                             glob="**/*.docx", 
    #                             loader_cls=UnstructuredWordDocumentLoader
    #                             )

    file_loader=Docx2txtLoader(directory)
    document=file_loader.load()
    return document


def chunk(docs, chunk_size=500,chunk_overlap=50):
    text_splitters=RecursiveCharacterTextSplitter(chunk_size=chunk_size,chunk_overlap=chunk_overlap)
    docs=text_splitters.split_documents(docs)
    return docs

def load_data(doc):
    doc=read_doc(doc)
    qs=generate_questions(doc)
    document=chunk(docs=doc)
    
    records = []

    for i, d in enumerate(document):
        file_name = d.metadata["source"]
        file_name = os.path.basename(file_name)
        
        print(f"{file_name}_{i}\n", qs )
        vector = embeddings.embed_query(d.page_content)
        records.append({
            "id": f"{file_name}_{i}",
            "values": vector,
            "metadata": {
                "text": d.page_content,
                "question":qs

            }
        })
    



    batch_size=50
    for i in range(0, len(records), batch_size):
        batch = records[i:i+batch_size]
        
        
        index.upsert(
            namespace="example-namespace",
            vectors=batch
        )

def start_loading(folder):
    for file in os.listdir(folder):
        if file.endswith(".docx"):
            path = os.path.join(folder, file)
            load_data(path)
            print(f"File: {path}")

folder = r"C:\Users\4187v\Downloads\document\app\data"
start_loading(folder)