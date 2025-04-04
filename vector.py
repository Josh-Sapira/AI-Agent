from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import pandas as pd

# Load dataframes
# pitching_df = pd.read_csv("./Data/pitching.csv", errors='replace')
# batting_df = pd.read_csv("./Data/batting.csv", errors='replace')
with open('Data/pitching.csv', 'r', encoding='utf-8', errors='replace') as f:
    pitching_df = pd.read_csv(f, delimiter=';')
with open('Data/batting.csv', 'r', encoding='utf-8', errors='replace') as f:
    batting_df = pd.read_csv(f, delimiter=';')

# Embeddings model
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

# Database 
db_location = "./chrome_langchain_db"

# Variable to make sure we only vectorize data once
add_documents = not os.path.exists(db_location)

# Fill database
if add_documents:

    # Initialize lists to store
    documents = []
    ids = []

    # For batting dataframe 
    for i, row in batting_df.iterrows():
        document = Document(
            # Content to look up in db
            page_content = " ".join([str(row[col]) for col in batting_df.columns]),

            # Include season in metadata
            metadata = {
                "season": 2023, 
                "position": "batter",
                "player": row["Name"],
                "team": row["Tm"]
            },
            
            # Set id to row number as string
            id = "batter" + str(i)
        )
        ids.append("batter" + str(i))
        documents.append(document)

    # For pitching dataframe 
    for i, row in pitching_df.iterrows():
        document = Document(
            # Content to look up in db
            page_content = " ".join([str(row[col]) for col in pitching_df.columns]),

            # Include season in metadata
            metadata = {
                "season": 2023, 
                "position": "pitcher",
                "player": row["Name"],
                "team": row["Tm"]
            },
            
            # Set id to row number as string
            id = "pitcher" + str(i)
        )
        ids.append("pitcher" + str(i))
        documents.append(document)

# Store vectors
vector_store = Chroma(
    collection_name = "baseball_data",
    persist_directory = db_location,
    embedding_function = embeddings
)

# Add documents
if add_documents:
    vector_store.add_documents(documents=documents, ids=ids)

# Allow LLM to retrieve information
retriever = vector_store.as_retriever(
    search_type="similarity"
)