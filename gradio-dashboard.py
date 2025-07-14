import numpy as np
import pandas as pd

from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma

import gradio as gr

books = pd.read_csv("books_with_emotions.csv")
books["large_thumbnail"] = books["thumbnail"] + "&fife=w800" # cover with better resolution
# Set fallback cover
books["large_thumbnail"] = np.where(books["large_thumbnail"].isna(), "cover-not-found.jpg", books["large_thumbnail"])

raw_documents = TextLoader('tagged_description.txt', encoding='utf-8').load()
# Set chunk_size to 0 to prioritize splitting by separator over chunk size
text_splitter = CharacterTextSplitter(chunk_size=0, chunk_overlap=0, separator='\n')
documents = text_splitter.split_documents(raw_documents)
# Create database
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db_books = Chroma.from_documents(documents, embedding=embeddings)

def retrieve_semantic_recommendations(query: str, category: str = None, tone: str = None, initial_top_k: int = 50, final_top_k: int = 16) -> pd.DataFrame:
    recs = db_books.similarity_search(query, k=initial_top_k)
    books_list = [int(rec.page_content.strip('"').split()[0]) for rec in recs]
    books_recs = books[books["isbn13"].isin(books_list)].head(final_top_k)

    # Recommendation
    if category != "All":
        books_recs = books_recs[books_recs["simple_categories"] == category][:final_top_k]
    else:
        books_recs = books_recs.head(final_top_k)

    # Emotion category
    tone_sort_map = {
        "Happy": "joy",
        "Surprising": "surprise",
        "Angry": "anger",
        "Sad": "sadness",
    }
    if tone == 'Happy':
        sort_by = tone_sort_map.get(tone, 'neutral')
        books_recs.sort_values(by=sort_by, ascending=False, inplace=True)

    return books_recs