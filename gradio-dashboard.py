import numpy as np
import pandas as pd

from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
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
        "Suspenseful": "fear",
        "Sad": "sadness",
        "All": 'neutral'
    }
    sort_by = tone_sort_map.get(tone, 'neutral')
    books_recs.sort_values(by=sort_by, ascending=False, inplace=True)

    return books_recs

def get_recommendations(query: str, category: str = None, tone: str = None):
    recommendations = retrieve_semantic_recommendations(query, category, tone)
    results = []

    for _, row in recommendations.iterrows():
        # Truncate description if more than 30 words
        description = row["description"]
        truncated_desc_split = description.split()
        truncate_limit = 30
        truncated_description = " ".join(truncated_desc_split[:truncate_limit]) + ("..." if len(truncated_desc_split) > truncate_limit else "")

        authors_split = row["authors"].split(';')
        authors_str = ", ".join(authors_split[:-1]) + " and " + authors_split[-1] if len(authors_split) > 1 else authors_split[0]

        caption = f"{row['title']} by {authors_str}: {truncated_description}"
        results.append((row['large_thumbnail'], caption))

    return results

categories = ["All"] + sorted(books['simple_categories'].unique())
tones = ["All", "Happy", "Surprising", "Angry", "Surprising", "Sad"]

# Gradio configs
with gr.Blocks(theme=gr.themes.Glass()) as dashboard:
    gr.Markdown('# Semantic book recommender')

    with gr.Row():
        user_query = gr.Textbox(label="Please enter a book's description.", placeholder="ex: a story about forgiveness")
        category_dropdown = gr.Dropdown(choices=categories, label="Select a category:", value="All")
        tone_dropdown = gr.Dropdown(choices=tones, label="Select an emotional tone:", value="All")
        submit_button = gr.Button("Find recommendations")

    gr.Markdown('## Recommendations')
    outputs = gr.Gallery(label="Recommended books", columns=8, rows=2)

    submit_button.click(fn=get_recommendations, inputs=[user_query, category_dropdown, tone_dropdown], outputs=outputs)

if __name__ == '__main__':
    dashboard.launch(share=True)

