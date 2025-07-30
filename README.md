# Project Overview
This Book Recommender App recommends books based on user queries. Under the hood, it uses vector search for semantic similarity, zero-shot classification to categorize books, and sentiment analysis to extract emotions from descriptions. The app uses Gradio to demo and for user to interact with the database with a friendly user interface.


# Technologies Used
Python: Core programming language for the project.

LangChain: Framework for integrating LLMs and building the vector database.

Hugging Face: Zero-shot classification model for categorization and text-classification model for sentiment analysis.

Weaviate: Vector database for storing and querying book embeddings.

Gradio: Framework for creating the interactive web-based dashboard.


# Dataset and Data Preprocessing
Dataset: The app uses the 7K Books Dataset by Dylan Castillo from Kaggle. It contains metadata and descriptions for approximately 7,000 books (https://www.kaggle.com/datasets/dylanjcastillo/7k-books-with-metadata)

Data cleaning: Remove books with no or short descriptions and duplicates, normalizing text, and handling special characters 

Zero-Shot Text Classification: Applied the zero-shot classifier to categorize books into predefined genre and category.

Sentiment Analysis: Applied the fine-tuned model to extract emotions (e.g., positive, negative, neutral) from book descriptions to enhance recommendation context.


# Vector Database Creation
Building the Vector Database: Created a vector database using Weaviate to store embeddings of book descriptions, enabling semantic search.

Text Splitting: Used LangChain's CharacterTextSplitter to split book descriptions into smaller chunks for efficient embedding and storage.


# Book Recommendation with Vector Search
Getting Book Recommendations Using Vector Search: Implemented a recommendation system by querying the vector database with user input, retrieving books with similar semantic content based on vector embeddings.

Build web UI using Gradio

Deploy the Gradio app to HuggingFace Space (https://huggingface.co/spaces/eneon12345/book-recommender)
    - Notice: since the app is a demo running on free cpu, it may enter sleep mode after period of inactivity. In this case, please be patient with the app as it rebuilds.

