from sklearn.feature_extraction.text import TfidfVectorizer

# Sample documents
documents = [
    "Retrieval Augmented Generation RAG",
    "Natural language processing NLP",
    "Machine learning and AI"
]

# Create vectorizer and fit
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(documents)

# Your print statements
print(f"Shape of TF-IDF Matrix: {tfidf_matrix.shape}")
print(vectorizer)
print("\nAll column names (features):")
print(vectorizer.get_feature_names_out())
