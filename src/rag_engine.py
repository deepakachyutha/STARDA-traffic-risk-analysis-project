import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class TrafficRAG:
    def __init__(self, kb_path="src/knowledge_base.txt"):
        self.kb_path = kb_path
        self.documents = []
        self.load_knowledge_base()
        
        # Initialize Vectorizer (The "Brain")
        self.vectorizer = TfidfVectorizer(stop_words='english')
        if self.documents:
            self.tfidf_matrix = self.vectorizer.fit_transform(self.documents)
        else:
            self.tfidf_matrix = None

    def load_knowledge_base(self):
        """Loads text and splits it into chunks based on [SECTION] headers."""
        if not os.path.exists(self.kb_path):
            print(f"Warning: Knowledge base not found at {self.kb_path}")
            return

        with open(self.kb_path, "r", encoding="utf-8") as f:
            text = f.read()

        # Split by double newlines or Section headers to create distinct "chunks"
        # We assume your text file has blank lines between paragraphs
        raw_chunks = text.split('\n\n') 
        self.documents = [chunk.strip() for chunk in raw_chunks if len(chunk.strip()) > 10]
        print(f"RAG Engine Loaded: {len(self.documents)} safety protocols indexed.")

    def retrieve(self, query):
        """Finds the text chunk most similar to the user query."""
        if not self.documents or self.tfidf_matrix is None:
            return "Knowledge Base Offline. Please check src/knowledge_base.txt."

        query_vec = self.vectorizer.transform([query])
        
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        
        best_match_idx = np.argmax(similarities)
        best_score = similarities[best_match_idx]

        if best_score < 0.1:
            return "No specific protocol found for these conditions. Exercise general caution."
            
        return self.documents[best_match_idx]

if __name__ == "__main__":
    rag = TrafficRAG()
    print("Test Search (Snow):", rag.retrieve("driving in heavy snow"))
    print("Test Search (Rain):", rag.retrieve("wet road skidding"))