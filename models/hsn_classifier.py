"""
HSN Semantic Retrieval Engine
=============================

WHY RETRIEVAL INSTEAD OF CLASSIFICATION:
- A classifier requires a fixed label set. India's HSN taxonomy has ~18,000+ codes.
  Training a classifier on 18k classes demands enormous labeled data per class,
  frequent retraining when codes change, and collapses on unseen product descriptions.
- Semantic retrieval treats HSN codes as a searchable corpus. A new product description
  is embedded and compared against pre-embedded HSN descriptions via cosine similarity.
  This generalizes to unseen inputs without retraining.
- Retrieval is also explainable: we can show the matched description and its similarity
  score. A classifier only outputs a class index with no interpretable reasoning.

IMPLEMENTATION NOTE:
- This uses TF-IDF vectorization as the embedding layer. TF-IDF captures term-frequency
  patterns and is a deterministic, dependency-light approach that runs on any Python version.
- For production with better semantic understanding, swap `TFIDFRetriever` for
  `SentenceTransformer('all-MiniLM-L6-v2')` embeddings. The `.predict()` interface
  stays identical — only the embedding backend changes.
- TF-IDF limitations: does not handle synonyms or paraphrasing as well as transformer
  embeddings (e.g., "cellular device" won't match "mobile phone" on TF-IDF alone).
  We mitigate this by enriching master descriptions with synonym expansions.

UPGRADE PATH:
- Swap TF-IDF with SentenceTransformer embeddings when torch is installable.
- For 18k+ codes, swap the numpy array with a vector DB (Pinecone, Milvus, Qdrant).
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class HSNSemanticRetriever:
    CONFIDENCE_THRESHOLD = 0.20  # TF-IDF cosine scores are lower than dense embeddings

    def __init__(self):
        # HSN master data with synonym-enriched descriptions for TF-IDF coverage.
        self._hsn_records = [
            {"code": "8517", "description": "smartphone mobile phone cellular device touchscreen handset cell"},
            {"code": "8471", "description": "laptop computer notebook personal computing device PC desktop"},
            {"code": "9403", "description": "wooden chair table office furniture seating desk cabinet"},
            {"code": "6109", "description": "cotton t-shirt tshirt knitted clothing apparel garment shirt"},
            {"code": "9983", "description": "software development IT consulting information technology service SaaS"},
            {"code": "8443", "description": "printer printing machine inkjet laserjet office equipment scanner"},
            {"code": "0402", "description": "milk cream concentrated sweetened dairy product butter cheese"},
            {"code": "3004", "description": "medicine pharmaceutical drug tablet capsule healthcare medical"},
            {"code": "8703", "description": "car automobile motor vehicle sedan SUV hatchback transport"},
            {"code": "7308", "description": "steel iron rod bar structural metal fabrication construction"},
        ]

        self._descriptions = [r["description"] for r in self._hsn_records]
        self._codes = [r["code"] for r in self._hsn_records]

        # Fit TF-IDF on the master descriptions
        self._vectorizer = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),  # unigrams + bigrams for phrase matching
            stop_words="english",
        )
        self._master_tfidf = self._vectorizer.fit_transform(self._descriptions)

    def predict(self, product_name: str) -> dict:
        query_tfidf = self._vectorizer.transform([product_name])
        similarities = cosine_similarity(query_tfidf, self._master_tfidf)[0]

        best_idx = int(np.argmax(similarities))
        confidence = float(similarities[best_idx])
        matched_desc = self._descriptions[best_idx]
        fallback = confidence < self.CONFIDENCE_THRESHOLD

        return {
            "code": self._codes[best_idx] if not fallback else None,
            "confidence": round(confidence, 4),
            "matched_description": matched_desc,
            "fallback_required": fallback,
        }
