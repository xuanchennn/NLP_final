# src/retriever.py

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class PetHealthRetriever:
    """
    Pet Health RAG Retriever

    功能：
    1. 讀取 Clinical Notes 作為 RAG knowledge base。
    2. 使用 embedding model 將 Clinical Notes 轉成向量。
    3. 使用 cosine similarity 找出與 query 最相似的文件。
    4. 支援 soft animal reranking：
       - 如果 query 是 dog，dog/canine/puppy 文件加分
       - cat/feline/kitten 文件扣分
       - 不直接刪除資料，避免正確 condition 被硬排除
    """

    def __init__(
        self,
        kb_df,
        embedding_model_name: str = "BAAI/bge-m3",
        device: str = None
    ):
        self.kb_df = kb_df.reset_index(drop=True)
        self.embedding_model_name = embedding_model_name

        print(f"Loading embedding model: {embedding_model_name}")

        if device is None:
            self.embedder = SentenceTransformer(embedding_model_name)
        else:
            self.embedder = SentenceTransformer(embedding_model_name, device=device)

        self.documents = self._build_documents()
        self.doc_embeddings = None

    def _build_documents(self):
        """
        將 Clinical Notes DataFrame 轉換成 document list。
        """
        documents = []

        for idx, row in self.kb_df.iterrows():
            doc = {
                "id": int(idx),
                "text": row["clean_text"],
                "condition": row["condition"],
                "record_type": row["record_type"]
            }
            documents.append(doc)

        return documents

    def build_index(self, batch_size: int = 32):
        """
        建立向量索引。
        """
        texts = [doc["text"] for doc in self.documents]

        print("Encoding knowledge base documents...")

        embeddings = self.embedder.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=True
        )

        self.doc_embeddings = np.array(embeddings)

        print("Index built successfully.")
        print("Number of documents:", len(self.documents))
        print("Embedding shape:", self.doc_embeddings.shape)

    def _detect_doc_animal(self, doc_text: str):
        """
        根據 Clinical Note 文字判斷文件是 dog / cat / unknown。
        """
        text = str(doc_text).lower()

        dog_terms = [
            "dog",
            "dogs",
            "canine",
            "canines",
            "puppy",
            "puppies"
        ]

        cat_terms = [
            "cat",
            "cats",
            "feline",
            "felines",
            "kitten",
            "kittens"
        ]

        if any(term in text for term in dog_terms):
            return "dog"

        if any(term in text for term in cat_terms):
            return "cat"

        return "unknown"

    def _animal_score_adjustment(self, doc_text: str, animal_filter: str):
        """
        soft animal reranking 分數調整。

        如果 query 是 dog：
        - dog 文件加分
        - cat 文件扣分
        - unknown 不動

        如果 query 是 cat：
        - cat 文件加分
        - dog 文件扣分
        - unknown 不動
        """
        if animal_filter is None:
            return 0.0

        doc_animal = self._detect_doc_animal(doc_text)

        # 這兩個值可以之後做實驗調整
        # 調低物種加權，避免 animal bonus 過度影響語意相似度，導致檢索結果偏離真正的症狀類別。
            # same_animal_bonus = 0.03、different_animal_penalty = -0.05
        same_animal_bonus = 0.005
        different_animal_penalty = -0.02

        if doc_animal == animal_filter:
            return same_animal_bonus

        if doc_animal != "unknown" and doc_animal != animal_filter:
            return different_animal_penalty

        return 0.0

    def retrieve(self, query: str, top_k: int = 5, animal_filter: str = None):
        """
        根據 query 找出最相關的 top-k Clinical Notes。

        Parameters
        ----------
        query:
            使用者問題，通常是英文 query。

        top_k:
            要取回幾筆最相關的資料。

        animal_filter:
            可選：
            - None
            - "dog"
            - "cat"

        Returns
        -------
        results:
            list of retrieved documents。
        """
        if self.doc_embeddings is None:
            raise RuntimeError("請先執行 build_index() 建立向量索引。")

        query_embedding = self.embedder.encode(
            [query],
            normalize_embeddings=True
        )

        similarities = cosine_similarity(
            query_embedding,
            self.doc_embeddings
        )[0]

        scored_docs = []

        for idx, semantic_score in enumerate(similarities):
            doc = self.documents[idx].copy()

            animal_adjustment = self._animal_score_adjustment(
                doc_text=doc["text"],
                animal_filter=animal_filter
            )

            final_score = float(semantic_score + animal_adjustment)

            doc["semantic_score"] = float(semantic_score)
            doc["animal_adjustment"] = float(animal_adjustment)
            doc["score"] = final_score
            doc["detected_doc_animal"] = self._detect_doc_animal(doc["text"])

            scored_docs.append(doc)

        # 用 final_score 排序
        scored_docs = sorted(
            scored_docs,
            key=lambda x: x["score"],
            reverse=True
        )

        results = []

        for rank, doc in enumerate(scored_docs[:top_k], start=1):
            doc["rank"] = rank
            results.append(doc)

        return results