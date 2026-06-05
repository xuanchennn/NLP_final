# main_rag_answer_demo.py

import os
import torch
import pandas as pd

from src.retriever import PetHealthRetriever
from src.query_translator import QueryTranslator
from src.generator import PetHealthGenerator


def detect_animal_from_chinese_query(query_zh: str):
    """
    根據中文 query 判斷使用者問的是狗還是貓。

    回傳：
    - "dog"
    - "cat"
    - None
    """

    dog_keywords = [
        "狗", "狗狗", "小狗", "老狗", "老犬", "高齡犬", "幼犬",
        "柯基", "柴犬", "貴賓", "博美", "吉娃娃", "馬爾濟斯",
        "瑪爾濟斯", "約克夏", "臘腸", "法鬥", "法國鬥牛犬",
        "黃金獵犬", "黃金", "拉布拉多", "拉拉", "哈士奇",
        "米克斯", "台灣犬", "雪納瑞", "邊牧", "邊境牧羊犬",
        "德牧", "德國牧羊犬", "秋田", "秋田犬"
    ]

    cat_keywords = [
        "貓", "貓咪", "小貓", "老貓", "高齡貓", "幼貓"
    ]

    for word in dog_keywords:
        if word in query_zh:
            return "dog"

    for word in cat_keywords:
        if word in query_zh:
            return "cat"

    return None


def print_retrieved_docs(retrieved_docs):
    """
    印出 retrieved documents。
    """

    print("\n" + "=" * 80)
    print("Retrieved Documents")
    print("=" * 80)

    for doc in retrieved_docs:
        print(f"Rank: {doc['rank']}")
        print(f"Final Score: {doc['score']:.4f}")

        if "semantic_score" in doc:
            print(f"Semantic Score: {doc['semantic_score']:.4f}")

        if "animal_adjustment" in doc:
            print(f"Animal Adjustment: {doc['animal_adjustment']:.4f}")

        if "detected_doc_animal" in doc:
            print(f"Detected Doc Animal: {doc['detected_doc_animal']}")

        print(f"Condition: {doc['condition']}")
        print(f"Text: {doc['text']}")
        print("-" * 80)


def main():
    """
    完整 RAG Answer Demo：

    中文 query
    → structured / enhanced English query
    → retrieve Clinical Notes
    → LLM 生成繁體中文回答
    """

    # ============================================================
    # 1. 讀取 Clinical Notes Knowledge Base
    # ============================================================
    kb_path = os.path.join("outputs", "knowledge_base_clinical_notes.csv")

    if not os.path.exists(kb_path):
        raise FileNotFoundError(
            f"找不到 {kb_path}，請先執行 python main_preprocess.py"
        )

    kb_df = pd.read_csv(kb_path)

    print("=" * 80)
    print("Loaded Knowledge Base")
    print("=" * 80)
    print(f"Knowledge base size: {len(kb_df)}")
    print("Columns:", kb_df.columns.tolist())

    # ============================================================
    # 2. 建立 Retriever
    # ============================================================
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("\nUsing device:", device)

    retriever = PetHealthRetriever(
        kb_df=kb_df,
        embedding_model_name="BAAI/bge-m3",
        device=device
    )

    retriever.build_index(batch_size=32)

    # ============================================================
    # 3. 建立 Query Translator
    # ============================================================
    translator = QueryTranslator(
        model_name="Helsinki-NLP/opus-mt-zh-en"
    )

    # ============================================================
    # 4. 建立 LLM Generator
    # ============================================================
    # 本機先建議用 1.5B，較不容易爆記憶體。
    # 若到 GPU 裝置，可改成 Qwen/Qwen2.5-3B-Instruct。
    generator = PetHealthGenerator(
        model_name="Qwen/Qwen2.5-1.5B-Instruct",
        device=device
    )

    # ============================================================
    # 5. 中文使用者問題
    # ============================================================
    #user_query_zh = "我的狗一直抓耳朵，而且耳朵有臭味。"

    # 其他可以測：
    #user_query_zh = "我家的老柯基身上長了一顆像疣的東西，醫生說可能是正常老化，但我還是有點擔心。"
    user_query_zh = "我的貓吃完飯後一直吐，精神也不太好。"
    # user_query_zh = "我的狗走路怪怪的，好像後腳會痛。"
    # user_query_zh = "我發現狗狗身上有小蟲，而且牠一直抓癢。"

    print("\n" + "=" * 80)
    print("Original Chinese Query")
    print("=" * 80)
    print(user_query_zh)

    # ============================================================
    # 6. 中文 query → structured / enhanced English query
    # ============================================================
    enhanced_query_en = translator.translate_and_enhance(user_query_zh)

    print("\n" + "=" * 80)
    print("Structured / Enhanced English Query")
    print("=" * 80)
    print(enhanced_query_en)

    # ============================================================
    # 7. Animal filter
    # ============================================================
    animal_filter = detect_animal_from_chinese_query(user_query_zh)

    print("\n" + "=" * 80)
    print("Detected Animal Filter")
    print("=" * 80)
    print(animal_filter)

    # ============================================================
    # 8. RAG Retrieval
    # ============================================================
    retrieved_docs = retriever.retrieve(
        query=enhanced_query_en,
        top_k=5,
        animal_filter=animal_filter
    )

    print_retrieved_docs(retrieved_docs)

    print("\n" + "=" * 80)
    print("Retrieved Conditions")
    print("=" * 80)

    for doc in retrieved_docs:
        print(f"Rank {doc['rank']}: {doc['condition']} | Score: {doc['score']:.4f}")

    # ============================================================
    # 9. With RAG Answer
    # ============================================================
    print("\n" + "=" * 80)
    print("With RAG Answer")
    print("=" * 80)

    rag_answer = generator.generate_with_rag(
        user_query_zh=user_query_zh,
        enhanced_query_en=enhanced_query_en,
        retrieved_docs=retrieved_docs,
        max_new_tokens=512
    )

    print(rag_answer)

    # ============================================================
    # 10. Optional: Without RAG baseline
    # ============================================================
    run_without_rag = True

    if run_without_rag:
        print("\n" + "=" * 80)
        print("Without RAG Answer")
        print("=" * 80)

        no_rag_answer = generator.generate_without_rag(
            user_query_zh=user_query_zh,
            max_new_tokens=512
        )

        print(no_rag_answer)


if __name__ == "__main__":
    main()