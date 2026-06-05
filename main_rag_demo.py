# main_rag_demo.py

import os
import torch
import pandas as pd

from src.retriever import PetHealthRetriever
from src.query_translator import QueryTranslator


def print_retrieved_docs(retrieved_docs):
    """
    印出 retrieval 結果。
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

def main():
    """
    中文輸入 → 英文翻譯 → 英文 RAG retrieval 的 demo 主程式。

    目前這個版本先完成：
    1. Query translation
    2. Embedding retrieval
    3. Top-k documents output

    之後再接 LLM 生成回答。
    """

    # 1. 讀取前處理後的 Clinical Notes
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

    # 2. 建立 Retriever
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("\nUsing device:", device)

    retriever = PetHealthRetriever(
        kb_df=kb_df,
        embedding_model_name="BAAI/bge-m3",
        device=device
    )

    retriever.build_index(batch_size=32)

    # 3. 建立中文 → 英文翻譯器
    translator = QueryTranslator(
        model_name="Helsinki-NLP/opus-mt-zh-en"
    )

    # 4. 中文使用者問題
    #user_query_zh = "我家的老柯基身上長了一顆像疣的東西，醫生說可能是正常老化，但我還是有點擔心。"

    # 你也可以改成以下測試問題：
    user_query_zh = "我的狗一直抓耳朵，而且耳朵有臭味。"
    # user_query_zh = "我的貓吃完飯後一直吐。"
    # user_query_zh = "我的狗走路怪怪的，好像腳會痛。"
    # user_query_zh = "我發現狗狗身上有小蟲，而且牠一直抓癢。"

    print("\n" + "=" * 80)
    print("Original Chinese Query")
    print("=" * 80)
    print(user_query_zh)

    # 5. 中文 query 翻成英文，並加入寵物健康關鍵字補強
    translated_query_en = translator.translate_and_enhance(user_query_zh)


    print("\n" + "=" * 80)
    print("Structured / Enhanced English Query")    
    print("=" * 80)
    print(translated_query_en)

    # 6. 使用英文 query 檢索英文 Clinical Notes
    animal_filter = detect_animal_from_chinese_query(user_query_zh)

    print("\n" + "=" * 80)
    print("Detected Animal Filter")
    print("=" * 80)
    print(animal_filter)

    retrieved_docs = retriever.retrieve(
        query=translated_query_en,
        top_k=5,
        animal_filter=animal_filter
    )

    # 7. 印出檢索結果
    print_retrieved_docs(retrieved_docs)

    # 8. 額外印出 Top-k condition，方便你觀察是否合理
    print("\n" + "=" * 80)
    print("Retrieved Conditions")
    print("=" * 80)

    for doc in retrieved_docs:
        print(f"Rank {doc['rank']}: {doc['condition']} | Score: {doc['score']:.4f}")


if __name__ == "__main__":
    main()