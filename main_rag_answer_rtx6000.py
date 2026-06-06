# main_rag_answer_gpu.py

"""
GPU version of Pet Health RAG Answer Demo.

這個檔案是專門給效能較好的裝置，例如 RTX6000Pro，執行完整 RAG + LLM 生成回答用。

流程：
1. 讀取 Clinical Notes knowledge base
2. 建立 embedding retriever
3. 將中文 query 轉成 structured / enhanced English query
4. 使用 RAG retrieve Top-k clinical notes
5. 使用較大的 LLM 生成繁體中文回答
6. 可選擇產生 Without RAG baseline
7. 將結果儲存到 outputs/rag_gpu_answer_result.txt
"""

import os
import argparse
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


def print_gpu_info():
    """
    印出目前 GPU 狀態，方便確認裝置是否真的使用 CUDA。
    """

    print("=" * 80)
    print("GPU / CUDA Information")
    print("=" * 80)

    print("torch.cuda.is_available():", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("CUDA device count:", torch.cuda.device_count())
        print("Current CUDA device:", torch.cuda.current_device())
        print("GPU name:", torch.cuda.get_device_name(0))

        total_memory = torch.cuda.get_device_properties(0).total_memory
        total_memory_gb = total_memory / 1024**3
        print(f"Total GPU memory: {total_memory_gb:.2f} GB")
    else:
        print("CUDA is not available. The program will run on CPU, which may be very slow.")


def print_retrieved_docs(retrieved_docs):
    """
    印出 retrieved documents，包含語意分數、animal reranking 分數與 condition。
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


def build_result_text(
    user_query_zh: str,
    enhanced_query_en: str,
    animal_filter,
    retrieved_docs,
    rag_answer: str,
    no_rag_answer: str = None,
    model_name: str = ""
):
    """
    將執行結果整理成文字，方便存成 txt。
    """

    lines = []

    lines.append("=" * 80)
    lines.append("Pet Health RAG GPU Answer Result")
    lines.append("=" * 80)
    lines.append(f"LLM model: {model_name}")
    lines.append("")
    lines.append("[Original Chinese Query]")
    lines.append(user_query_zh)
    lines.append("")
    lines.append("[Structured / Enhanced English Query]")
    lines.append(enhanced_query_en)
    lines.append("")
    lines.append("[Detected Animal Filter]")
    lines.append(str(animal_filter))
    lines.append("")
    lines.append("[Retrieved Documents]")

    for doc in retrieved_docs:
        lines.append("-" * 80)
        lines.append(f"Rank: {doc['rank']}")
        lines.append(f"Condition: {doc['condition']}")
        lines.append(f"Final Score: {doc['score']:.4f}")

        if "semantic_score" in doc:
            lines.append(f"Semantic Score: {doc['semantic_score']:.4f}")

        if "animal_adjustment" in doc:
            lines.append(f"Animal Adjustment: {doc['animal_adjustment']:.4f}")

        if "detected_doc_animal" in doc:
            lines.append(f"Detected Doc Animal: {doc['detected_doc_animal']}")

        lines.append(f"Clinical Note: {doc['text']}")

    lines.append("")
    lines.append("=" * 80)
    lines.append("[With RAG Answer]")
    lines.append("=" * 80)
    lines.append(rag_answer)

    if no_rag_answer is not None:
        lines.append("")
        lines.append("=" * 80)
        lines.append("[Without RAG Answer]")
        lines.append("=" * 80)
        lines.append(no_rag_answer)

    return "\n".join(lines)


def main():
    """
    GPU RAG Answer Demo 主程式。
    """

    parser = argparse.ArgumentParser(
        description="Run Pet Health RAG + LLM answer generation on GPU."
    )

    parser.add_argument(
        "--query",
        type=str,
        default="我的貓吃完飯後一直吐，精神也不太好。",
        help="Chinese user query."
    )

    parser.add_argument(
        "--llm_model",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help=(
            "LLM model name. "
            "Recommended for RTX6000Pro: Qwen/Qwen2.5-7B-Instruct. "
            "If stable, try Qwen/Qwen2.5-14B-Instruct."
        )
    )

    parser.add_argument(
        "--embedding_model",
        type=str,
        default="BAAI/bge-m3",
        help="Embedding model for retrieval."
    )

    parser.add_argument(
        "--translation_model",
        type=str,
        default="Helsinki-NLP/opus-mt-zh-en",
        help="Chinese-to-English translation model."
    )

    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Number of retrieved documents."
    )

    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Maximum new tokens generated by LLM."
    )

    parser.add_argument(
        "--without_rag",
        action="store_true",
        help="Also generate a Without RAG baseline answer."
    )

    parser.add_argument(
        "--output_path",
        type=str,
        default=os.path.join("outputs", "rag_gpu_answer_result.txt"),
        help="Path to save the result text."
    )

    args = parser.parse_args()

    # ============================================================
    # 0. 檢查 GPU
    # ============================================================
    print_gpu_info()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("\nUsing device:", device)

    # ============================================================
    # 1. 讀取 Clinical Notes Knowledge Base
    # ============================================================
    kb_path = os.path.join("outputs", "knowledge_base_clinical_notes.csv")

    if not os.path.exists(kb_path):
        raise FileNotFoundError(
            f"找不到 {kb_path}。請先執行：python main_preprocess.py"
        )

    kb_df = pd.read_csv(kb_path)

    print("\n" + "=" * 80)
    print("Loaded Knowledge Base")
    print("=" * 80)
    print(f"Knowledge base size: {len(kb_df)}")
    print("Columns:", kb_df.columns.tolist())

    # ============================================================
    # 2. 建立 Retriever
    # ============================================================
    retriever = PetHealthRetriever(
        kb_df=kb_df,
        embedding_model_name=args.embedding_model,
        device=device
    )

    retriever.build_index(batch_size=32)

    # ============================================================
    # 3. 建立 Query Translator
    # ============================================================
    translator = QueryTranslator(
        model_name=args.translation_model
    )

    # ============================================================
    # 4. 建立 LLM Generator
    # ============================================================
    generator = PetHealthGenerator(
        model_name=args.llm_model,
        device=device
    )

    # ============================================================
    # 5. 中文使用者問題
    # ============================================================
    user_query_zh = args.query

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
        top_k=args.top_k,
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
        max_new_tokens=args.max_new_tokens
    )

    print(rag_answer)

    # ============================================================
    # 10. Optional: Without RAG baseline
    # ============================================================
    no_rag_answer = None

    if args.without_rag:
        print("\n" + "=" * 80)
        print("Without RAG Answer")
        print("=" * 80)

        no_rag_answer = generator.generate_without_rag(
            user_query_zh=user_query_zh,
            max_new_tokens=args.max_new_tokens
        )

        print(no_rag_answer)

    # ============================================================
    # 11. 儲存結果
    # ============================================================
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    result_text = build_result_text(
        user_query_zh=user_query_zh,
        enhanced_query_en=enhanced_query_en,
        animal_filter=animal_filter,
        retrieved_docs=retrieved_docs,
        rag_answer=rag_answer,
        no_rag_answer=no_rag_answer,
        model_name=args.llm_model
    )

    with open(args.output_path, "w", encoding="utf-8") as f:
        f.write(result_text)

    print("\n" + "=" * 80)
    print("Result saved")
    print("=" * 80)
    print(args.output_path)


if __name__ == "__main__":
    main()