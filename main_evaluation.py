# main_evaluation.py

import os
import torch
import pandas as pd

from src.retriever import PetHealthRetriever
from src.evaluation import evaluate_retrieval


def print_metrics(metrics: dict):
    """
    印出 evaluation metrics。
    """
    print("\n" + "=" * 80)
    print("Retrieval Evaluation Metrics")
    print("=" * 80)

    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")


def main():
    """
    Retrieval Evaluation 主程式。

    評估方法：
    - Knowledge Base: Clinical Notes
    - Query Set: Owner Observation
    - Ground Truth: condition label

    指標：
    - Hit@1
    - Hit@3
    - Hit@5
    - MRR
    """

    # ============================================================
    # 1. 讀取資料
    # ============================================================
    kb_path = os.path.join("outputs", "knowledge_base_clinical_notes.csv")
    query_path = os.path.join("outputs", "query_owner_observations.csv")

    if not os.path.exists(kb_path):
        raise FileNotFoundError(
            f"找不到 {kb_path}，請先執行 python main_preprocess.py"
        )

    if not os.path.exists(query_path):
        raise FileNotFoundError(
            f"找不到 {query_path}，請先執行 python main_preprocess.py"
        )

    kb_df = pd.read_csv(kb_path)
    query_df = pd.read_csv(query_path)

    print("=" * 80)
    print("Loaded Evaluation Data")
    print("=" * 80)
    print(f"Knowledge Base Clinical Notes: {len(kb_df)}")
    print(f"Query Set Owner Observations: {len(query_df)}")

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
    # 3. 執行 Retrieval Evaluation
    # ============================================================

    # 如果你只是想先快速測試，可以改成 sample_size=100
    # 如果要跑完整 evaluation，就用 sample_size=None
    sample_size = None

    # 建議第一版 evaluation 先不要開 animal_filter，
    # 讓分數單純反映 condition retrieval 的效果。
    use_animal_filter = False

    metrics, detailed_df = evaluate_retrieval(
        retriever=retriever,
        query_df=query_df,
        top_k=5,
        sample_size=sample_size,
        use_animal_filter=use_animal_filter
    )

    print_metrics(metrics)

    # ============================================================
    # 4. 儲存 Evaluation 結果
    # ============================================================
    os.makedirs("outputs", exist_ok=True)

    metrics_df = pd.DataFrame([metrics])

    metrics_path = os.path.join("outputs", "retrieval_metrics.csv")
    detailed_path = os.path.join("outputs", "retrieval_detailed_results.csv")

    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    detailed_df.to_csv(detailed_path, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 80)
    print("Evaluation results saved")
    print("=" * 80)
    print(metrics_path)
    print(detailed_path)

    # ============================================================
    # 5. 額外列出一些錯誤案例，方便報告分析
    # ============================================================
    error_df = detailed_df[detailed_df["hit_at_5"] == False]

    print("\n" + "=" * 80)
    print("Example Failed Cases Hit@5 = False")
    print("=" * 80)

    if len(error_df) == 0:
        print("No failed cases in Hit@5.")
    else:
        preview_cols = [
            "query",
            "true_condition",
            "top1_condition",
            "retrieved_conditions"
        ]

        print(error_df[preview_cols].head(10).to_string(index=False))


if __name__ == "__main__":
    main()