# src/data_preprocessing.py

import re
import os
import pandas as pd


def clean_text(text: str) -> str:
    """
    清理文字內容。

    處理項目：
    1. 轉成字串，避免遇到 NaN 或其他格式報錯。
    2. 移除換行符號。
    3. 移除多餘空白。
    4. 去除前後空白。
    """
    text = str(text)
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


def load_dataset(csv_path: str) -> pd.DataFrame:
    """
    讀取 pet-health-symptoms-dataset.csv。

    預期欄位：
    - text：症狀描述或臨床筆記
    - condition：健康類別
    - record_type：Owner Observation 或 Clinical Notes
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"找不到資料集檔案：{csv_path}")

    df = pd.read_csv(csv_path)

    required_columns = ["text", "condition", "record_type"]

    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"資料集中缺少必要欄位：{col}")

    return df


def preprocess_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    對原始資料做基本前處理。
    """
    df = df.copy()

    # 清理文字欄位
    df["clean_text"] = df["text"].apply(clean_text)

    # 移除空文字資料
    df = df[df["clean_text"].str.len() > 0].reset_index(drop=True)

    # 移除重複資料
    df = df.drop_duplicates(
        subset=["clean_text", "condition", "record_type"]
    ).reset_index(drop=True)

    return df


def split_for_rag(df: pd.DataFrame):
    """
    將資料切成 RAG 需要的兩個部分。

    1. Clinical Notes：
       當作 knowledge base，也就是 RAG 要檢索的資料庫。

    2. Owner Observation：
       當作模擬使用者問題，也可以用來做 retrieval evaluation。
    """
    kb_df = df[df["record_type"] == "Clinical Notes"].reset_index(drop=True)
    query_df = df[df["record_type"] == "Owner Observation"].reset_index(drop=True)

    if len(kb_df) == 0:
        raise ValueError("找不到 Clinical Notes，無法建立 RAG knowledge base。")

    if len(query_df) == 0:
        raise ValueError("找不到 Owner Observation，無法建立 query set。")

    return kb_df, query_df


def save_processed_data(
    df: pd.DataFrame,
    kb_df: pd.DataFrame,
    query_df: pd.DataFrame,
    output_dir: str = "outputs"
):
    """
    將前處理後的資料存起來，方便後續報告與模型使用。
    """
    os.makedirs(output_dir, exist_ok=True)

    df.to_csv(os.path.join(output_dir, "processed_all.csv"), index=False)
    kb_df.to_csv(os.path.join(output_dir, "knowledge_base_clinical_notes.csv"), index=False)
    query_df.to_csv(os.path.join(output_dir, "query_owner_observations.csv"), index=False)


def print_dataset_summary(df: pd.DataFrame, kb_df: pd.DataFrame, query_df: pd.DataFrame):
    """
    印出資料集摘要，這些結果可以放進期末報告 Dataset 章節。
    """
    print("=" * 60)
    print("Dataset Summary")
    print("=" * 60)

    print(f"Total records: {len(df)}")
    print(f"Knowledge base records Clinical Notes: {len(kb_df)}")
    print(f"Query records Owner Observation: {len(query_df)}")

    print("\nRecord type distribution:")
    print(df["record_type"].value_counts())

    print("\nCondition distribution:")
    print(df["condition"].value_counts())

    print("\nMissing values:")
    print(df.isnull().sum())

    print("=" * 60)