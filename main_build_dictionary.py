# main_build_dictionary.py

import os
import pandas as pd

from src.keyword_dictionary import KeywordDictionaryBuilder


def print_top_keywords(dictionary_df: pd.DataFrame, title: str, top_n: int = 10):
    """
    印出每個 condition 的 top keywords，方便檢查結果。
    """
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

    for condition, group in dictionary_df.groupby("condition"):
        print(f"\nCondition: {condition}")
        top_group = group.sort_values("rank").head(top_n)

        for _, row in top_group.iterrows():
            print(f"  Rank {int(row['rank']):02d} | {row['keyword']} | score={row['score']:.4f}")


def main():
    """
    建立 Dataset-driven Keyword Dictionary。

    會分別建立：
    1. Clinical Notes dictionary：
       用於 RAG knowledge base 的 query expansion。

    2. Owner Observation dictionary：
       用於分析飼主常見描述方式，可放在報告裡。
    """

    clinical_path = os.path.join("outputs", "knowledge_base_clinical_notes.csv")
    owner_path = os.path.join("outputs", "query_owner_observations.csv")

    if not os.path.exists(clinical_path):
        raise FileNotFoundError(
            f"找不到 {clinical_path}，請先執行 python main_preprocess.py"
        )

    if not os.path.exists(owner_path):
        raise FileNotFoundError(
            f"找不到 {owner_path}，請先執行 python main_preprocess.py"
        )

    clinical_df = pd.read_csv(clinical_path)
    owner_df = pd.read_csv(owner_path)

    print("=" * 80)
    print("Building Dataset-driven Keyword Dictionaries")
    print("=" * 80)

    print(f"Clinical Notes size: {len(clinical_df)}")
    print(f"Owner Observation size: {len(owner_df)}")

    builder = KeywordDictionaryBuilder(
        top_n=40,
        ngram_range=(1, 3),
        max_features=5000
    )

    # 1. 建立 Clinical Notes Dictionary
    clinical_dictionary_df = builder.build_from_dataframe(
        df=clinical_df,
        text_col="clean_text",
        condition_col="condition"
    )

    clinical_csv_path = os.path.join(
        "outputs",
        "condition_keyword_dictionary_clinical.csv"
    )

    clinical_json_path = os.path.join(
        "outputs",
        "condition_keyword_dictionary_clinical.json"
    )

    builder.save_dictionary(
        dictionary_df=clinical_dictionary_df,
        output_csv_path=clinical_csv_path,
        output_json_path=clinical_json_path
    )

    # 2. 建立 Owner Observation Dictionary
    owner_dictionary_df = builder.build_from_dataframe(
        df=owner_df,
        text_col="clean_text",
        condition_col="condition"
    )

    owner_csv_path = os.path.join(
        "outputs",
        "condition_keyword_dictionary_owner.csv"
    )

    owner_json_path = os.path.join(
        "outputs",
        "condition_keyword_dictionary_owner.json"
    )

    builder.save_dictionary(
        dictionary_df=owner_dictionary_df,
        output_csv_path=owner_csv_path,
        output_json_path=owner_json_path
    )

    print_top_keywords(
        clinical_dictionary_df,
        title="Top Keywords from Clinical Notes Dictionary",
        top_n=10
    )

    print_top_keywords(
        owner_dictionary_df,
        title="Top Keywords from Owner Observation Dictionary",
        top_n=10
    )

    print("\n" + "=" * 80)
    print("Dictionary files saved")
    print("=" * 80)
    print(clinical_csv_path)
    print(clinical_json_path)
    print(owner_csv_path)
    print(owner_json_path)


if __name__ == "__main__":
    main()