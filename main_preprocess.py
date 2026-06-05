# main_preprocess.py

from src.data_preprocessing import (
    load_dataset,
    preprocess_dataset,
    split_for_rag,
    save_processed_data,
    print_dataset_summary
)


def main():
    """
    主程式：執行資料前處理。
    """

    # 你的原始資料集路徑
    csv_path = r"C:\Users\enxua\Desktop\NLP_final\pet-health-symptoms-dataset.csv"

    # 1. 讀取資料
    df = load_dataset(csv_path)

    # 2. 前處理
    processed_df = preprocess_dataset(df)

    # 3. 切分 RAG knowledge base 與 query set
    kb_df, query_df = split_for_rag(processed_df)

    # 4. 印出資料摘要
    print_dataset_summary(processed_df, kb_df, query_df)

    # 5. 儲存處理後資料
    save_processed_data(
        df=processed_df,
        kb_df=kb_df,
        query_df=query_df,
        output_dir="outputs"
    )

    print("前處理完成，檔案已儲存到 outputs/ 資料夾。")


if __name__ == "__main__":
    main()