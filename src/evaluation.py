# src/evaluation.py

import pandas as pd
from tqdm import tqdm


def evaluate_retrieval(
    retriever,
    query_df,
    top_k: int = 5,
    sample_size=None,
    use_animal_filter: bool = False
):
    """
    Retrieval Evaluation

    使用 Owner Observation 作為 query，
    使用 Clinical Notes 作為 knowledge base。

    評估邏輯：
    如果 query 的 true condition 出現在 retrieved top-k documents 的 condition 裡，
    就代表 retrieval 命中。

    Metrics:
    - Hit@1: Top 1 是否命中正確 condition
    - Hit@3: Top 3 是否包含正確 condition
    - Hit@5: Top 5 是否包含正確 condition
    - MRR: Mean Reciprocal Rank，正確 condition 越早出現分數越高

    Parameters
    ----------
    retriever:
        PetHealthRetriever 物件。

    query_df:
        query_owner_observations.csv 讀入後的 DataFrame。

    top_k:
        retrieval 要取回幾筆資料。

    sample_size:
        如果想先快速測試，可以設定 sample_size=100。
        如果要跑完整資料，設為 None。

    use_animal_filter:
        是否根據英文 query 估計 dog/cat，並使用 animal-aware reranking。
        目前建議先用 False，讓 evaluation 聚焦於 condition retrieval。
    """

    eval_df = query_df.copy()

    if sample_size is not None:
        eval_df = eval_df.sample(
            n=sample_size,
            random_state=42
        ).reset_index(drop=True)

    hit_at_1 = 0
    hit_at_3 = 0
    hit_at_5 = 0
    reciprocal_ranks = []

    detailed_results = []

    for _, row in tqdm(eval_df.iterrows(), total=len(eval_df)):
        query_text = row["clean_text"]
        true_condition = row["condition"]

        animal_filter = None

        if use_animal_filter:
            animal_filter = detect_animal_from_english_query(query_text)

        retrieved_docs = retriever.retrieve(
            query=query_text,
            top_k=top_k,
            animal_filter=animal_filter
        )

        retrieved_conditions = [doc["condition"] for doc in retrieved_docs]

        # Hit@1
        if true_condition in retrieved_conditions[:1]:
            hit_at_1 += 1

        # Hit@3
        if true_condition in retrieved_conditions[:3]:
            hit_at_3 += 1

        # Hit@5
        if true_condition in retrieved_conditions[:5]:
            hit_at_5 += 1

        # MRR
        correct_rank = None

        for rank, condition in enumerate(retrieved_conditions, start=1):
            if condition == true_condition:
                correct_rank = rank
                break

        if correct_rank is not None:
            reciprocal_ranks.append(1 / correct_rank)
        else:
            reciprocal_ranks.append(0)

        detailed_results.append({
            "query": query_text,
            "true_condition": true_condition,
            "animal_filter": animal_filter,
            "top1_condition": retrieved_conditions[0] if retrieved_conditions else None,
            "retrieved_conditions": retrieved_conditions,
            "correct_rank": correct_rank,
            "hit_at_1": true_condition in retrieved_conditions[:1],
            "hit_at_3": true_condition in retrieved_conditions[:3],
            "hit_at_5": true_condition in retrieved_conditions[:5],
            "top1_text": retrieved_docs[0]["text"] if retrieved_docs else None,
            "top1_score": retrieved_docs[0]["score"] if retrieved_docs else None,
        })

    n = len(eval_df)

    metrics = {
        "num_queries": n,
        "Hit@1": hit_at_1 / n,
        "Hit@3": hit_at_3 / n,
        "Hit@5": hit_at_5 / n,
        "MRR": sum(reciprocal_ranks) / n
    }

    detailed_df = pd.DataFrame(detailed_results)

    return metrics, detailed_df


def detect_animal_from_english_query(query_text: str):
    """
    從英文 Owner Observation 中簡單判斷 animal type。

    回傳：
    - "dog"
    - "cat"
    - None

    注意：
    這只是 evaluation 時的簡單輔助。
    如果不想讓 animal reranking 影響 evaluation，可以 use_animal_filter=False。
    """

    text = str(query_text).lower()

    dog_terms = [
        "dog", "dogs", "canine", "canines", "puppy", "puppies"
    ]

    cat_terms = [
        "cat", "cats", "feline", "felines", "kitten", "kittens"
    ]

    if any(term in text for term in dog_terms):
        return "dog"

    if any(term in text for term in cat_terms):
        return "cat"

    return None