# src/keyword_dictionary.py

import os
import json
import re
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def normalize_text(text: str) -> str:
    """
    將英文文字做簡單正規化。
    目的：
    - 統一大小寫
    - 移除多餘符號
    - 保留英文字母、空白與連字號
    """
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z\s\-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


class KeywordDictionaryBuilder:
    """
    根據資料集建立 condition-based keyword dictionary。

    核心概念：
    - 每個 condition 都有一批 Clinical Notes。
    - 使用 TF-IDF 找出每個 condition 中比較具代表性的詞。
    - 產出每個 condition 的 top keywords。
    """

    def __init__(
        self,
        top_n: int = 40,
        ngram_range: Tuple[int, int] = (1, 3),
        max_features: int = 5000
    ):
        """
        Parameters
        ----------
        top_n:
            每個 condition 要保留幾個關鍵詞。

        ngram_range:
            (1, 3) 代表會同時抓 unigram, bigram, trigram。
            例如：
            - vomiting
            - ear discharge
            - skin irritation

        max_features:
            TF-IDF 最多保留多少特徵詞。
        """
        self.top_n = top_n
        self.ngram_range = ngram_range
        self.max_features = max_features

        self.stop_words = self._build_stop_words()

    def _build_stop_words(self) -> List[str]:
        """
        自訂停用詞。

        這些詞在 Clinical Notes 裡很常見，
        但對判斷 condition 幫助不大，所以排除。
        """
        stop_words = {
            # general English stopwords
            "a", "an", "the", "and", "or", "but", "if", "then", "has",
            "is", "are", "was", "were", "be", "been", "being",
            "to", "of", "in", "on", "for", "with", "without",
            "by", "from", "as", "at", "this", "that", "these", "those",
            "it", "its", "he", "she", "they", "them", "his", "her",
            "not", "no", "yes", "may", "might", "can", "could", "should",

            # generic clinical words
            "noted", "suspected", "diagnosed", "observed", "shows",
            "showing", "signs", "sign", "history", "patient", "case",
            "clinical", "exam", "examination", "check", "monitor",
            "recommended", "consider", "rule", "out", "rule out", "discussed",
            "presented", "reported", "treated", "treatment", "therapy",
            "possible", "likely", "chronic", "acute", "mild", "severe",

            # animal words that are too broad
            "dog", "dogs", "canine", "cat", "cats", "feline",
            "puppy", "kitten", "pet", "animal",

            # common vague words
            "normal", "abnormal", "condition", "conditions",
            "issue", "issues", "problem", "problems"

            # additional noisy words from dictionary results
            "has", "have", "had", "having",
            "found", "coming", "hours", "less",
            "under", "over", "around", "near",
            "seems", "seem", "looks", "look",
            "gets", "get", "got",
            "one", "two", "three",
            "also", "often", "sometimes",
        }

        return sorted(list(stop_words))

    def build_from_dataframe(
        self,
        df: pd.DataFrame,
        text_col: str = "clean_text",
        condition_col: str = "condition"
    ) -> pd.DataFrame:
        """
        從 DataFrame 建立 condition keyword dictionary。

        Parameters
        ----------
        df:
            通常會使用 knowledge_base_clinical_notes.csv 讀進來的 DataFrame。

        text_col:
            文字欄位，預設使用 clean_text。

        condition_col:
            condition 欄位。

        Returns
        -------
        dictionary_df:
            包含 condition, keyword, score, rank 的 DataFrame。
        """
        if text_col not in df.columns:
            raise ValueError(f"DataFrame 缺少文字欄位：{text_col}")

        if condition_col not in df.columns:
            raise ValueError(f"DataFrame 缺少 condition 欄位：{condition_col}")

        working_df = df.copy()
        working_df[text_col] = working_df[text_col].apply(normalize_text)

        # 每個 condition 合併成一份大文件
        grouped = (
            working_df
            .groupby(condition_col)[text_col]
            .apply(lambda texts: " ".join(texts))
            .reset_index()
        )

        conditions = grouped[condition_col].tolist()
        condition_documents = grouped[text_col].tolist()

        vectorizer = TfidfVectorizer(
            stop_words=self.stop_words,
            ngram_range=self.ngram_range,
            max_features=self.max_features,
            token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z\-]{2,}\b",
            sublinear_tf=True
        )

        tfidf_matrix = vectorizer.fit_transform(condition_documents)
        feature_names = np.array(vectorizer.get_feature_names_out())

        rows = []

        for condition_idx, condition in enumerate(conditions):
            scores = tfidf_matrix[condition_idx].toarray().flatten()

            # 分數由高到低排序
            top_indices = scores.argsort()[::-1]

            rank = 1

            for feature_idx in top_indices:
                score = float(scores[feature_idx])

                if score <= 0:
                    continue

                keyword = feature_names[feature_idx]

                rows.append({
                    "condition": condition,
                    "keyword": keyword,
                    "score": score,
                    "rank": rank
                })

                rank += 1

                if rank > self.top_n:
                    break

        dictionary_df = pd.DataFrame(rows)

        return dictionary_df

    def save_dictionary(
        self,
        dictionary_df: pd.DataFrame,
        output_csv_path: str,
        output_json_path: str = None
    ):
        """
        儲存 dictionary 成 CSV 和 JSON。
        """
        os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)

        dictionary_df.to_csv(output_csv_path, index=False, encoding="utf-8-sig")

        if output_json_path is not None:
            grouped_dict = {}

            for condition, group in dictionary_df.groupby("condition"):
                grouped_dict[condition] = group["keyword"].tolist()

            with open(output_json_path, "w", encoding="utf-8") as f:
                json.dump(grouped_dict, f, ensure_ascii=False, indent=2)


class DatasetKeywordExpander:
    """
    使用資料集建立出來的 keyword dictionary 來補強 query。

    流程：
    1. 讀取 condition_keyword_dictionary_clinical.csv。
    2. 檢查 query 裡面是否出現某些 dictionary keywords。
    3. 推測 query 比較接近哪個 condition。
    4. 補上該 condition 的代表性 keyword，讓 retrieval 更穩。
    """

    def __init__(
        self,
        dictionary_csv_path: str,
        top_keywords_per_condition: int = 8
    ):
        if not os.path.exists(dictionary_csv_path):
            raise FileNotFoundError(f"找不到 keyword dictionary：{dictionary_csv_path}")

        self.dictionary_csv_path = dictionary_csv_path
        self.top_keywords_per_condition = top_keywords_per_condition

        self.dictionary_df = pd.read_csv(dictionary_csv_path)

        required_columns = ["condition", "keyword", "score", "rank"]

        for col in required_columns:
            if col not in self.dictionary_df.columns:
                raise ValueError(f"dictionary CSV 缺少欄位：{col}")

        self.condition_keywords = self._build_condition_keywords()

    def _build_condition_keywords(self) -> Dict[str, List[str]]:
        """
        將 dictionary 轉成：
        {
            "Skin Irritations": ["skin", "itching", ...],
            ...
        }
        """
        condition_keywords = {}

        for condition, group in self.dictionary_df.groupby("condition"):
            group_sorted = group.sort_values("rank")
            keywords = group_sorted["keyword"].tolist()
            condition_keywords[condition] = keywords

        return condition_keywords

    def score_conditions(self, query_text: str) -> Dict[str, float]:
        """
        根據 query 與各 condition keywords 的重疊程度，替每個 condition 打分數。

        注意：
        這不是分類器，只是 lightweight keyword matching。
        """
        query_norm = normalize_text(query_text)

        condition_scores = {}

        for condition, keywords in self.condition_keywords.items():
            score = 0.0

            for keyword in keywords:
                keyword_norm = normalize_text(keyword)

                if not keyword_norm:
                    continue

                # 如果 keyword 出現在 query 裡，就加分
                if keyword_norm in query_norm:
                    score += 1.0

            condition_scores[condition] = score

        return condition_scores

    def get_best_conditions(
        self,
        query_text: str,
        max_conditions: int = 1
    ) -> List[Tuple[str, float]]:
        """
        回傳分數最高的 condition。
        """
        condition_scores = self.score_conditions(query_text)

        sorted_conditions = sorted(
            condition_scores.items(),
            key=lambda item: item[1],
            reverse=True
        )

        # 只保留分數大於 0 的 condition
        sorted_conditions = [
            (condition, score)
            for condition, score in sorted_conditions
            if score > 0
        ]

        return sorted_conditions[:max_conditions]

    def expand_query(
        self,
        query_text: str,
        max_conditions: int = 1,
        top_keywords: int = 8
    ) -> Tuple[str, List[Tuple[str, float]], List[str]]:
        """
        根據資料集 keyword dictionary 補強 query。

        Parameters
        ----------
        query_text:
            通常是中文翻英文後，再加上 manual keyword enhancement 的 query。

        max_conditions:
            最多根據幾個 condition 補強。
            建議先用 1，避免補太多造成 query 污染。

        top_keywords:
            每個 condition 最多補幾個 keywords。

        Returns
        -------
        expanded_query:
            補強後 query。

        matched_conditions:
            被判定最相關的 condition 與分數。

        added_keywords:
            實際補上的 keywords。
        """
        matched_conditions = self.get_best_conditions(
            query_text=query_text,
            max_conditions=max_conditions
        )

        if not matched_conditions:
            return query_text, [], []

        query_norm = normalize_text(query_text)

        added_keywords = []

        for condition, _score in matched_conditions:
            keywords = self.condition_keywords.get(condition, [])

            for keyword in keywords[:top_keywords]:
                keyword_norm = normalize_text(keyword)

                if not keyword_norm:
                    continue

                # query 已經有的詞不要重複補
                if keyword_norm not in query_norm and keyword not in added_keywords:
                    added_keywords.append(keyword)

        if added_keywords:
            keyword_text = ", ".join(added_keywords)

            expanded_query = (
                f"{query_text} "
                f"Dataset-driven related keywords: {keyword_text}."
            )

            return expanded_query, matched_conditions, added_keywords

        return query_text, matched_conditions, []