# src/generator.py

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class PetHealthGenerator:
    """
    Pet Health RAG Generator

    功能：
    1. 接收使用者中文問題
    2. 接收 RAG retrieved documents
    3. 組成 prompt
    4. 使用 LLM 生成繁體中文回答

    注意：
    本系統是寵物健康資訊輔助，不是獸醫診斷系統。
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        device: str = None
    ):
        """
        Parameters
        ----------
        model_name:
            LLM 模型名稱。
            本機筆電建議先用 Qwen/Qwen2.5-1.5B-Instruct。
            GPU 環境可以改用 Qwen/Qwen2.5-3B-Instruct。

        device:
            "cuda" 或 "cpu"。
            若不指定，程式會自動判斷。
        """

        self.model_name = model_name

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"Loading LLM: {model_name}")
        print(f"LLM device: {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )

        if self.device == "cuda":
            torch_dtype = torch.float16
        else:
            torch_dtype = torch.float32

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map="auto" if self.device == "cuda" else None,
            trust_remote_code=True
        )

        if self.device == "cpu":
            self.model.to(self.device)

        self.model.eval()

        print("LLM loaded successfully.")

    def build_context_from_docs(self, retrieved_docs) -> str:
        """
        將 retrieved documents 整理成 prompt context。
        """

        context_blocks = []

        for doc in retrieved_docs:
            block = f"""
[Document {doc.get('rank', '')}]
Condition: {doc.get('condition', '')}
Clinical Note: {doc.get('text', '')}
Retrieval Score: {doc.get('score', '')}
"""
            context_blocks.append(block.strip())

        context_text = "\n\n".join(context_blocks)

        return context_text

    def build_rag_prompt(
        self,
        user_query_zh: str,
        enhanced_query_en: str,
        retrieved_docs
    ) -> str:
        """
        建立 With RAG prompt。

        prompt 設計重點：
        1. 明確要求不能診斷
        2. 必須根據 retrieved documents
        3. 回答繁體中文
        4. 回答要有固定結構
        """

        context_text = self.build_context_from_docs(retrieved_docs)

        prompt = f"""
你是一個謹慎的「寵物健康資訊輔助系統」，你的任務是根據檢索到的臨床筆記，協助飼主理解可能相關的健康資訊。

請嚴格遵守以下規則：
1. 你不能宣稱自己正在診斷疾病。
2. 你不能取代獸醫。
3. 你只能根據 Retrieved Clinical Notes 提供資訊。
4. 如果資料不足，請明確說「目前資料不足以判斷」。
5. 不要編造 Retrieved Clinical Notes 沒有支持的疾病名稱。
6. 回答請使用繁體中文。
7. 回答語氣要清楚、謹慎、飼主友善。
8. 若症狀持續、惡化、疼痛、流血、精神食慾下降、呼吸異常或快速變化，請提醒應諮詢獸醫。

使用者原始問題：
{user_query_zh}

系統轉換後的英文檢索 query：
{enhanced_query_en}

Retrieved Clinical Notes:
{context_text}

請根據以上資料，用以下格式回答：

一、可能相關的健康類別：
二、為什麼這些資料可能相關：
三、飼主可以觀察的重點：
四、什麼情況下建議就醫：
五、安全提醒：
"""
        return prompt.strip()

    def build_without_rag_prompt(self, user_query_zh: str) -> str:
        """
        建立 Without RAG baseline prompt。

        這個版本不提供 retrieved documents，
        可用來和 With RAG 比較。
        """

        prompt = f"""
你是一個謹慎的「寵物健康資訊輔助系統」，你的任務是協助飼主理解寵物症狀可能代表的健康資訊。

請嚴格遵守以下規則：
1. 你不能宣稱自己正在診斷疾病。
2. 你不能取代獸醫。
3. 回答請使用繁體中文。
4. 若資料不足，請說明不確定性。
5. 回答語氣要清楚、謹慎、飼主友善。
6. 若症狀持續、惡化、疼痛、流血、精神食慾下降、呼吸異常或快速變化，請提醒應諮詢獸醫。

使用者問題：
{user_query_zh}

請用以下格式回答：

一、可能相關的健康類別：
二、可能原因或相關資訊：
三、飼主可以觀察的重點：
四、什麼情況下建議就醫：
五、安全提醒：
"""
        return prompt.strip()

    def generate_text(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.3,
        top_p: float = 0.9
    ) -> str:
        """
        使用 LLM 生成文字。
        """

        messages = [
            {
                "role": "system",
                "content": "你是一個謹慎的寵物健康資訊輔助系統，不能取代獸醫診斷。"
            },
            {
                "role": "user",
                "content": prompt
            }
        ]

        input_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=4096
        )

        inputs = {key: value.to(self.model.device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=self.tokenizer.eos_token_id
            )

        decoded = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )

        # 嘗試只取 assistant 後面的回答
        if "assistant" in decoded:
            answer = decoded.split("assistant")[-1].strip()
        else:
            # 若無 assistant 標記，移除原 prompt，保留新生成內容
            answer = decoded.replace(input_text, "").strip()

        return answer

    def generate_with_rag(
        self,
        user_query_zh: str,
        enhanced_query_en: str,
        retrieved_docs,
        max_new_tokens: int = 512
    ) -> str:
        """
        With RAG：
        根據 retrieved documents 生成回答。
        """

        prompt = self.build_rag_prompt(
            user_query_zh=user_query_zh,
            enhanced_query_en=enhanced_query_en,
            retrieved_docs=retrieved_docs
        )

        answer = self.generate_text(
            prompt=prompt,
            max_new_tokens=max_new_tokens
        )

        return answer

    def generate_without_rag(
        self,
        user_query_zh: str,
        max_new_tokens: int = 512
    ) -> str:
        """
        Without RAG：
        不提供 retrieved documents，直接讓 LLM 回答。
        可作為 baseline。
        """

        prompt = self.build_without_rag_prompt(
            user_query_zh=user_query_zh
        )

        answer = self.generate_text(
            prompt=prompt,
            max_new_tokens=max_new_tokens
        )

        return answer