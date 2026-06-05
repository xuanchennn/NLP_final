# src/query_translator.py

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


class QueryTranslator:
    """
    Query Translation Module

    功能：
    1. 將使用者輸入的繁體中文寵物健康描述翻譯成英文。
    2. 根據中文原句補上寵物健康相關英文關鍵字。
    3. 根據中文犬種名稱補上英文犬種名稱與體型資訊。

    為什麼需要這個模組？
    因為目前 pet-health-symptoms-dataset 的 Clinical Notes 是英文。
    如果使用者直接用中文提問，跨語言檢索效果可能不穩定。
    所以我們先將中文 query 翻成英文，再用英文 query 去檢索英文知識庫。

    為什麼要做關鍵字補強？
    因為一般翻譯模型可能會把寵物醫療詞彙翻錯。
    例如「疣」可能被錯翻成 vinegar，但 retrieval 需要的是 wart / bumps / skin lumps。
    """

    def __init__(self, model_name: str = "Helsinki-NLP/opus-mt-zh-en"):
        """
        Parameters
        ----------
        model_name:
            預設使用 Helsinki-NLP/opus-mt-zh-en 做中文到英文翻譯。
        """
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"Loading translation model: {model_name}")
        print(f"Translation device: {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        self.model.to(self.device)
        self.model.eval()

        print("Translation model loaded successfully.")

    def translate_zh_to_en(self, text: str) -> str:
        """
        將中文文字翻譯成英文。

        Parameters
        ----------
        text:
            使用者輸入的中文寵物健康描述。

        Returns
        -------
        translated_text:
            英文翻譯結果。
        """
        if not text or not text.strip():
            return ""

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256
        )

        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with torch.no_grad():
            generated_tokens = self.model.generate(
                **inputs,
                max_length=256,
                num_beams=4,
                early_stopping=True
            )

        translated_text = self.tokenizer.decode(
            generated_tokens[0],
            skip_special_tokens=True
        )

        return translated_text

    def add_domain_keywords(self, original_zh_text: str, translated_en_text: str) -> str:
        """
        根據中文原句補上寵物健康領域的英文關鍵字。

        這不是翻譯全文，而是補強 retrieval 需要的關鍵症狀詞。

        例如：
        - 「疣」補成 wart-like bumps, skin lumps
        - 「耳朵臭」補成 bad smell from ears, ear infection
        - 「貴賓」補成 Poodle, small dog breed
        - 「黃金獵犬」補成 Golden Retriever, large dog breed
        """

        keyword_map = {
            # =====================================================
            # Skin Irritations 皮膚問題
            # =====================================================
            "疣": "wart-like bumps, skin lumps, skin irritation",
            "一顆一顆": "bumps, lumps",
            "一粒一粒": "bumps, lumps",
            "一塊一塊": "patches, skin lesions",
            "腫塊": "lump, mass, swelling",
            "肉瘤": "skin lump, mass, growth",
            "皮膚": "skin",
            "身上": "skin, body",
            "紅腫": "redness, swelling, skin irritation",
            "掉毛": "hair loss, alopecia",
            "抓癢": "itching, scratching, skin irritation",
            "一直抓": "itching, scratching",
            "過敏": "allergy, skin irritation",
            "紅疹": "rash, skin irritation",
            "脫皮": "flaky skin, skin irritation",
            "皮屑": "dandruff, flaky skin, skin irritation",
            "結痂": "scab, crusting, skin irritation",
            "破皮": "skin wound, broken skin, irritation",
            "流血": "bleeding, wound",
            "發炎": "inflammation",
            "舔腳": "paw licking, skin irritation, allergy",
            "一直舔": "licking, irritation",

            # =====================================================
            # Ear Infections 耳朵感染
            # =====================================================
            "耳朵": "ears, ear infection",
            "耳朵臭": "bad smell from ears, ear infection",
            "耳臭": "bad smell from ears, ear infection",
            "臭味": "bad smell",
            "甩頭": "head shaking, ear infection",
            "耳垢": "ear wax, ear discharge",
            "耳朵紅": "red ears, ear inflammation",
            "抓耳朵": "scratching ears, ear infection",
            "耳朵癢": "itchy ears, ear infection",
            "耳朵分泌物": "ear discharge, ear infection",

            # =====================================================
            # Digestive Issues 消化問題
            # =====================================================
            "吐": "vomiting, digestive issues",
            "嘔吐": "vomiting, digestive issues",
            "拉肚子": "diarrhea, digestive issues",
            "腹瀉": "diarrhea, digestive issues",
            "吃完": "after eating, digestive issues",
            "沒食慾": "loss of appetite, digestive issues",
            "不吃飯": "loss of appetite",
            "軟便": "soft stool, digestive issues",
            "血便": "bloody stool, digestive issues",
            "便便": "stool, feces, digestive issues",
            "肚子痛": "abdominal pain, digestive issues",
            "脹氣": "bloating, gas, digestive issues",

            # =====================================================
            # Mobility Problems 行動問題
            # =====================================================
            "走路": "walking, mobility problems",
            "跛腳": "limping, mobility problems",
            "腳痛": "leg pain, mobility problems",
            "站不起來": "difficulty standing, mobility problems",
            "關節": "joint pain, mobility problems",
            "後腳無力": "weak hind legs, mobility problems",
            "前腳無力": "weak front legs, mobility problems",
            "跳不上去": "difficulty jumping, mobility problems",
            "不想走": "reluctance to walk, mobility problems",
            "走路怪怪": "abnormal gait, mobility problems",
            "走路不穩": "unsteady walking, mobility problems",
            "癱軟": "weakness, mobility problems",

            # =====================================================
            # Parasites 寄生蟲
            # =====================================================
            "小蟲": "fleas, ticks, parasites",
            "跳蚤": "fleas, parasites",
            "壁蝨": "ticks, parasites",
            "蜱蟲": "ticks, parasites",
            "寄生蟲": "parasites",
            "蟲": "parasites, insects",
            "蟲卵": "parasite eggs",
            "一直舔": "licking, irritation, parasites",
            "身上有蟲": "parasites on skin, fleas, ticks",

            # =====================================================
            # Age / animal 年齡與動物種類
            # =====================================================
            "老狗": "senior dog, older dog",
            "老犬": "senior dog, older dog",
            "高齡犬": "senior dog, older dog",
            "幼犬": "puppy, young dog",
            "小狗": "dog, puppy",
            "狗狗": "dog",
            "狗": "dog",
            "貓咪": "cat",
            "貓": "cat",

            # =====================================================
            # Small dog breeds 小型犬
            # =====================================================
            "吉娃娃": "Chihuahua, small dog breed",
            "博美": "Pomeranian, small dog breed",
            "馬爾濟斯": "Maltese, small dog breed",
            "瑪爾濟斯": "Maltese, small dog breed",
            "約克夏": "Yorkshire Terrier, small dog breed",
            "貴賓": "Poodle, small dog breed",
            "玩具貴賓": "Toy Poodle, small dog breed",
            "迷你貴賓": "Miniature Poodle, small dog breed",
            "臘腸": "Dachshund, small dog breed",
            "臘腸狗": "Dachshund, small dog breed",
            "西施": "Shih Tzu, small dog breed",
            "巴哥": "Pug, small dog breed",
            "法鬥": "French Bulldog, small dog breed",
            "法國鬥牛犬": "French Bulldog, small dog breed",
            "比熊": "Bichon Frise, small dog breed",
            "雪納瑞": "Schnauzer, small to medium dog breed",
            "迷你雪納瑞": "Miniature Schnauzer, small dog breed",
            "傑克羅素": "Jack Russell Terrier, small dog breed",
            "柴犬": "Shiba Inu, small to medium dog breed",

            # =====================================================
            # Medium dog breeds 中型犬
            # =====================================================
            "柯基": "Corgi, medium dog breed",
            "威爾斯柯基": "Welsh Corgi, medium dog breed",
            "米格魯": "Beagle, medium dog breed",
            "比格犬": "Beagle, medium dog breed",
            "邊境牧羊犬": "Border Collie, medium dog breed",
            "邊牧": "Border Collie, medium dog breed",
            "喜樂蒂": "Shetland Sheepdog, medium dog breed",
            "可卡": "Cocker Spaniel, medium dog breed",
            "可卡犬": "Cocker Spaniel, medium dog breed",
            "鬥牛犬": "Bulldog, medium dog breed",
            "澳洲牧羊犬": "Australian Shepherd, medium dog breed",
            "沙皮": "Shar Pei, medium dog breed",
            "台灣犬": "Taiwan Dog, medium dog breed",
            "米克斯": "mixed breed dog",
            "混種犬": "mixed breed dog",

            # =====================================================
            # Large dog breeds 大型犬
            # =====================================================
            "黃金獵犬": "Golden Retriever, large dog breed",
            "黃金": "Golden Retriever, large dog breed",
            "拉布拉多": "Labrador Retriever, large dog breed",
            "拉拉": "Labrador Retriever, large dog breed",
            "哈士奇": "Siberian Husky, large dog breed",
            "薩摩耶": "Samoyed, large dog breed",
            "德國牧羊犬": "German Shepherd, large dog breed",
            "德牧": "German Shepherd, large dog breed",
            "阿拉斯加": "Alaskan Malamute, large dog breed",
            "杜賓": "Doberman Pinscher, large dog breed",
            "羅威納": "Rottweiler, large dog breed",
            "伯恩山": "Bernese Mountain Dog, large dog breed",
            "大白熊": "Great Pyrenees, large dog breed",
            "聖伯納": "Saint Bernard, giant dog breed",
            "大丹": "Great Dane, giant dog breed",
            "秋田": "Akita, large dog breed",
            "秋田犬": "Akita, large dog breed",
            "鬆獅": "Chow Chow, medium to large dog breed",
        }

        extra_keywords = []

        for zh_key, en_keywords in keyword_map.items():
            if zh_key in original_zh_text:
                extra_keywords.append(en_keywords)

        # 去除重複補強詞，避免 query 太冗長
        unique_keywords = []

        for item in extra_keywords:
            if item not in unique_keywords:
                unique_keywords.append(item)

        if unique_keywords:
            keyword_text = ", ".join(unique_keywords)

            enhanced_query = (
                f"{translated_en_text} "
                f"Additional pet health keywords: {keyword_text}."
            )

            return enhanced_query

        return translated_en_text

    def build_structured_medical_query(self, original_zh_text: str, translated_en_text: str) -> str:
            """
            建立較乾淨的 structured medical query。

            原因：
            一般翻譯模型可能會把「疣」翻成 wrench，把「柯基」翻成 Corky。
            因此我們根據中文原句抽取可靠的症狀、物種、犬種、年齡資訊，
            建立一個更適合 retrieval 的英文 query。
            """

            query_terms = []

            # =====================================================
            # Animal / species 物種
            # =====================================================
            dog_keywords = [
                "狗", "狗狗", "小狗", "老狗", "老犬", "高齡犬", "幼犬",
                "柯基", "柴犬", "貴賓", "博美", "吉娃娃", "馬爾濟斯",
                "瑪爾濟斯", "約克夏", "臘腸", "法鬥", "黃金獵犬",
                "拉布拉多", "哈士奇", "米克斯", "台灣犬", "雪納瑞",
                "邊境牧羊犬", "邊牧"
            ]

            cat_keywords = [
                "貓", "貓咪", "小貓", "老貓", "幼貓", "高齡貓"
            ]

            if any(word in original_zh_text for word in dog_keywords):
                query_terms.append("dog")

            if any(word in original_zh_text for word in cat_keywords):
                query_terms.append("cat")

            # =====================================================
            # Age 年齡
            # =====================================================
            if any(word in original_zh_text for word in ["老狗", "老犬", "高齡犬"]):
                query_terms.extend([
                    "senior dog",
                    "older dog",
                    "aging"
                ])

            if any(word in original_zh_text for word in ["老貓", "高齡貓"]):
                query_terms.extend([
                    "senior cat",
                    "older cat",
                    "aging"
                ])

            if any(word in original_zh_text for word in ["幼犬", "小狗"]):
                query_terms.extend([
                    "puppy",
                    "young dog"
                ])

            if any(word in original_zh_text for word in ["幼貓", "小貓"]):
                query_terms.extend([
                    "kitten",
                    "young cat"
                ])

            # =====================================================
            # Dog breeds 犬種
            # =====================================================
            breed_map = {
                "柯基": "Corgi, medium dog breed",
                "威爾斯柯基": "Welsh Corgi, medium dog breed",
                "柴犬": "Shiba Inu, small to medium dog breed",
                "貴賓": "Poodle, small dog breed",
                "玩具貴賓": "Toy Poodle, small dog breed",
                "迷你貴賓": "Miniature Poodle, small dog breed",
                "博美": "Pomeranian, small dog breed",
                "吉娃娃": "Chihuahua, small dog breed",
                "馬爾濟斯": "Maltese, small dog breed",
                "瑪爾濟斯": "Maltese, small dog breed",
                "臘腸": "Dachshund, small dog breed",
                "法鬥": "French Bulldog, small dog breed",
                "黃金獵犬": "Golden Retriever, large dog breed",
                "拉布拉多": "Labrador Retriever, large dog breed",
                "哈士奇": "Siberian Husky, large dog breed",
                "米克斯": "mixed breed dog",
                "台灣犬": "Taiwan Dog, medium dog breed",
                "雪納瑞": "Schnauzer, small to medium dog breed",
                "邊境牧羊犬": "Border Collie, medium dog breed",
                "邊牧": "Border Collie, medium dog breed"
            }

            # 犬種資訊容易讓 retrieval 偏向 breed / mobility 相關資料，
            # 先不加入檢索 query，避免干擾症狀檢索。
            # for zh_breed, en_breed in breed_map.items():
            #     if zh_breed in original_zh_text:
            #         query_terms.append(en_breed)

            # =====================================================
            # Skin-related symptoms 皮膚相關症狀
            # =====================================================
            if "疣" in original_zh_text:
                query_terms.extend([
                    "wart-like bump",
                    "wart-like skin growth",
                    "skin lump",
                    "skin mass",
                    "skin growth",
                    "skin lesion",
                    "skin irritation",
                    "dermatitis",
                    "cutaneous lesion"
                ])

            if any(word in original_zh_text for word in ["一顆", "一顆一顆", "一粒", "一粒一粒", "腫塊", "肉瘤", "凸起"]):
                query_terms.extend([
                    "bump",
                    "lump",
                    "skin growth",
                    "mass"
                ])

            if any(word in original_zh_text for word in ["身上", "皮膚"]):
                query_terms.extend([
                    "skin",
                    "body"
                ])

            if any(word in original_zh_text for word in ["抓癢", "一直抓", "癢", "搔癢"]):
                query_terms.extend([
                    "itching",
                    "scratching",
                    "skin irritation"
                ])

            if any(word in original_zh_text for word in ["掉毛", "禿毛", "毛掉"]):
                query_terms.extend([
                    "hair loss",
                    "alopecia"
                ])

            if any(word in original_zh_text for word in ["紅腫", "紅疹", "發紅", "紅紅"]):
                query_terms.extend([
                    "redness",
                    "rash",
                    "erythema",
                    "skin irritation"
                ])

            if any(word in original_zh_text for word in ["脫皮", "皮屑"]):
                query_terms.extend([
                    "flaky skin",
                    "scaling",
                    "skin irritation"
                ])

            if any(word in original_zh_text for word in ["結痂", "破皮", "流血"]):
                query_terms.extend([
                    "scab",
                    "crusting",
                    "skin wound",
                    "skin irritation"
                ])

            # =====================================================
            # Ear symptoms 耳朵症狀
            # =====================================================
            if any(word in original_zh_text for word in ["耳朵", "抓耳朵", "耳朵臭", "耳臭", "甩頭", "耳垢"]):
                query_terms.extend([
                    "ear",
                    "ear infection",
                    "ear canal",
                    "ear discharge",
                    "head shaking",
                    "bad smell from ears"
                ])

            # =====================================================
            # Digestive symptoms 消化症狀
            # =====================================================
            if any(word in original_zh_text for word in ["吐", "嘔吐", "拉肚子", "腹瀉", "軟便", "血便", "沒食慾", "不吃飯"]):
                query_terms.extend([
                    "vomiting",
                    "diarrhea",
                    "digestive issues",
                    "gastrointestinal",
                    "loss of appetite"
                ])

            # =====================================================
            # Mobility symptoms 行動症狀
            # =====================================================
            if any(word in original_zh_text for word in ["走路", "跛腳", "腳痛", "站不起來", "關節", "後腳無力", "走路不穩"]):
                query_terms.extend([
                    "walking",
                    "limping",
                    "lameness",
                    "joint pain",
                    "mobility problems",
                    "weak legs"
                ])

            # =====================================================
            # Parasites 寄生蟲相關
            # =====================================================
            if any(word in original_zh_text for word in ["跳蚤", "壁蝨", "蜱蟲", "小蟲", "寄生蟲", "蟲"]):
                query_terms.extend([
                    "fleas",
                    "ticks",
                    "parasites",
                    "itching"
                ])

            # =====================================================
            # Owner concern 飼主擔心、不確定性
            # =====================================================
            # 這些是回答時的語境，不是檢索症狀的核心關鍵字，先不加入 retrieval query。
            """
            if any(word in original_zh_text for word in ["擔心", "不放心", "害怕", "正常嗎"]):
                query_terms.extend([
                    "worried owner",
                    "health concern"
                ])

            if any(word in original_zh_text for word in ["醫生說", "獸醫說"]):
                query_terms.extend([
                    "vet said",
                    "veterinary advice"
                ])
            """

            # 去除重複詞
            unique_terms = []

            for term in query_terms:
                if term not in unique_terms:
                    unique_terms.append(term)

            if unique_terms:
                structured_query = ", ".join(unique_terms)

                return (
                    f"Structured pet health query: {structured_query}. "
                    f"Original machine translation: {translated_en_text}"
                )

            return translated_en_text

    def translate_and_enhance(self, text: str) -> str:
        """
        完整流程：
        1. 中文 query 翻成英文。
        2. 根據中文原句建立 structured medical query。
        3. 再根據中文原句補上 manual domain keyword enhancement。
        4. 回傳最終給 RAG retrieval 使用的英文 query。
        """
        translated_text = self.translate_zh_to_en(text)

        structured_query = self.build_structured_medical_query(
            original_zh_text=text,
            translated_en_text=translated_text
        )

        enhanced_text = self.add_domain_keywords(
            original_zh_text=text,
            translated_en_text=structured_query
        )

        return enhanced_text