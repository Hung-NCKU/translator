"""開源模型翻譯引擎（CTranslate2）。

完全離線、永久免費、沒有用量限制，影片內容不會送到任何外部服務。
用的是專案裡已經裝好的 CTranslate2，不需要再多裝任何執行環境，
模型從 HuggingFace 下載到 models/translate。

提供兩類模型：
  llm   Qwen3 之類的語言模型。看得懂整段上下文，語氣與多句處理好很多，但較慢。
  nllb  NLLB-200 翻譯專用模型。快上百倍，但一條字幕含兩句話時容易漏譯。
"""
from __future__ import annotations

import json
import re
from typing import Callable

from .config import LANGUAGES, MODELS, quiet_hub_warnings
from .transcribe import Cue

# 顯示名稱 -> (HuggingFace repo, 類型, 大約下載大小)
CT2_MODELS = {
    "Qwen3 4B（推薦：品質好，8GB 顯卡夠用）":
        ("ctranslate2-4you/Qwen3-4B-ct2-AWQ", "llm", "2.5GB"),
    "Qwen3 8B（品質最好，較吃顯存與時間）":
        ("ctranslate2-4you/Qwen3-8B-ct2-AWQ", "llm", "5GB"),
    "NLLB-200 1.3B（極快，但長句易漏譯）":
        ("OpenNMT/nllb-200-distilled-1.3B-ct2-int8", "nllb", "1.3GB"),
}

_THINK_TAG = re.compile(r"<think>.*?</think>", re.DOTALL)

# NLLB 用的是 FLORES-200 語言代碼，和一般 ISO 代碼不一樣
_NLLB_TARGET = {
    "zh-TW": "zho_Hant", "zh-CN": "zho_Hans", "en": "eng_Latn",
    "ja": "jpn_Jpan", "ko": "kor_Hang", "es": "spa_Latn", "fr": "fra_Latn",
    "de": "deu_Latn", "vi": "vie_Latn", "th": "tha_Thai", "id": "ind_Latn",
    "ru": "rus_Cyrl", "pt": "por_Latn", "it": "ita_Latn", "ar": "arb_Arab",
    "hi": "hin_Deva",
}
_NLLB_SOURCE = {
    "zh": "zho_Hans", "ja": "jpn_Jpan", "en": "eng_Latn", "ko": "kor_Hang",
    "es": "spa_Latn", "fr": "fra_Latn", "de": "deu_Latn", "vi": "vie_Latn",
    "th": "tha_Thai", "id": "ind_Latn", "ru": "rus_Cyrl", "pt": "por_Latn",
    "it": "ita_Latn", "ar": "arb_Arab", "hi": "hin_Deva", "nl": "nld_Latn",
    "pl": "pol_Latn", "tr": "tur_Latn", "uk": "ukr_Cyrl", "sv": "swe_Latn",
}

_SYSTEM = """你是專業影視字幕譯者，將對白翻譯成{target}。

規則：
1. 一句原文對應一句譯文，絕不合併、拆分或省略；每個 id 都要有譯文。
2. 口語、自然、簡潔，符合觀眾閱讀速度。
3. 保留語氣與情緒（敬語、粗話、猶豫、玩笑都要譯出對應感覺）。
4. 語音辨識可能有錯字，請依上下文合理推斷，不要照錯字直譯。
5. 只輸出 JSON，不要任何說明文字。"""

_TAIWAN_STYLE = """
6. 一律使用台灣慣用詞，不要中國大陸用語。例如：
   早安（不是早上好）、影片（不是視頻）、網路（不是網絡）、
   軟體（不是軟件）、資訊（不是信息）、品質（不是質量）、
   計程車（不是出租車）、腳踏車（不是自行車）。"""


def _repo_dir(repo: str):
    return MODELS / "translate" / repo.replace("/", "__")


def download_model(repo: str,
                   on_setup: Callable[[float, str], None] | None = None) -> str:
    quiet_hub_warnings()
    from huggingface_hub import snapshot_download

    target = _repo_dir(repo)
    if (target / "model.bin").exists():
        return str(target)
    if on_setup:
        on_setup(0.0, f"第一次使用，正在下載翻譯模型（{repo}）")
    path = snapshot_download(repo_id=repo, local_dir=str(target))
    if on_setup:
        on_setup(1.0, "翻譯模型下載完成")
    return path


def _pick_device() -> tuple[str, str]:
    import ctranslate2
    if ctranslate2.get_cuda_device_count() > 0:
        return "cuda", "default"
    return "cpu", "int8"


class _Base:
    def __init__(self, repo: str,
                 on_setup: Callable[[float, str], None] | None = None):
        self.repo = repo
        self.model_dir = download_model(repo, on_setup)
        self._engine = None
        self._tokenizer = None
        self._on_setup = on_setup

    def _tokenizer_for(self, path: str):
        import transformers
        return transformers.AutoTokenizer.from_pretrained(path)


class CT2LLMTranslator(_Base):
    """用 Qwen3 這類語言模型翻譯，整批送出並帶上下文，語氣與一致性比較好。"""

    def __init__(self, repo: str, glossary: str = "", batch_size: int = 16,
                 on_setup: Callable[[float, str], None] | None = None):
        super().__init__(repo, on_setup)
        self.glossary = glossary.strip()
        self.batch_size = max(4, batch_size)

    def _load(self, on_progress=None):
        if self._engine is not None:
            return
        import ctranslate2
        device, compute = _pick_device()
        if on_progress:
            on_progress(0.0, f"載入翻譯模型（{device.upper()}）")
        try:
            self._engine = ctranslate2.Generator(self.model_dir, device=device,
                                                 compute_type=compute)
        except Exception:
            if device == "cpu":
                raise
            # 顯存不夠就退到 CPU，慢很多但至少跑得完
            self._engine = ctranslate2.Generator(self.model_dir, device="cpu",
                                                 compute_type="int8")
        self._tokenizer = self._tokenizer_for(self.model_dir)

    def _generate(self, system: str, user: str, max_tokens: int) -> str:
        tok = self._tokenizer
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": user}]
        try:
            # Qwen3 預設會先「想一遍」，字幕翻譯用不到，關掉快很多
            prompt = tok.apply_chat_template(messages, tokenize=False,
                                             add_generation_prompt=True,
                                             enable_thinking=False)
        except TypeError:
            prompt = tok.apply_chat_template(messages, tokenize=False,
                                             add_generation_prompt=True)
        tokens = tok.convert_ids_to_tokens(tok.encode(prompt))
        result = self._engine.generate_batch(
            [tokens], max_length=max_tokens, sampling_temperature=0.3,
            include_prompt_in_result=False)
        text = tok.decode(tok.convert_tokens_to_ids(result[0].sequences[0]),
                          skip_special_tokens=True)
        return _THINK_TAG.sub("", text).strip()

    def translate(self, cues: list[Cue], target_lang: str, source_name: str = "",
                  on_progress: Callable[[float, str], None] | None = None) -> None:
        from .translate import localize_taiwan as localize
        from .translate import target_display_name

        self._load(on_progress)
        to_taiwan = target_lang == "繁體中文"
        system = _SYSTEM.format(target=target_display_name(target_lang))
        if to_taiwan:
            system += _TAIWAN_STYLE
        if self.glossary:
            system += f"\n\n【本片專用指示與術語表，優先遵守】\n{self.glossary}"
        if source_name:
            system += f"\n\n原文語言：{source_name}。"

        done, missing = 0, 0
        for start in range(0, len(cues), self.batch_size):
            batch = cues[start:start + self.batch_size]
            context = [c.text for c in cues[max(0, start - 3):start]]
            payload = {
                "previous_lines_for_context_only": context,
                "lines_to_translate": [{"id": i, "text": c.text}
                                       for i, c in enumerate(batch)],
            }
            user = ('請翻譯 lines_to_translate 的每一句，輸出格式：\n'
                    '{"translations":[{"id":0,"text":"譯文"}]}\n\n'
                    + json.dumps(payload, ensure_ascii=False))

            budget = 120 + sum(len(c.text) for c in batch) * 4
            mapping = self._translate_batch(system, user, budget)
            for i, cue in enumerate(batch):
                text = mapping.get(i, "").strip()
                if text:
                    cue.translation = localize(text) if to_taiwan else text
                else:
                    # 這一句沒翻出來就原樣保留。千萬不能套中文化，
                    # 不然日文原句的漢字會被轉成中文字（来週→來週），變成四不像
                    cue.translation = cue.text
                    missing += 1

            done += len(batch)
            if on_progress:
                on_progress(done / max(1, len(cues)),
                            f"本地翻譯中：{done}/{len(cues)} 句"
                            + (f"（{missing} 句沒翻出來）" if missing else ""))

        # 大部分都沒翻出來時直接報錯。交出一份其實全是原文的「譯文」最糟，
        # 使用者會以為翻好了。通常是模型吐不出合法 JSON，或顯存不夠。
        if missing > len(cues) // 2:
            raise RuntimeError(
                f"本地模型有 {missing}/{len(cues)} 句翻譯失敗。\n"
                "通常是模型太大、顯存不夠，或這個模型不擅長照格式輸出。\n"
                "建議改用 Qwen3 4B，或改用 Claude API。")

    @staticmethod
    def _parse(raw: str) -> dict[int, str]:
        """從模型回應裡把 id→譯文抓出來。

        模型不會每次都照格式走：有的回 {"translations":[...]}，
        有的直接回裸陣列 [...]，有的還會在前後多講幾句話。
        這裡把常見的幾種形狀都接住，不然模型明明翻對了卻被當成失敗。
        """
        candidates = [raw]
        for pattern in (r"\{.*\}", r"\[.*\]"):      # 挖出夾在閒聊中間的 JSON
            found = re.search(pattern, raw, re.DOTALL)
            if found:
                candidates.append(found.group(0))

        for candidate in candidates:
            try:
                data = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if isinstance(data, dict):
                items = next((data[key] for key in
                              ("translations", "lines", "result", "data")
                              if isinstance(data.get(key), list)), None)
            elif isinstance(data, list):
                items = data
            else:
                items = None
            if not items:
                continue

            mapping: dict[int, str] = {}
            for position, item in enumerate(items):
                if isinstance(item, dict):
                    text = item.get("text") or item.get("translation") or ""
                    # 沒給 id 就照出現順序當作 id
                    key = item.get("id", position)
                else:
                    text, key = item, position
                try:
                    mapping[int(key)] = str(text).strip()
                except (TypeError, ValueError):
                    continue
            if mapping:
                return mapping
        return {}

    def _translate_batch(self, system: str, user: str,
                         budget: int) -> dict[int, str]:
        for _attempt in range(2):
            mapping = self._parse(self._generate(system, user, budget))
            if mapping:
                return mapping
        return {}


class CT2NLLBTranslator(_Base):
    """NLLB-200：速度極快的翻譯專用模型，適合趕時間或長片。"""

    def __init__(self, repo: str, batch_size: int = 16,
                 on_setup: Callable[[float, str], None] | None = None):
        super().__init__(repo, on_setup)
        self.batch_size = max(4, batch_size)

    def _load(self, on_progress=None):
        if self._engine is not None:
            return
        import ctranslate2
        device, _compute = _pick_device()
        compute = "int8_float16" if device == "cuda" else "int8"
        if on_progress:
            on_progress(0.0, f"載入翻譯模型（{device.upper()}）")
        try:
            self._engine = ctranslate2.Translator(self.model_dir, device=device,
                                                  compute_type=compute)
        except Exception:
            if device == "cpu":
                raise
            self._engine = ctranslate2.Translator(self.model_dir, device="cpu",
                                                  compute_type="int8")
        self._tokenizer = self._tokenizer_for(self.model_dir)

    def translate(self, cues: list[Cue], target_lang: str, source_name: str = "",
                  on_progress: Callable[[float, str], None] | None = None) -> None:
        self._load(on_progress)
        tok = self._tokenizer
        target_code = _NLLB_TARGET.get(LANGUAGES[target_lang][0], "eng_Latn")
        source_code = _NLLB_SOURCE.get((source_name or "").split("-")[0])
        if source_code:
            tok.src_lang = source_code

        done = 0
        for start in range(0, len(cues), self.batch_size):
            batch = cues[start:start + self.batch_size]
            tokens = [tok.convert_ids_to_tokens(tok.encode(c.text))
                      for c in batch]
            results = self._engine.translate_batch(
                tokens, target_prefix=[[target_code]] * len(batch),
                beam_size=4, max_batch_size=self.batch_size)
            for cue, result in zip(batch, results):
                # 第一個 token 是我們指定的語言標記，要丟掉
                produced = result.hypotheses[0][1:]
                text = tok.decode(tok.convert_tokens_to_ids(produced),
                                  skip_special_tokens=True).strip()
                cue.translation = text or cue.text

            done += len(batch)
            if on_progress:
                on_progress(done / max(1, len(cues)),
                            f"本地翻譯中：{done}/{len(cues)} 句")


def build_local(repo: str, glossary: str = "",
                on_setup: Callable[[float, str], None] | None = None):
    """依模型種類挑對應的翻譯器。"""
    kind = next((k for _name, (r, k, _s) in CT2_MODELS.items() if r == repo),
                "llm")
    if kind == "nllb":
        return CT2NLLBTranslator(repo, on_setup=on_setup)
    return CT2LLMTranslator(repo, glossary=glossary, on_setup=on_setup)
