"""影片字幕翻譯器 — Windows 端圖形介面。

真正的辨識與翻譯跑在 WSL 裡（見 backend/），這裡只負責選檔、設定與顯示進度。
"""
from __future__ import annotations

import json
import queue
import subprocess
import sys
import threading
from pathlib import Path

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

ROOT = Path(__file__).resolve().parent.parent
SETTINGS_FILE = ROOT / "settings.json"
DISTRO = "Ubuntu-24.04"

VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".avi", ".wmv", ".flv", ".webm", ".m4v",
              ".mpg", ".mpeg", ".ts", ".m2ts", ".vob", ".3gp", ".mts"}

WHISPER_MODELS = {
    "large-v3-turbo（推薦：快又準）": "deepdml/faster-whisper-large-v3-turbo-ct2",
    "large-v3（最準，較慢）": "large-v3",
    "medium（中等）": "medium",
    "small（快）": "small",
    "base（很快，品質低）": "base",
}

CLAUDE_MODELS = {
    "Claude Opus 5（品質最好）": "claude-opus-5",
    "Claude Sonnet 5（快且便宜）": "claude-sonnet-5",
    "Claude Haiku 4.5（最便宜）": "claude-haiku-4-5",
}

TARGET_LANGS = ["繁體中文", "簡體中文", "英文", "日文", "韓文", "西班牙文", "法文",
                "德文", "越南文", "泰文", "印尼文", "俄文", "葡萄牙文", "義大利文",
                "阿拉伯文", "印地文"]

SOURCE_LANGS = {
    "自動偵測": "auto", "日文": "ja", "英文": "en", "中文": "zh", "韓文": "ko",
    "西班牙文": "es", "法文": "fr", "德文": "de", "俄文": "ru", "泰文": "th",
    "越南文": "vi", "印尼文": "id", "義大利文": "it", "葡萄牙文": "pt",
}

ENGINES = {
    "本地開源模型（免費、離線）": "local",
    "Claude API（品質最好）": "claude",
    "Google 免費翻譯（常被限流）": "google",
    "不翻譯（只產生原文字幕）": "none",
}

# 本地模型：顯示名稱 -> (HuggingFace repo, 每個字大約幾秒)
# 秒數是在 RTX 5060 上實測的：模型要逐字生成，所以時間跟字數成正比，不是跟句數
LOCAL_MODELS = {
    "Qwen3 4B（推薦：快一些）": ("ctranslate2-4you/Qwen3-4B-ct2-AWQ", 0.43),
    "Qwen3 8B（語氣較好，慢 40%）": ("ctranslate2-4you/Qwen3-8B-ct2-AWQ", 0.60),
    "NLLB 1.3B（極快，長句易漏譯）":
        ("OpenNMT/nllb-200-distilled-1.3B-ct2-int8", 0.0008),
}

SUB_MODES = {
    "只輸出 .srt 字幕檔": "srt",
    "嵌入軟字幕（可開關，快）": "soft",
    "燒錄進畫面（硬字幕，慢）": "hard",
}

LAYOUTS = {"只有譯文": "target", "雙語（譯文在上）": "bilingual", "只有原文": "source"}

EFFORTS = {"標準（建議）": "medium", "高品質（較慢較貴）": "high", "省錢快速": "low"}


def to_wsl_path(win_path: str) -> str:
    """D:\\a\\b.mp4 -> /mnt/d/a/b.mp4"""
    p = Path(win_path)
    drive = p.drive.rstrip(":").lower()
    rest = str(p)[len(p.drive):].replace("\\", "/").lstrip("/")
    return f"/mnt/{drive}/{rest}" if drive else str(p).replace("\\", "/")


class App(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("影片字幕翻譯器")
        self.minsize(880, 640)

        self.files: list[str] = []
        self.queue: queue.Queue[dict] = queue.Queue()
        self.proc: subprocess.Popen | None = None
        self.running = False
        self.file_pct = 0.0
        self.current_index = -1
        self._last_setup_msg = ""
        self.stage_eta: float | None = None
        self.overall_eta: float | None = None
        self.predicted_mt: float | None = None

        self._build_style()
        self._build_widgets()
        self._load_settings()
        self._fit_to_screen()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after(100, self._drain_queue)

    def _fit_to_screen(self) -> None:
        """依實際內容決定視窗大小，螢幕放不下就縮到螢幕高度，別讓按鈕跑出畫面。"""
        self.update_idletasks()
        width = max(1020, self.winfo_reqwidth())
        height = self.winfo_reqheight()
        max_h = self.winfo_screenheight() - 80      # 留給工作列與標題列
        max_w = self.winfo_screenwidth() - 40
        width, height = min(width, max_w), min(height, max_h)
        x = max(0, (self.winfo_screenwidth() - width) // 2)
        y = max(0, (self.winfo_screenheight() - height) // 3)
        self.geometry(f"{width}x{height}+{x}+{y}")

    # ---------------------------------------------------------------- 介面
    def _build_style(self) -> None:
        try:
            from ctypes import windll
            windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            pass
        style = ttk.Style(self)
        if "vista" in style.theme_names():
            style.theme_use("vista")
        font = ("Microsoft JhengHei UI", 10)
        for name in ("TLabel", "TButton", "TCheckbutton", "TRadiobutton",
                     "TLabelframe.Label", "TEntry", "TCombobox"):
            style.configure(name, font=font)
        style.configure("Treeview", font=font, rowheight=26)
        style.configure("Treeview.Heading", font=(font[0], 10, "bold"))
        self.option_add("*TCombobox*Listbox.font", font)

    def _build_widgets(self) -> None:
        outer = ttk.Frame(self, padding=10)
        outer.pack(fill="both", expand=True)

        # 用 grid 的權重配置：清單與紀錄按 3:2 分配多出來的空間，會跟著視窗長大縮小；
        # 設定區與按鈕列不分配伸縮空間，但也不會被壓掉。
        outer.columnconfigure(0, weight=1)
        outer.rowconfigure(0, weight=3, minsize=80)
        outer.rowconfigure(1, weight=2, minsize=60)

        # --- 檔案清單 ---
        files_box = ttk.LabelFrame(outer, text=" 影片檔案 ", padding=8)

        bar = ttk.Frame(files_box)
        bar.pack(fill="x", pady=(0, 6))
        ttk.Button(bar, text="加入影片…", command=self.add_files).pack(side="left")
        ttk.Button(bar, text="加入整個資料夾…",
                   command=self.add_folder).pack(side="left", padx=4)
        ttk.Button(bar, text="移除選取",
                   command=self.remove_selected).pack(side="left", padx=4)
        ttk.Button(bar, text="清空", command=self.clear_files).pack(side="left")
        self.count_label = ttk.Label(bar, text="尚未選擇檔案")
        self.count_label.pack(side="right")

        cols = ("name", "status")
        self.tree = ttk.Treeview(files_box, columns=cols, show="headings", height=5)
        self.tree.heading("name", text="檔案")
        self.tree.heading("status", text="狀態")
        # stretch 讓欄寬跟著視窗一起變，檔名欄吃掉多出來的寬度
        self.tree.column("name", width=560, minwidth=200, anchor="w", stretch=True)
        self.tree.column("status", width=240, minwidth=120, anchor="w", stretch=False)
        scroll = ttk.Scrollbar(files_box, orient="vertical",
                               command=self.tree.yview)
        self.tree.configure(yscrollcommand=scroll.set)
        self.tree.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")

        # --- 設定 ---
        cfg = ttk.LabelFrame(outer, text=" 設定 ", padding=8)
        for col in (1, 3):
            cfg.columnconfigure(col, weight=1)

        # 預設值一律取選單的第一項，這樣以後改了選項文字也不會對不上而當掉
        def first(options) -> str:
            return next(iter(options))

        self.var_source = tk.StringVar(value=first(SOURCE_LANGS))
        self.var_target = tk.StringVar(value="繁體中文")
        self.var_model = tk.StringVar(value=first(WHISPER_MODELS))
        self.var_engine = tk.StringVar(value=first(ENGINES))
        self.var_claude = tk.StringVar(value=first(CLAUDE_MODELS))
        self.var_local = tk.StringVar(value=first(LOCAL_MODELS))
        self.var_effort = tk.StringVar(value=first(EFFORTS))
        self.var_mode = tk.StringVar(value=first(SUB_MODES))
        self.var_layout = tk.StringVar(value="雙語（譯文在上）")
        self.var_key = tk.StringVar()
        self.var_outdir = tk.StringVar()
        self.var_same_dir = tk.BooleanVar(value=True)
        self.var_fontsize = tk.IntVar(value=20)

        def combo(row, col, label, var, values, width=26):
            text = ttk.Label(cfg, text=label)
            text.grid(row=row, column=col * 2, sticky="w", padx=(0, 6), pady=4)
            box = ttk.Combobox(cfg, textvariable=var, values=values,
                               state="readonly", width=width)
            box.grid(row=row, column=col * 2 + 1, sticky="ew", padx=(0, 16), pady=4)
            return text, box

        combo(0, 0, "影片語言", self.var_source, list(SOURCE_LANGS))
        combo(0, 1, "翻譯成", self.var_target, TARGET_LANGS)
        combo(1, 0, "語音辨識模型", self.var_model, list(WHISPER_MODELS))
        combo(1, 1, "翻譯引擎", self.var_engine, list(ENGINES))[1].bind(
            "<<ComboboxSelected>>", lambda _e: self._sync_engine())
        # 這兩組佔同一格，依照選到的引擎只顯示其中一組
        self.claude_row = combo(2, 0, "Claude 模型", self.var_claude,
                                list(CLAUDE_MODELS))
        self.local_row = combo(2, 0, "本地翻譯模型", self.var_local,
                               list(LOCAL_MODELS))
        self.effort_row = combo(2, 1, "翻譯品質", self.var_effort, list(EFFORTS))
        combo(3, 0, "字幕輸出方式", self.var_mode, list(SUB_MODES))[1].bind(
            "<<ComboboxSelected>>", lambda _e: self._sync_mode())
        combo(3, 1, "字幕內容", self.var_layout, list(LAYOUTS))

        ttk.Label(cfg, text="Anthropic API Key").grid(row=4, column=0, sticky="w",
                                                      padx=(0, 6), pady=4)
        self.key_entry = ttk.Entry(cfg, textvariable=self.var_key, show="•")
        self.key_entry.grid(row=4, column=1, sticky="ew", padx=(0, 16), pady=4)
        ttk.Label(cfg, text="硬字幕字級").grid(row=4, column=2, sticky="w",
                                              padx=(0, 6), pady=4)
        self.font_spin = ttk.Spinbox(cfg, from_=10, to=48, width=6,
                                     textvariable=self.var_fontsize)
        self.font_spin.grid(row=4, column=3, sticky="w", pady=4)

        out_row = ttk.Frame(cfg)
        out_row.grid(row=5, column=0, columnspan=4, sticky="ew", pady=(6, 0))
        ttk.Checkbutton(out_row, text="輸出到影片原本的資料夾",
                        variable=self.var_same_dir,
                        command=self._sync_outdir).pack(side="left")
        self.outdir_entry = ttk.Entry(out_row, textvariable=self.var_outdir)
        self.outdir_entry.pack(side="left", fill="x", expand=True, padx=8)
        self.outdir_btn = ttk.Button(out_row, text="瀏覽…",
                                     command=self.choose_outdir)
        self.outdir_btn.pack(side="left")

        ttk.Label(cfg, text="翻譯提示 / 術語表（選填，例如角色名、專有名詞對照）"
                  ).grid(row=6, column=0, columnspan=4, sticky="w", pady=(8, 2))
        self.glossary = tk.Text(cfg, height=2, wrap="word",
                                font=("Microsoft JhengHei UI", 10))
        self.glossary.grid(row=7, column=0, columnspan=4, sticky="ew")

        # --- 執行紀錄（和檔案清單一起分配剩餘空間）---
        log_box = ttk.LabelFrame(outer, text=" 執行紀錄 ", padding=8)
        self.log = tk.Text(log_box, height=5, wrap="word", state="disabled",
                           font=("Consolas", 9), background="#111418",
                           foreground="#d7dde3", insertbackground="#d7dde3")
        log_scroll = ttk.Scrollbar(log_box, orient="vertical",
                                   command=self.log.yview)
        self.log.configure(yscrollcommand=log_scroll.set)
        self.log.pack(side="left", fill="both", expand=True)
        log_scroll.pack(side="right", fill="y")

        # --- 進度 ---
        prog = ttk.Frame(outer)
        self.status = ttk.Label(prog, text="待命中", anchor="w")
        self.status.pack(fill="x")
        self.progress = ttk.Progressbar(prog, mode="determinate", maximum=1000)
        self.progress.pack(fill="x", pady=4)

        actions = ttk.Frame(outer)
        self.start_btn = ttk.Button(actions, text="開始處理", command=self.start)
        self.start_btn.pack(side="left")
        self.stop_btn = ttk.Button(actions, text="停止", command=self.stop,
                                   state="disabled")
        self.stop_btn.pack(side="left", padx=6)
        ttk.Button(actions, text="開啟輸出資料夾",
                   command=self.open_output).pack(side="left")
        ttk.Button(actions, text="安裝 / 修復環境",
                   command=self.run_setup).pack(side="right")

        files_box.grid(row=0, column=0, sticky="nsew")
        log_box.grid(row=1, column=0, sticky="nsew", pady=(8, 0))
        cfg.grid(row=2, column=0, sticky="ew", pady=(8, 0))
        prog.grid(row=3, column=0, sticky="ew", pady=(8, 0))
        actions.grid(row=4, column=0, sticky="ew", pady=(8, 0))

        self._sync_engine()
        self._sync_mode()
        self._sync_outdir()

    # ---------------------------------------------------------------- 狀態同步
    def _sync_engine(self) -> None:
        engine = ENGINES[self.var_engine.get()]
        is_claude, is_local = engine == "claude", engine == "local"

        # Claude 模型與本地模型共用同一格，換引擎就換掉整組（標籤加下拉）
        for row, shown in ((self.claude_row, is_claude), (self.local_row, is_local)):
            for widget in row:
                if shown:
                    widget.grid()
                else:
                    widget.grid_remove()

        # 翻譯品質（effort）只有 Claude 用得到
        self.effort_row[1].configure(state="readonly" if is_claude else "disabled")
        self.key_entry.configure(state="normal" if is_claude else "disabled")

    def _sync_mode(self) -> None:
        is_hard = SUB_MODES[self.var_mode.get()] == "hard"
        self.font_spin.configure(state="normal" if is_hard else "disabled")

    def _sync_outdir(self) -> None:
        state = "disabled" if self.var_same_dir.get() else "normal"
        self.outdir_entry.configure(state=state)
        self.outdir_btn.configure(state=state)

    # ---------------------------------------------------------------- 檔案
    def add_files(self) -> None:
        patterns = " ".join(f"*{e}" for e in sorted(VIDEO_EXTS))
        paths = filedialog.askopenfilenames(
            title="選擇影片（可多選）",
            filetypes=[("影片檔", patterns), ("所有檔案", "*.*")])
        self._add(paths)

    def add_folder(self) -> None:
        folder = filedialog.askdirectory(title="選擇資料夾（含子資料夾）")
        if not folder:
            return
        found = [str(p) for p in sorted(Path(folder).rglob("*"))
                 if p.suffix.lower() in VIDEO_EXTS and p.is_file()]
        if not found:
            messagebox.showinfo("沒有找到影片", "這個資料夾裡找不到支援的影片檔。")
        self._add(found)

    def _add(self, paths) -> None:
        added = 0
        for path in paths:
            if path not in self.files:
                self.files.append(path)
                self.tree.insert("", "end", values=(Path(path).name, "等待中"))
                added += 1
        if added:
            self._log(f"加入 {added} 個檔案")
            self._refresh_names()
        self._refresh_count()

    def _refresh_names(self) -> None:
        """清單裡有同名檔案時，補上上一層資料夾，才分得出誰是誰。"""
        counts: dict[str, int] = {}
        for path in self.files:
            name = Path(path).name
            counts[name] = counts.get(name, 0) + 1

        for item, path in zip(self.tree.get_children(), self.files):
            p = Path(path)
            label = f"{p.parent.name}\\{p.name}" if counts[p.name] > 1 else p.name
            self.tree.set(item, "name", label)

    def remove_selected(self) -> None:
        if self.running:
            return
        for item in reversed(self.tree.selection()):
            index = self.tree.index(item)
            self.tree.delete(item)
            del self.files[index]
        self._refresh_names()      # 移掉其中一個之後可能就不再撞名了
        self._refresh_count()

    def clear_files(self) -> None:
        if self.running:
            return
        self.files.clear()
        self.tree.delete(*self.tree.get_children())
        self._refresh_count()

    def _refresh_count(self) -> None:
        n = len(self.files)
        self.count_label.configure(text=f"共 {n} 個檔案" if n else "尚未選擇檔案")

    def choose_outdir(self) -> None:
        folder = filedialog.askdirectory(title="選擇輸出資料夾")
        if folder:
            self.var_outdir.set(folder)

    def open_output(self) -> None:
        target = self.var_outdir.get() if not self.var_same_dir.get() else ""
        if not target:
            target = str(Path(self.files[0]).parent) if self.files else str(ROOT)
        subprocess.Popen(["explorer", str(Path(target))])

    # ---------------------------------------------------------------- 設定存檔
    def _load_settings(self) -> None:
        if not SETTINGS_FILE.exists():
            return
        try:
            data = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        # 選項文字會隨版本調整，舊的設定檔不能直接灌進去，否則一開啟就當掉。
        # 完全相符就用，不然找「開頭一樣」的新選項，再不然就保留預設值。
        choices = [("source", self.var_source, SOURCE_LANGS),
                   ("target", self.var_target, {k: k for k in TARGET_LANGS}),
                   ("model", self.var_model, WHISPER_MODELS),
                   ("engine", self.var_engine, ENGINES),
                   ("claude_model", self.var_claude, CLAUDE_MODELS),
                   ("local_model", self.var_local, LOCAL_MODELS),
                   ("effort", self.var_effort, EFFORTS),
                   ("mode", self.var_mode, SUB_MODES),
                   ("layout", self.var_layout, LAYOUTS)]
        for key, var, valid in choices:
            saved = data.get(key)
            if not saved:
                continue
            if saved in valid:
                var.set(saved)
                continue
            head = saved.split("（")[0]
            match = next((option for option in valid
                          if option.split("（")[0] == head), None)
            if match:
                var.set(match)

        for key, var in (("api_key", self.var_key), ("outdir", self.var_outdir)):
            if data.get(key):
                var.set(data[key])
        self.var_same_dir.set(bool(data.get("same_dir", True)))
        self.var_fontsize.set(int(data.get("font_size", 20)))
        if data.get("glossary"):
            self.glossary.insert("1.0", data["glossary"])
        self._sync_engine()
        self._sync_mode()
        self._sync_outdir()

    def _save_settings(self) -> None:
        data = {
            "source": self.var_source.get(), "target": self.var_target.get(),
            "model": self.var_model.get(), "engine": self.var_engine.get(),
            "claude_model": self.var_claude.get(), "effort": self.var_effort.get(),
            "local_model": self.var_local.get(),
            "mode": self.var_mode.get(), "layout": self.var_layout.get(),
            "api_key": self.var_key.get(), "outdir": self.var_outdir.get(),
            "same_dir": self.var_same_dir.get(),
            "font_size": int(self.var_fontsize.get()),
            "glossary": self.glossary.get("1.0", "end").strip(),
        }
        try:
            SETTINGS_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2),
                                     encoding="utf-8")
        except OSError:
            pass

    # ---------------------------------------------------------------- 執行
    def start(self) -> None:
        if self.running:
            return
        if not self.files:
            messagebox.showwarning("還沒有檔案", "請先加入至少一個影片檔。")
            return
        engine = ENGINES[self.var_engine.get()]
        if engine == "claude" and not self.var_key.get().strip():
            messagebox.showwarning(
                "缺少 API Key",
                "使用 Claude 翻譯需要填入 Anthropic API Key。\n"
                "可以到 console.anthropic.com 建立，或改選 Google 免費翻譯。")
            return
        if not self.var_same_dir.get() and not self.var_outdir.get().strip():
            messagebox.showwarning("缺少輸出資料夾", "請選擇輸出資料夾。")
            return

        self._save_settings()
        job = {
            "files": [to_wsl_path(f) for f in self.files],
            "output_dir": "" if self.var_same_dir.get()
                          else to_wsl_path(self.var_outdir.get()),
            "model": WHISPER_MODELS[self.var_model.get()],
            "source_lang": SOURCE_LANGS[self.var_source.get()],
            "source_code": SOURCE_LANGS[self.var_source.get()],
            "target_lang": self.var_target.get(),
            "engine": engine,
            "api_key": self.var_key.get().strip(),
            "claude_model": CLAUDE_MODELS[self.var_claude.get()],
            "local_model": LOCAL_MODELS[self.var_local.get()][0],
            "effort": EFFORTS[self.var_effort.get()],
            "glossary": self.glossary.get("1.0", "end").strip(),
            "subtitle_mode": SUB_MODES[self.var_mode.get()],
            "layout": LAYOUTS[self.var_layout.get()],
            "font_size": int(self.var_fontsize.get()),
        }

        for item in self.tree.get_children():
            self.tree.set(item, "status", "等待中")
        self.running = True
        self.stage_eta = self.overall_eta = self.predicted_mt = None
        self._last_setup_msg = ""
        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.progress.configure(value=0)
        self._log("=" * 60)
        self._log(f"開始處理 {len(self.files)} 個檔案")
        threading.Thread(target=self._worker, args=(job,), daemon=True).start()

    def _worker(self, job: dict) -> None:
        script = to_wsl_path(str(ROOT / "scripts" / "run_backend.sh"))
        cmd = ["wsl.exe", "-d", DISTRO, "--", "bash", script]
        try:
            self.proc = subprocess.Popen(
                cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                errors="replace", bufsize=1,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        except OSError as exc:
            self.queue.put({"type": "fatal", "error": f"無法啟動 WSL：{exc}"})
            return

        assert self.proc.stdin and self.proc.stdout
        self.proc.stdin.write(json.dumps(job, ensure_ascii=False))
        self.proc.stdin.close()
        for line in self.proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                self.queue.put(json.loads(line))
            except json.JSONDecodeError:
                self.queue.put({"type": "log", "msg": line})
        code = self.proc.wait()
        self.queue.put({"type": "finished", "code": code})

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            self._log("正在停止…")
            try:
                subprocess.run(["wsl.exe", "-d", DISTRO, "--", "pkill", "-f",
                                "backend.cli"], capture_output=True,
                               creationflags=getattr(subprocess,
                                                     "CREATE_NO_WINDOW", 0))
                self.proc.terminate()
            except OSError:
                pass

    # ---------------------------------------------------------------- 訊息處理
    def _drain_queue(self) -> None:
        try:
            while True:
                self._handle(self.queue.get_nowait())
        except queue.Empty:
            pass
        self.after(100, self._drain_queue)

    def _handle(self, msg: dict) -> None:
        kind = msg.get("type")
        items = self.tree.get_children()

        if kind == "file_start":
            self.current_index = msg["index"]
            self.file_pct = 0.0
            if msg["index"] < len(items):
                self.tree.set(items[msg["index"]], "status", "處理中…")
                self.tree.see(items[msg["index"]])
            self._log(f"[{msg['index'] + 1}/{msg['total']}] {msg['name']}")

        elif kind == "progress":
            self.file_pct = msg.get("pct", 0.0)
            eta = msg.get("eta")
            # 翻譯是最慢的一段，但要跑完第一批才量得到速度。
            # 在那之前先用句數推估，才不會一開始完全沒有時間可看。
            if (eta is None and msg.get("stage") == "mt"
                    and self.predicted_mt is not None):
                eta = self.predicted_mt * (1.0 - msg.get("frac", 0.0))
            self.stage_eta = eta
            self._update_progress(msg.get("msg", ""))
            if msg.get("msg") and msg["index"] < len(items):
                self.tree.set(items[msg["index"]], "status", msg["msg"])

        elif kind == "cue_count":
            # 辨識一做完就能估翻譯要多久，不用等翻譯跑起來才知道
            self.predicted_mt = self._predict_translation(
                msg.get("count", 0), msg.get("chars", 0))
            if self.predicted_mt:
                self._log(f"    預估翻譯時間：約 {self._format(self.predicted_mt)}"
                          f"（{msg.get('count')} 句、{msg.get('chars', 0)} 字）")

        elif kind == "overall_eta":
            self.overall_eta = msg.get("eta")

        elif kind == "file_done":
            if msg["index"] < len(items):
                self.tree.set(items[msg["index"]], "status", "✔ 完成")
            for path in msg.get("outputs", []):
                self._log(f"    產出：{Path(path).name}")
            self.file_pct = 1.0
            self._update_progress("完成一個檔案")

        elif kind == "file_error":
            if msg["index"] < len(items):
                self.tree.set(items[msg["index"]], "status", "✘ 失敗")
            self._log(f"    錯誤：{msg.get('error')}")

        elif kind == "setup":
            # 本地翻譯引擎第一次要下載模型，進度走自己的一條，不算在檔案進度裡
            text = msg.get("msg", "")
            self.progress.configure(value=int(msg.get("pct", 0.0) * 1000))
            self.status.configure(text=f"準備中：{text}")
            if text != self._last_setup_msg:
                self._last_setup_msg = text
                self._log(f"    {text}")

        elif kind == "log":
            self._log(f"    {msg.get('msg', '')}")

        elif kind == "fatal":
            self._log(f"無法執行：{msg.get('error')}")
            messagebox.showerror("執行失敗", msg.get("error", ""))

        elif kind == "all_done":
            failed, total = msg.get("failed", 0), msg.get("total", 0)
            self._log(f"全部結束：成功 {total - failed} / {total}")

        elif kind == "finished":
            self.running = False
            self.proc = None
            self.start_btn.configure(state="normal")
            self.stop_btn.configure(state="disabled")
            self.status.configure(text="待命中")
            if msg.get("code") not in (0, 1):
                self._log(f"後端結束碼 {msg.get('code')}")

    @staticmethod
    def _format(seconds: float) -> str:
        seconds = int(max(0, seconds))
        if seconds < 60:
            return f"{seconds} 秒"
        if seconds < 3600:
            return f"{seconds // 60} 分 {seconds % 60} 秒"
        return f"{seconds // 3600} 小時 {(seconds % 3600) // 60} 分"

    # 雲端引擎的時間由網路來回決定，用每句秒數估比較準
    _ENGINE_SPEED = {"claude": 0.4, "google": 0.6, "none": 0.0}

    def _predict_translation(self, cue_count: int, chars: int = 0) -> float | None:
        """用所選引擎的實測速度，估算這個檔案的翻譯時間。"""
        if not cue_count:
            return None
        engine = ENGINES[self.var_engine.get()]
        if engine == "local":
            # 本地模型逐字生成，時間跟字數成正比；載入模型另外要 3 分鐘左右
            per_char = LOCAL_MODELS.get(self.var_local.get(), (None, 0))[1]
            if not per_char:
                return None
            return (chars or cue_count * 20) * per_char + 180
        per_cue = self._ENGINE_SPEED.get(engine, 0.0)
        return cue_count * per_cue if per_cue else None

    def _update_progress(self, message: str) -> None:
        total = max(1, len(self.files))
        done = max(0, self.current_index)
        overall = (done + self.file_pct) / total
        self.progress.configure(value=int(overall * 1000))

        # 剩餘時間優先用整批的實測平均，其次用目前階段的推估
        if self.overall_eta:
            remaining = f"　|　全部剩餘約 {self._format(self.overall_eta)}"
        elif self.stage_eta:
            remaining = f"　|　此階段剩餘約 {self._format(self.stage_eta)}"
        else:
            remaining = ""
        self.status.configure(
            text=f"總進度 {overall * 100:.1f}%　|　檔案 {done + 1}/{total}"
                 f"　|　{message}{remaining}")

    def _log(self, text: str) -> None:
        self.log.configure(state="normal")
        self.log.insert("end", text + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    # ---------------------------------------------------------------- 其他
    def run_setup(self) -> None:
        if not messagebox.askyesno(
                "安裝環境",
                "會開一個視窗在 WSL 裡安裝所需環境（第一次約需數分鐘）。要繼續嗎？"):
            return
        script = to_wsl_path(str(ROOT / "scripts" / "setup_wsl.sh"))
        subprocess.Popen(["cmd", "/c", "start", "", "wsl.exe", "-d", DISTRO,
                          "--", "bash", "-lc",
                          f"bash '{script}'; echo; read -p '按 Enter 關閉…'"])

    def _on_close(self) -> None:
        if self.running and not messagebox.askyesno(
                "還在處理中", "處理尚未完成，確定要關閉嗎？"):
            return
        self.stop()
        self._save_settings()
        self.destroy()


if __name__ == "__main__":
    if sys.platform != "win32":
        print("這個介面要在 Windows 上執行。")
        sys.exit(1)
    App().mainloop()
