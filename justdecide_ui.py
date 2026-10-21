"""
JustDecide test window.

    python decide_ui.py      (= justdecide_ui.main)   with JustDecide-q8_0.gguf next to it (offers to download it)

Write the situation (as fields or as JSON / text), a question, its type and up to 26
options (A-Z), then Decide (or Ctrl+Enter): the decision, its confidence, every option's probability
and how long it took. "Request JSON" shows the same request as you would POST it to the API.
The model runs with llama-cpp-python (see requirements.txt); tkinter comes with Python.
"""
import argparse
import glob
import json
import os
import re
import threading
import time
import tkinter as tk
from tkinter import messagebox, ttk

from justdecide import (HERE, HF_REPO, LETTERS, MODEL_FILE, MODEL_NAME, MODEL_URL, Engine, QuestionError,
                        answer_probs, as_api_question, default_files, download_model, open_engine)

KINDS = [("choice", "choice"), ("score (levels, lowest first)", "score"), ("yes / no", "yes/no")]
EXAMPLE = {
    "sentence": "Battery is at 14 percent, 2 orders are waiting, the robot is in aisle 3 and the lidar is clear.",
    "state": {"battery_pct": 14, "order_queue": 2, "position": "aisle 3", "lidar": "clear"},
    "question": "What should the AGV do next?",
    "kind": "choice",
    "options": [("pick next order", ""), ("go charge", "drive to the charger now"), ("go to parking", "")],
}
BAR, BAR_BEST, MUTED = "#9db7d5", "#2f6db5", "#666666"
QUESTION_NAME = "decision"

# --- sentence -> JSON with a plain chat model (optional) ----------------------------------------
EXTRACT_SYSTEM = (
    "You turn a description of a situation into one flat JSON object of facts. Keys are short snake_case "
    "English names. Numbers are JSON numbers without units; put the unit in the key (battery_pct, distance_m, "
    "temp_c). Yes/no facts are true or false. Other values are short strings. Use only facts written in the "
    "LAST message.")
# Three different kinds of example, so a 0.6B model learns the format without copying one example's facts.
EXTRACT_EXAMPLES = [
    ("The robot is in aisle 7 at 80% battery, carrying rack R2 to dock 5. Emergency stop is not pressed and the "
     "motor is at 61 degrees.",
     '{"position": "aisle 7", "battery_pct": 80, "carrying": "rack R2", "destination": "dock 5", '
     '"estop_pressed": false, "motor_temp_c": 61}'),
    ("Customer says the charger is dead and wants a replacement this week.",
     '{"customer_issue": "charger is dead", "customer_request": "replacement this week"}'),
    ("The lift holds 120 kg of its 300 kg limit and the safety door is closed.",
     '{"lift_load_kg": 120, "lift_limit_kg": 300, "safety_door_closed": true}'),
]
FLAT_OBJECT = {"type": "object", "additionalProperties": {"type": ["string", "number", "boolean", "null"]}}


def extract_prompt(sentence):
    turns = [f"<|im_start|>system\n{EXTRACT_SYSTEM}<|im_end|>\n"]
    for said, made in EXTRACT_EXAMPLES:
        turns.append(f"<|im_start|>user\n{said}<|im_end|>\n<|im_start|>assistant\n{made}<|im_end|>\n")
    turns.append(f"<|im_start|>user\n{sentence}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
    return "".join(turns)


# --- field values <-> JSON values ---------------------------------------------------------------
def parse_value(text):
    """What a field's text means in JSON: 14 -> 14, 2.5 -> 2.5, true -> true, empty -> null, else a string."""
    t = text.strip()
    if t == "" or t.lower() in ("null", "none"):
        return None
    if t.lower() in ("true", "false"):
        return t.lower() == "true"
    if re.fullmatch(r"-?\d+", t):
        return int(t)
    if re.fullmatch(r"-?\d*\.\d+(e-?\d+)?|-?\d+e-?\d+", t, re.I):
        return float(t)
    if t[:1] in "[{":
        try:
            return json.loads(t)
        except ValueError:
            pass
    return t


def format_value(v):
    if v is None:
        return ""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (list, dict)):
        return json.dumps(v, ensure_ascii=False)
    return str(v)


class FieldTable(ttk.Frame):
    """Rows of  name | value | x  inside a scrollable area. on_change() runs after every edit."""

    def __init__(self, parent, on_change):
        super().__init__(parent)
        self.on_change, self.rows, self.quiet = on_change, [], False
        self.canvas = tk.Canvas(self, highlightthickness=0, height=110)
        bar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.inner = ttk.Frame(self.canvas)
        self.inner.columnconfigure(0, weight=2)
        self.inner.columnconfigure(1, weight=3)
        self.inner.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        win = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.canvas.bind("<Configure>", lambda e: self.canvas.itemconfigure(win, width=e.width))
        self.canvas.configure(yscrollcommand=bar.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        bar.pack(side="right", fill="y")
        ttk.Label(self.inner, text="name", foreground=MUTED).grid(row=0, column=0, sticky="w")
        ttk.Label(self.inner, text="value  (14, 2.5, true, text; empty = null)", foreground=MUTED).grid(
            row=0, column=1, sticky="w")
        self.next_row = 1

    def add(self, key="", value="", focus=False):
        k, v = tk.StringVar(value=key), tk.StringVar(value=value)
        ek = ttk.Entry(self.inner, textvariable=k)
        ev = ttk.Entry(self.inner, textvariable=v)
        x = ttk.Button(self.inner, text="x", width=3)
        row = (k, v, [ek, ev, x])
        x.config(command=lambda: self.remove(row))
        ek.grid(row=self.next_row, column=0, sticky="ew", padx=(0, 4), pady=1)
        ev.grid(row=self.next_row, column=1, sticky="ew", padx=(0, 4), pady=1)
        x.grid(row=self.next_row, column=2, pady=1)
        self.next_row += 1
        ev.bind("<Return>", lambda e: self.add(focus=True))
        for var in (k, v):
            var.trace_add("write", lambda *a: self.changed())
        self.rows.append(row)
        if focus:
            ek.focus_set()
            self.after(50, lambda: self.canvas.yview_moveto(1.0))
        self.changed()

    def remove(self, row):
        for w in row[2]:
            w.destroy()
        self.rows.remove(row)
        self.changed()

    def set_dict(self, d):
        self.quiet = True
        for row in list(self.rows):
            self.remove(row)
        for k, v in d.items():
            self.add(str(k), format_value(v))
        self.quiet = False

    def to_dict(self):
        return {k.get().strip(): parse_value(v.get()) for k, v, _ in self.rows if k.get().strip()}

    def changed(self):
        if not self.quiet:
            self.on_change()


class InputPanel(ttk.Frame):
    """
    Situation (fields / JSON, plus a Sentence tab when a chat model is given), question, type, options
    and the Decide button. chat_gguf + llama_server: a plain chat GGUF that writes JSON from a sentence.
    """

    def __init__(self, parent, on_decide, decide_text="Decide  (Ctrl+Enter)", chat_gguf=None, llama_server=None):
        super().__init__(parent)
        self.on_decide, self.options, self.chat = on_decide, [], None
        self.chat_gguf, self.llama_server = chat_gguf, llama_server
        self.columnconfigure(0, weight=1)

        sit = ttk.LabelFrame(self, text="Situation (state)", padding=6)
        sit.grid(row=0, column=0, sticky="nsew")
        self.rowconfigure(0, weight=2)
        self.state_tabs = ttk.Notebook(sit)
        self.state_tabs.pack(fill="both", expand=True)

        tab = ttk.Frame(self.state_tabs, padding=6)
        self.state_tabs.add(tab, text="Fields")
        self.fields = FieldTable(tab, self.fields_changed)
        self.fields.pack(fill="both", expand=True)
        row = ttk.Frame(tab)
        row.pack(fill="x", pady=(6, 0))
        ttk.Button(row, text="+ field", command=lambda: self.fields.add(focus=True)).pack(side="left")
        ttk.Label(row, text="every change rewrites the JSON", foreground=MUTED).pack(side="left", padx=8)

        tab = ttk.Frame(self.state_tabs, padding=6)
        self.state_tabs.add(tab, text="JSON / text")
        self.state = tk.Text(tab, height=4, wrap="word", font=("Consolas", 10), undo=True)
        self.state.pack(fill="both", expand=True)
        ttk.Label(tab, text="This is exactly what the model reads (JSON or plain text).",
                  foreground=MUTED).pack(anchor="w", pady=(6, 0))

        self.sentence = None
        if chat_gguf and os.path.exists(chat_gguf) and llama_server and os.path.exists(llama_server):
            tab = ttk.Frame(self.state_tabs, padding=6)
            self.state_tabs.add(tab, text="Sentence")
            self.sentence = tk.Text(tab, height=4, wrap="word", font=("Segoe UI", 10), undo=True)
            self.sentence.pack(fill="both", expand=True)
            row = ttk.Frame(tab)
            row.pack(fill="x", pady=(6, 0))
            self.make_btn = ttk.Button(row, text="Make JSON", command=self.make_json)
            self.make_btn.pack(side="left")
            self.make_status = ttk.Label(row, text=f"{os.path.basename(chat_gguf)} writes it; loads on first use",
                                         foreground=MUTED)
            self.make_status.pack(side="left", padx=8)
        self.state_tabs.bind("<<NotebookTabChanged>>", lambda e: self.tab_changed())

        ttk.Label(self, text="Question").grid(row=1, column=0, sticky="w", pady=(8, 0))
        self.question = ttk.Entry(self, font=("Segoe UI", 10))
        self.question.grid(row=2, column=0, sticky="ew", pady=(2, 6))

        kinds = ttk.Frame(self)
        kinds.grid(row=3, column=0, sticky="w", pady=(0, 4))
        ttk.Label(kinds, text="Type:").pack(side="left")
        self.kind = tk.StringVar(value="choice")
        for text, value in KINDS:
            ttk.Radiobutton(kinds, text=text, value=value, variable=self.kind,
                            command=self.kind_changed).pack(side="left", padx=6)

        o = ttk.LabelFrame(self, text="Options", padding=6)
        o.grid(row=4, column=0, sticky="nsew")
        self.rowconfigure(4, weight=3)
        o.columnconfigure(0, weight=1)
        o.columnconfigure(1, weight=1)
        ttk.Label(o, text="option", foreground=MUTED).grid(row=0, column=0, sticky="w")
        ttk.Label(o, text="what it means (optional)", foreground=MUTED).grid(row=0, column=1, sticky="w")
        self.opt_label = ttk.Entry(o)
        self.opt_label.grid(row=1, column=0, sticky="ew", padx=(0, 4))
        self.opt_desc = ttk.Entry(o)
        self.opt_desc.grid(row=1, column=1, sticky="ew", padx=(0, 4))
        self.add_btn = ttk.Button(o, text="Add", command=self.add_option)
        self.add_btn.grid(row=1, column=2)
        for w in (self.opt_label, self.opt_desc):
            w.bind("<Return>", lambda e: self.add_option())
        self.opt_list = tk.Listbox(o, height=6, font=("Consolas", 10), activestyle="none")
        self.opt_list.grid(row=2, column=0, columnspan=2, sticky="nsew", pady=6)
        self.opt_list.bind("<Delete>", lambda e: self.remove_option())
        o.rowconfigure(2, weight=1)
        side = ttk.Frame(o)
        side.grid(row=2, column=2, sticky="n", pady=6)
        self.side_buttons = []
        for text, cmd in [("Remove", self.remove_option), ("Up", lambda: self.move(-1)),
                          ("Down", lambda: self.move(1)), ("Clear", self.clear_options),
                          ("Paste list...", self.bulk_add)]:
            b = ttk.Button(side, text=text, command=cmd, width=11)
            b.pack(pady=1)
            self.side_buttons.append(b)
        self.count = ttk.Label(o, text="", foreground=MUTED)
        self.count.grid(row=3, column=0, columnspan=3, sticky="w")

        buttons = ttk.Frame(self)
        buttons.grid(row=5, column=0, sticky="ew", pady=(8, 0))
        self.decide_btn = ttk.Button(buttons, text=decide_text, command=self.on_decide)
        self.decide_btn.pack(side="left", ipadx=10, ipady=4)
        ttk.Button(buttons, text="Load example", command=self.load_example).pack(side="left", padx=8)
        self.load_example()

    # --- state --------------------------------------------------------------------------------
    def set_state_text(self, text):
        self.state.delete("1.0", "end")
        self.state.insert("1.0", text)

    def fields_changed(self):
        self.set_state_text(json.dumps(self.fields.to_dict(), ensure_ascii=False))

    def tab_changed(self):
        """Opening Fields shows what's in the JSON tab (if it is a JSON object)."""
        if self.state_tabs.index("current") != 0:
            return
        try:
            d = json.loads(self.state.get("1.0", "end"))
        except ValueError:
            return
        if isinstance(d, dict) and d != self.fields.to_dict():
            self.fields.set_dict(d)

    def make_json(self):
        sentence = self.sentence.get("1.0", "end").strip()
        if not sentence:
            return
        self.make_btn.config(state="disabled")
        self.make_status.config(text="loading the chat model ..." if self.chat is None else "reading ...")

        def work():
            try:
                if self.chat is None:
                    self.chat = Engine(self.chat_gguf, self.llama_server,
                                       log_path=os.path.splitext(self.chat_gguf)[0] + "-chat.log")
                    self.after(0, lambda: self.make_status.config(text="reading ..."))
                started = time.perf_counter()
                r = self.chat.post("/completion", {"prompt": extract_prompt(sentence), "n_predict": 300,
                                                   "temperature": 0, "json_schema": FLAT_OBJECT,
                                                   "cache_prompt": True})
                ms = (time.perf_counter() - started) * 1000
                d = json.loads(r["content"])
                self.after(0, lambda: self.json_made(d, ms))
            except Exception as e:
                msg = f"could not make JSON: {e}"
                self.after(0, lambda: (self.make_status.config(text=msg), self.make_btn.config(state="normal")))
        threading.Thread(target=work, daemon=True).start()

    def json_made(self, d, ms):
        self.fields.set_dict(d)
        self.fields_changed()
        self.make_btn.config(state="normal")
        self.make_status.config(text=f"made {len(d)} fields in {ms:.0f} ms - check them in Fields")
        self.state_tabs.select(0)

    # --- options ------------------------------------------------------------------------------
    def refresh_options(self):
        self.opt_list.delete(0, "end")
        for i, (label, desc) in enumerate(self.options):
            self.opt_list.insert("end", f"{LETTERS[i]}. {label}" + (f"  ({desc})" if desc else ""))
        yes_no = self.kind.get() == "yes/no"
        self.count.config(text="yes / no questions need no options" if yes_no else
                          f"{len(self.options)} / {len(LETTERS)} options")

    def add_option(self):
        label, desc = self.opt_label.get().strip(), self.opt_desc.get().strip()
        if not label:
            return
        if len(self.options) >= len(LETTERS):
            messagebox.showinfo("Options", f"At most {len(LETTERS)} options (A-Z).")
            return
        if label in [l for l, _ in self.options]:
            messagebox.showinfo("Options", f'"{label}" is already an option.')
            return
        self.options.append((label, desc))
        self.opt_label.delete(0, "end")
        self.opt_desc.delete(0, "end")
        self.opt_label.focus_set()
        self.refresh_options()

    def bulk_add(self):
        win = tk.Toplevel(self)
        win.title("Paste options")
        win.transient(self.winfo_toplevel())
        ttk.Label(win, padding=8, text="One option per line.  Optional meaning after a |"
                                       "   e.g.   go charge | drive to the charger").pack(anchor="w")
        text = tk.Text(win, width=70, height=14, font=("Consolas", 10))
        text.pack(fill="both", expand=True, padx=8)

        def done():
            for line in text.get("1.0", "end").splitlines():
                label, _, desc = line.partition("|")
                label, desc = label.strip(), desc.strip()
                if label and label not in [l for l, _ in self.options] and len(self.options) < len(LETTERS):
                    self.options.append((label, desc))
            self.refresh_options()
            win.destroy()
        ttk.Button(win, text="Add these", command=done).pack(pady=8)
        text.focus_set()

    def remove_option(self):
        sel = self.opt_list.curselection()
        if sel:
            del self.options[sel[0]]
            self.refresh_options()
            if self.options:
                self.opt_list.selection_set(min(sel[0], len(self.options) - 1))

    def move(self, step):
        sel = self.opt_list.curselection()
        if not sel:
            return
        i, j = sel[0], sel[0] + step
        if 0 <= j < len(self.options):
            self.options[i], self.options[j] = self.options[j], self.options[i]
            self.refresh_options()
            self.opt_list.selection_set(j)

    def clear_options(self):
        self.options = []
        self.refresh_options()

    def kind_changed(self):
        state = "disabled" if self.kind.get() == "yes/no" else "normal"
        for w in [self.opt_label, self.opt_desc, self.add_btn, self.opt_list] + self.side_buttons:
            w.config(state=state)
        self.refresh_options()

    def load_example(self):
        if self.sentence is not None:
            self.sentence.delete("1.0", "end")
            self.sentence.insert("1.0", EXAMPLE["sentence"])
        self.fields.set_dict(EXAMPLE["state"])
        self.fields_changed()
        self.question.delete(0, "end")
        self.question.insert(0, EXAMPLE["question"])
        self.kind.set(EXAMPLE["kind"])
        self.options = list(EXAMPLE["options"])
        self.kind_changed()

    # --- the request --------------------------------------------------------------------------
    def build_request(self):
        """(state, question) as the API takes them; QuestionError says what is missing."""
        raw = self.state.get("1.0", "end").strip()
        if not raw or raw == "{}":
            raise QuestionError("Give the situation first (Fields or JSON / text).")
        try:
            state = json.loads(raw)
        except ValueError:
            state = raw
        question = self.question.get().strip()
        if not question:
            raise QuestionError("Write a question.")
        kind = self.kind.get()
        q = {"type": kind, "instructions": question}
        if kind != "yes/no":
            if len(self.options) < 2:
                raise QuestionError("Add at least two options.")
            if any(d for _, d in self.options):
                q["criteria"] = {l: (d or None) for l, d in self.options}
            else:
                q["criteria"] = [l for l, _ in self.options]
        return state, q

    def close(self):
        if self.chat:
            self.chat.close()


class ResultView(ttk.Frame):
    """Decision, confidence, time and a bar per option."""

    def __init__(self, parent, title=None):
        super().__init__(parent)
        self.columnconfigure(0, weight=1)
        self.probs = []
        r = 0
        if title:
            self.title = ttk.Label(self, text=title, font=("Segoe UI", 11, "bold"))
            self.title.grid(row=r, column=0, sticky="w")
            r += 1
        self.answer = ttk.Label(self, text="-", font=("Segoe UI", 20, "bold"))
        self.answer.grid(row=r, column=0, sticky="w")
        self.confidence = ttk.Label(self, text="", font=("Segoe UI", 12))
        self.confidence.grid(row=r + 1, column=0, sticky="w")
        self.timing = ttk.Label(self, text="", font=("Segoe UI", 12))
        self.timing.grid(row=r + 2, column=0, sticky="w", pady=(0, 6))
        self.chart = tk.Canvas(self, background="white", highlightthickness=1, highlightbackground="#cccccc",
                               height=120)
        self.chart.grid(row=r + 3, column=0, sticky="nsew")
        self.rowconfigure(r + 3, weight=1)
        self.chart.bind("<Configure>", lambda e: self.draw_chart())

    def show(self, q, answer, ms):
        probs = answer_probs(answer)
        best = max(probs, key=probs.get)
        self.answer.config(text=best)
        extra = f"   (expected level {answer['score']:.2f} of {len(probs) - 1})" if q["type"] == "score" else ""
        self.confidence.config(text=f"Confidence: {probs[best] * 100:.1f}%{extra}")
        self.timing.config(text=f"Time to decide: {ms:.0f} ms")
        self.probs = sorted(probs.items(), key=lambda kv: -kv[1])
        self.draw_chart()
        return best, probs[best]

    def message(self, text):
        self.answer.config(text="-")
        self.confidence.config(text=text)
        self.timing.config(text="")
        self.probs = []
        self.draw_chart()

    def draw_chart(self):
        c = self.chart
        c.delete("all")
        if not self.probs:
            c.create_text(12, 12, anchor="nw", text="Probabilities appear here.", fill=MUTED)
            return
        w, h = c.winfo_width(), c.winfo_height()
        n = len(self.probs)
        row = max(12, min(34, (h - 16) // max(n, 1)))
        name_w = min(220, max(80, int(w * 0.35)))
        bar_w = max(40, w - name_w - 80)
        font = ("Segoe UI", 8 if row < 16 else 9 if row < 22 else 10)
        for i, (label, p) in enumerate(self.probs):
            y = 8 + i * row
            text = label if len(label) <= 30 else label[:29] + "..."
            c.create_text(name_w - 6, y + row / 2, anchor="e", text=text, font=font)
            c.create_rectangle(name_w, y + 2, name_w + bar_w * p, y + row - 2, width=0,
                               fill=BAR_BEST if i == 0 else BAR)
            c.create_text(name_w + bar_w * p + 6, y + row / 2, anchor="w", text=f"{p * 100:.1f}%", font=font)


class TestWindow:
    """models: {name shown: GGUF path}. The first one (or `default`) is loaded at start."""

    def __init__(self, root, models, llama_server, default=None, chat_gguf=None, title=f"{MODEL_NAME} - test"):
        self.root, self.models, self.llama_server = root, models, llama_server
        self.engine, self.busy, self.last_request = None, False, ""
        root.title(title)
        root.geometry("1180x800")
        root.minsize(980, 660)
        root.protocol("WM_DELETE_WINDOW", self.close)

        top = ttk.Frame(root, padding=(10, 8))
        top.pack(fill="x")
        ttk.Label(top, text="Model:").pack(side="left")
        names = list(models)
        self.model_var = tk.StringVar(value=default if default in models else (names[0] if names else ""))
        self.model_box = ttk.Combobox(top, textvariable=self.model_var, values=names, state="readonly", width=28)
        self.model_box.pack(side="left", padx=6)
        self.model_box.bind("<<ComboboxSelected>>", lambda e: self.load_model())
        self.status = ttk.Label(top, text="", foreground=MUTED)
        self.status.pack(side="left", padx=10)

        body = ttk.Frame(root, padding=(10, 0, 10, 10))
        body.pack(fill="both", expand=True)
        body.columnconfigure(0, weight=1, uniform="col")
        body.columnconfigure(1, weight=1, uniform="col")
        body.rowconfigure(0, weight=1)
        self.inputs = InputPanel(body, self.decide, chat_gguf=chat_gguf, llama_server=llama_server)
        self.inputs.grid(row=0, column=0, sticky="nsew", padx=(0, 10))

        right = ttk.Frame(body)
        right.grid(row=0, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1)
        right.rowconfigure(0, weight=3)
        right.rowconfigure(2, weight=2)
        self.result = ResultView(right)
        self.result.grid(row=0, column=0, sticky="nsew")
        self.note = ttk.Label(right, text="", foreground=MUTED, wraplength=520, justify="left")
        self.note.grid(row=1, column=0, sticky="w", pady=4)
        tabs = ttk.Notebook(right)
        tabs.grid(row=2, column=0, sticky="nsew", pady=(4, 0))
        tab = ttk.Frame(tabs)
        tabs.add(tab, text="History")
        self.history = tk.Listbox(tab, height=7, font=("Consolas", 9), activestyle="none")
        self.history.pack(fill="both", expand=True)
        tab = ttk.Frame(tabs)
        tabs.add(tab, text="Request JSON")
        self.request = tk.Text(tab, height=7, wrap="none", font=("Consolas", 9))
        self.request.pack(fill="both", expand=True)
        row = ttk.Frame(tab)
        row.pack(fill="x")
        ttk.Button(row, text="Copy", command=self.copy_request).pack(side="left", pady=3)
        ttk.Label(row, text="POST this to http://127.0.0.1:8090/decide", foreground=MUTED).pack(side="left", padx=8)

        root.bind("<Control-Return>", lambda e: self.decide())
        self.load_model()
        self.refresh_request()

    def load_model(self):
        name = self.model_var.get()
        if not name:
            self.status.config(text="no model found")
            return
        self.set_busy(True, f"loading {name} ...")
        old, self.engine = self.engine, None

        def work():
            if old:
                old.close()
            try:
                started = time.perf_counter()
                engine = open_engine(self.models[name], self.llama_server)
                engine.decide("warm up", {"q": {"type": "yes/no"}})
                s = time.perf_counter() - started
                self.root.after(0, lambda: self.model_ready(engine, name, s))
            except Exception as e:
                msg = f"could not load {name}: {e}"
                self.root.after(0, lambda: self.set_busy(False, msg))
        threading.Thread(target=work, daemon=True).start()

    def model_ready(self, engine, name, seconds):
        self.engine = engine
        self.set_busy(False, f"ready: {name}  (CPU, loaded in {seconds:.1f} s)")

    def set_busy(self, busy, text=None):
        self.busy = busy
        self.inputs.decide_btn.config(state="disabled" if busy else "normal")
        self.model_box.config(state="disabled" if busy else "readonly")
        if text is not None:
            self.status.config(text=text)

    def refresh_request(self):
        """Keep the Request JSON tab in step with the inputs (checked twice a second)."""
        try:
            state, q = self.inputs.build_request()
            text = json.dumps({"state": state, "questions": {QUESTION_NAME: q}}, ensure_ascii=False, indent=2)
        except QuestionError as e:
            text = f"// not complete yet: {e}"
        if text != self.last_request:
            self.last_request = text
            self.request.delete("1.0", "end")
            self.request.insert("1.0", text)
        self.refresh_id = self.root.after(500, self.refresh_request)

    def copy_request(self):
        self.root.clipboard_clear()
        self.root.clipboard_append(self.last_request)

    def decide(self):
        if self.busy or not self.engine:
            return
        try:
            state, q = self.inputs.build_request()
        except QuestionError as e:
            messagebox.showwarning("Decide", str(e))
            return
        self.set_busy(True, "deciding ...")
        engine = self.engine

        def work():
            try:
                started = time.perf_counter()
                result = engine.decide(state, {QUESTION_NAME: q})
                ms = (time.perf_counter() - started) * 1000
                self.root.after(0, lambda: self.show(q, result, ms))
            except Exception as e:
                msg = f"error: {e}"
                self.root.after(0, lambda: self.set_busy(False, msg))
        threading.Thread(target=work, daemon=True).start()

    def show(self, q, result, ms):
        best, conf = self.result.show(q, result["answers"][QUESTION_NAME], ms)
        coverage = result["coverage"][QUESTION_NAME]
        self.note.config(text=f"Coverage {coverage:.2f}: how much of the model's probability landed on the options "
                              "before renormalising (near 1.0 is normal). A second question about the SAME state "
                              "is faster, because the model keeps the state it has already read.")
        self.history.insert(0, f"{time.strftime('%H:%M:%S')}  {ms:6.0f} ms  {conf * 100:5.1f}%  "
                               f"{best}  <-  {q['instructions']}")
        self.set_busy(False, f"ready: {self.model_var.get()}")

    def close(self):
        self.root.after_cancel(self.refresh_id)
        if self.engine:
            self.engine.close()
        self.inputs.close()
        self.root.destroy()


def main():
    """python decide_ui.py - JustDecide*.gguf next to this file."""
    ap = argparse.ArgumentParser(description=f"{MODEL_NAME} test window")
    ap.add_argument("--model", help="a GGUF file (default: the JustDecide*.gguf next to this file)")
    args = ap.parse_args()
    root = tk.Tk()
    if args.model:
        models = {os.path.splitext(os.path.basename(args.model))[0]: args.model}
    else:
        models = ensure_model(root)
    if not models:
        root.destroy()
        return
    TestWindow(root, models, default_files()[1])
    root.mainloop()


def models_here():
    """{name: path} of the JustDecide*.gguf files next to this file (or in its model folder)."""
    found = sorted(glob.glob(os.path.join(HERE, f"{MODEL_NAME}*.gguf")) +
                   glob.glob(os.path.join(HERE, "model", f"{MODEL_NAME}*.gguf")))
    return {os.path.splitext(os.path.basename(p))[0]: p for p in found}


def ensure_model(root):
    """The models next to this file; when there are none, offer to download JustDecide from Hugging Face."""
    models = models_here()
    if models:
        return models
    root.withdraw()
    if not messagebox.askyesno(MODEL_NAME, f"{MODEL_FILE} is not in\n{HERE}\n\nDownload it now from "
                                           f"huggingface.co/{HF_REPO}?  (about 640 MB, once)"):
        return {}
    win = tk.Toplevel(root)
    win.title(f"Downloading {MODEL_NAME}")
    ttk.Label(win, text=MODEL_URL, padding=(12, 10, 12, 4)).pack(anchor="w")
    bar = ttk.Progressbar(win, length=420, maximum=1000)
    bar.pack(padx=12, pady=4)
    text = ttk.Label(win, text="starting ...", padding=(12, 0, 12, 10))
    text.pack(anchor="w")
    outcome = {}

    def show(done, total):
        bar.config(value=1000 * done / total if total else 0)
        text.config(text=f"{done / 1e6:.0f} of {total / 1e6:.0f} MB")

    def work():
        try:
            download_model(progress=lambda d, t: root.after(0, show, d, t))
        except Exception as e:
            outcome["error"] = str(e)
        root.after(0, win.destroy)
    threading.Thread(target=work, daemon=True).start()
    root.wait_window(win)
    if "error" in outcome:
        messagebox.showerror(MODEL_NAME, f"Download failed: {outcome['error']}\n\nYou can also download "
                                         f"{MODEL_FILE} yourself from huggingface.co/{HF_REPO} into {HERE}")
        return {}
    root.deiconify()
    return models_here()


if __name__ == "__main__":
    main()

