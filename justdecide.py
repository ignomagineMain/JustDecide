"""
JustDecide - a decision model. It reads a situation (the "state") and a typed question and returns
the probability of every allowed answer in one pass. It never writes free text.

    python justdecide.py                       HTTP API on http://127.0.0.1:8090/decide
    python justdecide.py --port 9000 --threads 8

POST /decide
    {"state": "text or any JSON",
     "questions": {
        "next":    {"type": "choice", "instructions": "What should the AGV do next?",
                    "criteria": ["pick next order", "go charge", "go to parking"]},
        "urgency": {"type": "score",  "criteria": ["low", "medium", "high"]},        <- lowest first
        "refund":  {"type": "yes/no", "instructions": "Is the customer asking for a refund?"}}}
->  {"answers": {"next": {"choice": "go charge", "probabilities": {...}},
                 "urgency": {"score": 1.62, "level": "high", "probabilities": {...}},
                 "refund": {"noul": 0.06}},
     "ms": 410, "coverage": {...}}

"criteria" may also say what each option means: {"go charge": "drive to the charger now", ...}.
A question with no "instructions" asks its name: "go_charge" -> "go charge?". Up to 26 options (A-Z).

The model runs on the CPU with llama.cpp: in this process with llama-cpp-python (LocalEngine,
pip install -r requirements.txt), or in a llama-server.exe process (Engine) - same answers.
In Python:  from justdecide import open_engine;  open_engine("JustDecide-q8_0.gguf").decide(state, questions)
"""
import argparse
import atexit
import glob
import json
import math
import os
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODEL_NAME = "JustDecide"
HERE = os.path.dirname(os.path.abspath(__file__))
LETTERS = [chr(ord("A") + i) for i in range(26)]
YES_NO = ["yes", "no"]
TOP = 64                 # how many of the most likely next tokens llama-server reports

SYSTEM = ("You are a decision function. Read the state, then answer the question by choosing "
          "exactly one option. Reply with the option's letter only.")

# Kinds of question: pick one option, a level on a scale, or yes/no.
KINDS = {"choice": "choice", "choose": "choice", "option": "choice",
         "score": "score", "scale": "score", "level": "score",
         "noul": "noul", "yesno": "noul", "bool": "noul", "boolean": "noul"}


class QuestionError(ValueError):
    pass


def state_text(state):
    """Text states go in as they are; anything else as JSON."""
    return state if isinstance(state, str) else json.dumps(state, ensure_ascii=False, indent=1)


def build_prompt(state, question, labels, descriptions=None):
    """
    The prompt JustDecide was trained with: Qwen3's chat format with thinking switched off, ending
    where the reply begins, so the next token is the answer letter. The state comes first, so several
    questions about one state share that start and llama-server reads the state only once.
    Options: "A. label" or "A. label (description)".
    """
    descriptions = descriptions or [None] * len(labels)
    lines = [f"{LETTERS[i]}. {l} ({d})" if d else f"{LETTERS[i]}. {l}"
             for i, (l, d) in enumerate(zip(labels, descriptions))]
    user = f"[State]\n{state_text(state)}\n\n[Question]\n{question}\n\n[Options]\n" + "\n".join(lines)
    return (f"<|im_start|>system\n{SYSTEM}<|im_end|>\n"
            f"<|im_start|>user\n{user}<|im_end|>\n"
            f"<|im_start|>assistant\n<think>\n\n</think>\n\n")


def parse_question(name, q):
    """
    One API question -> (kind, question text, labels, descriptions).
      choice: "criteria": ["a", "b"] or {"a": "what a means", ...}
      score:  "criteria": ["low", "medium", "high"]   (lowest first)
      yes/no: optional "criteria": {"true": "what yes means", "false": "what no means"}
    """
    if not isinstance(q, dict):
        raise QuestionError(f'question "{name}" must be an object with a "type"')
    kind = KINDS.get("".join(c for c in str(q.get("type", "")).lower() if c.isalnum()))
    if kind is None:
        raise QuestionError(f'question "{name}" needs "type": "choice", "score" or "yes/no"')
    text = q.get("instructions") or name.replace("_", " ").strip() + "?"
    crit = q.get("criteria")
    if kind == "noul":
        crit = crit if isinstance(crit, dict) else {}
        return kind, text, YES_NO, [crit.get("true", crit.get("yes")), crit.get("false", crit.get("no"))]
    if isinstance(crit, dict):
        labels = [str(k) for k in crit]
        descriptions = [v if isinstance(v, str) or v is None else json.dumps(v, ensure_ascii=False)
                        for v in crit.values()]
    elif isinstance(crit, list):
        labels = [c if isinstance(c, str) else json.dumps(c, ensure_ascii=False) for c in crit]
        descriptions = [None] * len(labels)
    else:
        labels, descriptions = [], []
    if len(labels) < 2:
        raise QuestionError(f'{kind} question "{name}" needs "criteria" with at least two options')
    if len(labels) > len(LETTERS):
        raise QuestionError(f'question "{name}" has more than {len(LETTERS)} options')
    return kind, text, labels, descriptions


def answer_from_probs(kind, labels, p):
    """Probabilities over the options -> the answer: choice / score (+ level) / noul = P(yes)."""
    probs = {l: round(float(v), 4) for l, v in zip(labels, p)}
    best = max(range(len(p)), key=lambda i: p[i])
    if kind == "noul":
        return {"noul": probs["yes"]}
    if kind == "score":
        # Expected level: 0 = lowest ... n-1 = highest.
        return {"score": round(sum(i * float(v) for i, v in enumerate(p)), 4), "level": labels[best],
                "probabilities": probs}
    return {"choice": labels[best], "probabilities": probs}


def as_api_question(ex):
    """A test example {kind, question, labels, descriptions} written as an API question."""
    q = {"type": ex["kind"], "instructions": ex["question"]}
    if ex["kind"] == "noul":
        if any(ex["descriptions"]):
            q["criteria"] = {"true": ex["descriptions"][0], "false": ex["descriptions"][1]}
    elif any(ex["descriptions"]):
        q["criteria"] = dict(zip(ex["labels"], ex["descriptions"]))
    else:
        q["criteria"] = ex["labels"]
    return q


def answer_probs(answer):
    """{option: probability} for any answer shape (yes/no answers only carry P(yes))."""
    if "noul" in answer:
        return {"yes": answer["noul"], "no": round(1 - answer["noul"], 4)}
    return answer["probabilities"]


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _Decider:
    """decide() for any engine that has option_probs(prompt, n) and a lock."""

    def decide(self, state, questions):
        """{"answers": {name: answer}, "ms": total, "coverage": {name: share of probability on the options}}"""
        parsed = {name: parse_question(name, q) for name, q in questions.items()}
        start = time.time()
        answers, coverage = {}, {}
        with self.lock:
            for name, (kind, text, labels, descriptions) in parsed.items():
                p, cov = self.option_probs(build_prompt(state, text, labels, descriptions), len(labels))
                answers[name] = answer_from_probs(kind, labels, p)
                coverage[name] = round(cov, 4)
        return {"answers": answers, "ms": int((time.time() - start) * 1000), "coverage": coverage}


class LocalEngine(_Decider):
    """
    The model inside this Python process, with llama-cpp-python (pip install -r requirements.txt).
    Reads the prompt once, keeps what it read, and next time only reads what changed - so several
    questions about the same state are cheap.
    """

    def __init__(self, gguf, threads=None, n_ctx=4096):
        import numpy as np
        from llama_cpp import Llama
        self.np, self.gguf = np, gguf
        self.llm = Llama(model_path=gguf, n_ctx=n_ctx, n_threads=threads, n_batch=512, verbose=False)
        self.lock = threading.Lock()
        self.letter_ids = []
        for letter in LETTERS:
            ids = self.llm.tokenize(letter.encode(), add_bos=False, special=False)
            assert len(ids) == 1, f"letter {letter} is {len(ids)} tokens in this model"
            self.letter_ids.append(ids[0])

    def option_probs(self, prompt, n):
        np, llm = self.np, self.llm
        tokens = llm.tokenize(prompt.encode("utf-8"), add_bos=False, special=True)
        keep = 0                                             # how much of the last prompt is the same
        for old, new in zip(llm.input_ids[:llm.n_tokens], tokens):
            if old != new:
                break
            keep += 1
        llm.n_tokens = min(keep, len(tokens) - 1)            # read at least the last token again
        llm.eval(tokens[llm.n_tokens:])
        z = np.ctypeslib.as_array(llm._ctx.get_logits(), shape=(llm.n_vocab(),)).astype(np.float64)
        top = z.max()
        log_total = top + math.log(np.exp(z - top).sum())   # over the whole vocabulary
        raw = [math.exp(z[i] - log_total) for i in self.letter_ids[:n]]
        coverage = sum(raw)
        return ([x / coverage for x in raw] if coverage > 0 else [1.0 / n] * n), coverage

    def close(self):
        llm, self.llm = getattr(self, "llm", None), None
        if llm is not None and hasattr(llm, "close"):
            llm.close()


def open_engine(gguf, llama_server=None, threads=None):
    """llama-server if its executable is given and exists, otherwise llama-cpp-python in this process."""
    if llama_server and os.path.exists(llama_server):
        return Engine(gguf, llama_server, threads=threads or 4)
    return LocalEngine(gguf, threads=threads)


class Engine(_Decider):
    """One llama-server process holding one GGUF. decide() is safe to call from several threads."""

    def __init__(self, gguf, llama_server, port=None, threads=4, log_path=None):
        self.gguf = gguf
        self.url = f"http://127.0.0.1:{port or free_port()}"
        self.lock = threading.Lock()          # one llama-server slot: decisions go one at a time
        log_path = log_path or os.path.splitext(gguf)[0] + "-server.log"
        log = open(log_path, "w")
        self.proc = subprocess.Popen([llama_server, "-m", gguf, "--host", "127.0.0.1",
                                      "--port", self.url.rsplit(":", 1)[1], "-c", "4096", "-np", "1",
                                      "-t", str(threads), "--no-webui"], stdout=log, stderr=subprocess.STDOUT)
        atexit.register(self.close)
        for _ in range(240):
            if self.proc.poll() is not None:
                raise RuntimeError(f"llama-server stopped - see {log_path}")
            try:
                if self._get("/health").get("status") == "ok":
                    break
            except (urllib.error.URLError, ConnectionError):
                pass
            time.sleep(0.5)
        else:
            raise RuntimeError(f"llama-server did not start - see {log_path}")
        self.letter_ids = [self.post("/tokenize", {"content": l})["tokens"][0] for l in LETTERS]

    def _get(self, path):
        with urllib.request.urlopen(self.url + path, timeout=5) as r:
            return json.load(r)

    def post(self, path, body):
        req = urllib.request.Request(self.url + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=300) as r:
            return json.load(r)

    def option_probs(self, prompt, n):
        """Ask for ONE token with its top probabilities; keep the option letters and renormalise."""
        r = self.post("/completion", {"prompt": prompt, "n_predict": 1, "n_probs": TOP,
                                      "cache_prompt": True, "temperature": 0})
        top = {t["id"]: t["logprob"] for t in r["completion_probabilities"][0]["top_logprobs"]}
        raw = [math.exp(top[i]) if i in top else 0.0 for i in self.letter_ids[:n]]
        coverage = sum(raw)
        return ([x / coverage for x in raw] if coverage > 0 else [1.0 / n] * n), coverage

    def close(self):
        if self.proc.poll() is None:
            self.proc.kill()


def handler_for(engine):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code, obj):
            body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/health":
                return self._send(200, {"status": "ok", "model": MODEL_NAME})
            self._send(200, {"usage": 'POST /decide {"state": ..., "questions": {"name": {"type": '
                                      '"choice"|"score"|"yes/no", "instructions": "...", "criteria": [...]}}}'})

        def do_POST(self):
            if self.path != "/decide":
                return self._send(404, {"error": "POST /decide"})
            try:
                req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                if not isinstance(req, dict) or req.get("state") in (None, ""):
                    raise QuestionError('needs a "state": the situation to decide about')
                if not isinstance(req.get("questions"), dict) or not req["questions"]:
                    raise QuestionError('needs "questions": {"name": {"type": ..., ...}}')
                result = engine.decide(req["state"], req["questions"])
            except (ValueError, QuestionError) as e:
                return self._send(400, {"error": str(e)})
            except Exception as e:
                return self._send(500, {"error": f"{type(e).__name__}: {e}"})
            print(f"{result['ms']:5d} ms  " + "  ".join(
                f"{k}={v.get('choice', v.get('level', v.get('noul')))}" for k, v in result["answers"].items()),
                flush=True)
            self._send(200, result)

        def log_message(self, *a):
            pass

    return Handler


def serve(engine, port=8090, host="127.0.0.1"):
    try:
        server = ThreadingHTTPServer((host, port), handler_for(engine))
    except OSError as e:
        engine.close()
        raise SystemExit(f"port {port} is in use by another program ({e}) - start with --port 9000 (or another)")
    print(f"{MODEL_NAME} API on http://{host}:{port}/decide   (Ctrl+C to stop)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        engine.close()


def default_files():
    """(model, llama-server or None): JustDecide*.gguf next to this file (or in model\\), llama\\llama-server.exe."""
    models = sorted(glob.glob(os.path.join(HERE, f"{MODEL_NAME}*.gguf")) +
                    glob.glob(os.path.join(HERE, "model", f"{MODEL_NAME}*.gguf")))
    llama = os.path.join(HERE, "llama", "llama-server.exe")
    return (models[0] if models else None), (llama if os.path.exists(llama) else None)


HF_REPO = "ignomaginerobotics/JustDecide"               # where the model is published
MODEL_FILE = f"{MODEL_NAME}-q8_0.gguf"
MODEL_URL = f"https://huggingface.co/{HF_REPO}/resolve/main/{MODEL_FILE}"


def download_model(folder=HERE, progress=None):
    """
    Download JustDecide-q8_0.gguf from Hugging Face into `folder` (about 640 MB) and return its path.
    progress(done_bytes, total_bytes) is called as it goes. A broken download is never left as the model.
    """
    path = os.path.join(folder, MODEL_FILE)
    part = path + ".part"
    with urllib.request.urlopen(urllib.request.Request(MODEL_URL, headers={"User-Agent": "justdecide"}),
                                timeout=60) as r, open(part, "wb") as f:
        total, done = int(r.headers.get("Content-Length") or 0), 0
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if progress:
                progress(done, total)
    if total and done != total:
        raise IOError(f"download stopped at {done} of {total} bytes - try again")
    os.replace(part, path)
    return path


def main():
    model, llama = default_files()
    ap = argparse.ArgumentParser(description=f"{MODEL_NAME} decision API")
    ap.add_argument("--model", default=model, help="the GGUF file")
    ap.add_argument("--download", action="store_true", help=f"download {MODEL_FILE} from huggingface.co/{HF_REPO}")
    ap.add_argument("--llama-server", default=llama, help="use this llama-server executable "
                                                           "(default: llama-cpp-python in this process)")
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--host", default="127.0.0.1", help="0.0.0.0 to accept other computers")
    ap.add_argument("--threads", type=int, help="CPU threads (default: half the logical cores)")
    args = ap.parse_args()
    if args.download:
        print(f"downloading {MODEL_URL}", flush=True)
        args.model = download_model(progress=lambda d, t: print(f"\r  {d / 1e6:.0f} / {t / 1e6:.0f} MB", end="",
                                                                flush=True))
        print()
    if not args.model or not os.path.exists(args.model):
        ap.error(f"no model file - run  python {os.path.basename(__file__)} --download  "
                 f"or put {MODEL_FILE} next to it")
    print(f"loading {args.model}", flush=True)
    serve(open_engine(args.model, args.llama_server, args.threads), args.port, args.host)


if __name__ == "__main__":
    main()
