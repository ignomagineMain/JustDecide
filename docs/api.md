# JustDecide API reference

## HTTP

Start it with `python justdecide.py`. The options are:

| option | meaning |
|---|---|
| `--port 8090` | the port to listen on |
| `--host 127.0.0.1` | `0.0.0.0` accepts other computers on your network |
| `--threads N` | CPU threads (default: half the logical cores) |
| `--model file.gguf` | another model file |
| `--download` | first download `JustDecide-q8_0.gguf` from Hugging Face |
| `--llama-server path` | run the model in llama-server rather than in-process |

### `POST /decide`

```json
{
  "state": {"battery_pct": 14, "order_queue": 2, "position": "aisle 3", "lidar": "clear"},
  "questions": {
    "next":      {"type": "choice", "instructions": "What should the AGV do next?",
                  "criteria": {"pick next order": null, "go charge": "drive to the charger now", "go to parking": null}},
    "battery":   {"type": "score", "instructions": "How much battery is left?",
                  "criteria": ["very low", "low", "ok", "full"]},
    "go_charge": {"type": "yes/no"}
  }
}
```

- **`state`:** a string, or any JSON value. It is required and can't be empty.
- **`questions`:** `{name: question}`, with one or more questions. The names are yours and come back in the answer.
- **`type`:**
  - `choice` (also accepted: `choose`, `option`)
  - `score` (also: `scale`, `level`)
  - `yes/no` (also: `yesno`, `bool`, `boolean`, `noul`)
- **`instructions`:** the question text. If you leave it out, the name is asked (`go_charge` → "go charge?").
- **`criteria`:**
  - **choice:** a list of 2–26 options, or `{option: meaning or null}`.
  - **score:** a list of 2–26 levels, **lowest first**.
  - **yes/no:** optional `{"true": "...", "false": "..."}`.

Answer (`200`):

```json
{
  "answers": {
    "next":      {"choice": "go charge", "probabilities": {"pick next order": 0.2535, "go charge": 0.6668, "go to parking": 0.0798}},
    "battery":   {"score": 1.3875, "level": "low", "probabilities": {"very low": 0.1347, "low": 0.4158, "ok": 0.377, "full": 0.0726}},
    "go_charge": {"noul": 0.3959}
  },
  "ms": 2370,
  "coverage": {"next": 0.98, "battery": 0.99, "go_charge": 0.95}
}
```

- **`noul`:** the probability of *yes*.
- **`score`:** the expected level (0 = lowest level).
- **`ms`:** the time for all the questions in the request.
- **`coverage`:** see [how it works](how-it-works.md#2-one-pass-letters-only).

Errors:
- `400 {"error": "..."}` for a malformed request (missing state, unknown type, fewer than 2 options, …).
- `500 {"error": "..."}` if the model failed.

### `GET /health`

Returns `{"status": "ok", "model": "JustDecide"}`.

## Python

```python
from justdecide import open_engine, download_model

path = download_model()                  # once: fetches JustDecide-q8_0.gguf next to justdecide.py
engine = open_engine(path)               # LocalEngine (llama-cpp-python); open_engine(path, "llama-server") for Engine
r = engine.decide(state, questions)      # the same dicts as the HTTP request and answer
engine.close()
```

- **`decide()`** can be called from several threads; the calls run one at a time.
- **Keep one engine** and reuse it, so you load the model once and the state cache stays warm.
- **`open_engine(gguf, llama_server=None, threads=None)`:** `Engine` if `llama_server` is an existing
  executable, otherwise `LocalEngine`.
- **`LocalEngine(gguf, threads=None, n_ctx=4096)`** and **`Engine(gguf, llama_server, port=None,
  threads=4, log_path=None)`:** the two engines. Both have `decide(state, questions)` and `close()`.
- **`build_prompt(state, question, labels, descriptions=None)`:** the exact prompt. It's useful if you
  run the model yourself.
- **`parse_question(name, question)`** returns `(kind, text, labels, descriptions)` and raises
  `QuestionError` for a bad question.
