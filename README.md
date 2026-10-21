# JustDecide

**A small language model that only decides.** You give JustDecide a situation and a question with the
answers you allow. It returns the probability of each answer in a single pass, on an ordinary CPU,
fully offline. It never writes free text, so its answer is always one of your options, and its
probabilities are calibrated: when it says 80%, it is right about 80% of the time.

Made by **[Ignomagine](#about-ignomagine)** for robots, the machines around them and the people who
run them.

**Model:** [huggingface.co/ignomaginerobotics/JustDecide](https://huggingface.co/ignomaginerobotics/JustDecide), a 639 MB GGUF.
**Licence:** Apache-2.0.

```
state     {"battery_pct": 14, "order_queue": 2, "position": "aisle 3", "lidar": "clear"}
question  What should the AGV do next?
options   pick next order | go charge | go to parking

answer    go charge 62%  ·  pick next order 26%  ·  go to parking 12%        (about 1 s on a laptop CPU)
```

---

## Contents

- [Why a decision model](#why-a-decision-model)
- [Quick start](#quick-start)
- [The test window](#the-test-window)
- [Use it in your code](#use-it-in-your-code)
- [Writing good questions](#writing-good-questions)
- [How it works](#how-it-works)
- [Accuracy and speed](#accuracy-and-speed)
- [Limitations](#limitations)
- [About Ignomagine](#about-ignomagine)
- [Licence and credits](#licence-and-credits)

## Why a decision model

A chat model is the wrong tool when a machine has to choose:

- **Its answer can't be predicted.** It may answer in a sentence, choose something you never offered,
  or explain itself for half a minute.
- **It gives one answer with no confidence.** You can't tell a sure "stop" from a coin-flip "stop".
- **It's slow.** It writes its answer one word at a time.

JustDecide reads the situation once and then scores only your options. That brings four things:

- **Bounded:** the answer is always one of your options, never anything else.
- **Calibrated:** the probabilities mean what they say, so you can set a threshold (act above 0.8,
  otherwise ask a person).
- **Fast and small:** 0.6 billion parameters, about 1 s on a 4-core laptop CPU, no GPU, no internet,
  and it runs on the robot itself.
- **Three question types:** pick one (*choice*), place on a scale (*score*), or *yes / no*.

## Quick start

You need Python 3.8–3.13, 64-bit, on Windows, Linux (x86_64 or ARM64) or macOS (Apple chips).

```bash
git clone https://github.com/ignomagineMain/JustDecide.git
cd JustDecide
pip install -r requirements.txt
python decide_ui.py
```

- **The model downloads on first run.** It's 639 MB, and the window offers to fetch it. To do it
  yourself: `python justdecide.py --download`, or download `JustDecide-q8_0.gguf` from
  [Hugging Face](https://huggingface.co/ignomaginerobotics/JustDecide) into this folder.
- **Nothing is compiled.** `requirements.txt` installs llama.cpp as a ready-made build for your system.
- **On Linux,** if the window doesn't open, run `sudo apt install python3-tk`.

| File | What it is |
|---|---|
| `decide_ui.py` | the test window |
| `justdecide.py` | the runtime: load the model, ask questions, or run an HTTP API |
| `justdecide_ui.py` | the test window's code |
| `examples/` | a quick start, an API client and an example request |
| `docs/` | [how it works](docs/how-it-works.md) and the [API reference](docs/api.md) |

## The test window

`python decide_ui.py`

1. **Situation.** In *Fields*, enter name/value rows (`battery_pct` = `14`, `lidar` = `clear`). Numbers,
   `true`/`false` and empty values (empty = null) become proper JSON. Or write JSON or plain text in
   *JSON / text*.
2. **Question.** For example, *What should the AGV do next?*
3. **Type.** *choice*, *score* (levels from lowest to highest) or *yes / no*.
4. **Options.** Add up to 26, each with an optional meaning ("go charge" means "drive to the charger
   now"). *Paste list…* takes one per line, in the form `option | meaning`.
5. **Decide** (Ctrl+Enter). It shows the decision, its confidence, a bar for every option and the time
   it took. *History* keeps your earlier questions. *Request JSON* shows the same question as an API
   request, ready to copy.

## Use it in your code

### In Python

```python
from justdecide import open_engine

engine = open_engine("JustDecide-q8_0.gguf")          # loads in a second or two; keep it and reuse it

result = engine.decide(
    {"battery_pct": 14, "order_queue": 2, "position": "aisle 3", "lidar": "clear"},
    {
        "next":      {"type": "choice", "instructions": "What should the AGV do next?",
                      "criteria": {"pick next order": None,
                                   "go charge": "drive to the charger now",
                                   "go to parking": None}},
        "battery":   {"type": "score", "instructions": "How much battery is left?",
                      "criteria": ["very low", "low", "ok", "full"]},
        "go_charge": {"type": "yes/no"},
    })

result["answers"]["next"]       # {"choice": "go charge", "probabilities": {"pick next order": 0.25, "go charge": 0.67, ...}}
result["answers"]["battery"]    # {"score": 1.39, "level": "low", "probabilities": {...}}
result["answers"]["go_charge"]  # {"noul": 0.40}    <- probability of "yes"
result["ms"]                    # time for all three questions
```

### Over HTTP

```bash
python justdecide.py                      # http://127.0.0.1:8090/decide   (--port, --host 0.0.0.0, --threads)
```
```bash
curl -X POST http://127.0.0.1:8090/decide -H "Content-Type: application/json" --data-binary @examples/request.json
```

The request is `{"state": ..., "questions": {...}}`, the same as above, and the answer has the same
shape. Any language that can send JSON can use it. See [docs/api.md](docs/api.md) and
`examples/call_api.py`.

### Question types

| type | `criteria` | answer |
|---|---|---|
| `choice` | a list of options, or `{option: meaning}` | `choice` (the most likely option) + `probabilities` |
| `score` | the levels, **lowest first** | `level` (the most likely), `score` (the expected level, 0 … n-1) + `probabilities` |
| `yes/no` | optional: `{"true": "what yes means", "false": "what no means"}` | `noul` = probability of yes |

- **Untitled questions:** a question without `"instructions"` asks its own name, so `"go_charge"` becomes
  "go charge?".
- **Several questions:** ask several questions about one state in one call. The state is read once and
  each further question costs a fraction of the first.
- **coverage:** each answer also carries a `coverage` value, the share of the model's probability that
  landed on your options at all. Near 1.0 is normal. A low value means the question or the options
  confused it.

## Writing good questions

- **Name your facts and put units in the names:** `battery_pct`, `distance_m`, `motor_temp_c`.
  JSON works better than prose for machine states.
- **Keep options short and clearly different.** Give a meaning when a label alone is ambiguous:
  `"replan": "take another route"`.
- **Use 2–5 options when you can.** Up to 26 work, but the model was trained with 2–5 and is less
  sure with many.
- **Use `score` for anything with an order** (urgency, how full, how good) and `yes/no` for a single fact.
- **Use the probability:** act when it is high, and ask a person or gather more data when it is not.

## How it works

JustDecide is **Qwen3-0.6B** (Alibaba's open model), trained at Ignomagine to answer typed decisions.

1. **The prompt.** Your state, question and options become one fixed prompt. The options are lettered
   A, B, C…
2. **One pass.** The model reads it once. The scores for the letters A, B, C… at the point where the
   answer would begin become the probabilities, and nothing is generated.
3. **Calibration.** The scores were calibrated after training so the probabilities are honest.
4. **Running it.** The model runs with [llama.cpp](https://github.com/ggml-org/llama.cpp), via
   llama-cpp-python, as an 8-bit GGUF.

Details: [docs/how-it-works.md](docs/how-it-works.md).

## Accuracy and speed

Tested on 1,212 examples JustDecide never saw in training: four public decision tasks and robot
decisions written at Ignomagine.

| | Qwen3-0.6B before training | **JustDecide v1** |
|---|---|---|
| right answer, all tasks | 53.8% | **72.9%** |
| news topic, 4 options | 79.3% | 85.7% |
| does the passage say yes or no | 71.3% | 79.7% |
| rating, 5 levels | 24.3% | 47.0% |
| does B follow from A | 40.0% | 79.7% |
| robot decisions | 58% | 67% |
| calibration error (ECE, lower is better) | 0.37 | **0.03** |

**Speed** on a 4-core laptop CPU (AMD Ryzen 5 3550H), with no GPU:
- **First question about a new situation:** about 1–1.5 s.
- **Each further question about the same situation:** about 0.5 s.
- **Loading the model:** about 2 s.

## Limitations

- **English only.**
- **A small model:** good at reading a situation and choosing among options, weak on world knowledge
  and multi-step reasoning. Put the facts it needs into the state.
- **2–5 options work best.** More work, with lower confidence.
- **It can be wrong, and confidently so in situations unlike its training.** Treat it as a fast,
  honest-but-fallible advisor. For safety-critical actions (motion, lifting, people nearby), keep your
  hard limits and interlocks in code, and use JustDecide for the judgement calls between them.

## About Ignomagine

**Ignomagine Private Limited** is a robotics company in Bengaluru, India. We build autonomous mobile
robots and the software that runs them:
- **ignoAPI:** the I/O and control layer that connects a robot's motors, sensors and apps.
- **IgnoSIM:** a robot simulator.
- **Tools** for motor control and camera-based alignment.

JustDecide came out of that work. A working robot makes small judgement calls all day: should it
charge now, is this obstacle going to move, who should get this alert. We wanted a model that makes
those calls quickly, on the robot, with an honest confidence.

## Licence and credits

- **JustDecide**, the model and this code: © 2026 Ignomagine Private Limited, under the
  [Apache License 2.0](LICENSE). See [NOTICE](NOTICE).
- **Base model:** [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B), © Alibaba Cloud, Apache-2.0.
- **Runtime:** [llama.cpp](https://github.com/ggml-org/llama.cpp) via
  [llama-cpp-python](https://github.com/abetlen/llama-cpp-python), MIT.

If you use JustDecide in your work:

```bibtex
@software{justdecide_2026,
  title  = {JustDecide: a small calibrated decision model},
  author = {{Ignomagine Private Limited}},
  year   = {2026},
  url    = {https://huggingface.co/ignomaginerobotics/JustDecide}
}
```
