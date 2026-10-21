# How JustDecide works

JustDecide is [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) trained so that its first answer
token is always the letter of one of your options. We then read the probabilities of those letters
directly and never let it generate. This page explains each step. You can reproduce it with any
llama.cpp-based runtime.

## 1. The prompt

Every question becomes the same prompt (Qwen3's chat format with thinking switched off).
`justdecide.build_prompt()` builds it:

```
<|im_start|>system
You are a decision function. Read the state, then answer the question by choosing exactly one option. Reply with the option's letter only.<|im_end|>
<|im_start|>user
[State]
{state}

[Question]
{question}

[Options]
A. {option 1}
B. {option 2} ({what option 2 means})
C. {option 3}<|im_end|>
<|im_start|>assistant
<think>

</think>

```

- **`{state}`:** text goes in as it is. JSON goes in as `json.dumps(state, ensure_ascii=False, indent=1)`.
- **Options:** `A. label`, or `A. label (meaning)` when you give a meaning.
- **yes/no:** the options are `A. yes` and `B. no`. A meaning, if you give one, is added in brackets.
- **score:** the levels in your order, lowest first.
- **The prompt ends exactly where the reply begins,** so the next token is the answer.

JustDecide was trained with exactly this format. Other formats still give answers, but worse ones.

## 2. One pass, letters only

The model reads the prompt once, which llama.cpp calls a prefill. At the last position we take the
model's probability for the tokens `A`, `B`, `C`, … (one token each in Qwen's vocabulary), keep only as
many letters as there are options, and divide by their sum. That gives the answer's `probabilities`:

- **choice:** the option with the highest probability.
- **score:** `level` is the most likely level, and `score = Σ i·p(i)` is the expected level (0 = lowest).
- **yes/no:** `noul` = p(yes).

Before renormalising, the letters' share of the whole vocabulary is reported as **`coverage`**. A
decision model puts almost everything on the letters (coverage near 1.0). A low coverage is a warning:
the question or the options didn't make sense to the model.

Nothing is sampled, so the same input always gives the same probabilities.

## 3. Calibrated probabilities

After training, the model's scores were calibrated on held-out examples so that its confidence matches
its accuracy. The calibration error (ECE) on the test set is about 0.03, meaning a stated 80% is right
roughly 77–83% of the time. The correction is built into the weights, so the raw probabilities are
already calibrated and you need no extra step.

## 4. Asking several questions about one state

The state comes first in the prompt. When you ask several questions about the same state, the runtime
keeps what it has already read (llama.cpp's cache) and reads only the new question and options. That's
why the second and later questions take a fraction of the first.

## 5. Running it

`justdecide.py` has two engines with the same `decide()`:

- **`LocalEngine`:** the model inside your Python process, via
  [llama-cpp-python](https://github.com/abetlen/llama-cpp-python). This is the default and needs only
  `pip install -r requirements.txt`.
- **`Engine`:** the model in a separate `llama-server` process (from
  [llama.cpp](https://github.com/ggml-org/llama.cpp)), which we talk to over HTTP. Use it if you already
  run llama-server, or have a GPU build of it.

`open_engine(gguf, llama_server=None)` picks `Engine` when you pass a llama-server executable, and
`LocalEngine` otherwise.

### With your own llama.cpp setup

Build the prompt above and ask for **one** token with its top probabilities. With llama-server:

```bash
curl http://127.0.0.1:8080/completion -d '{"prompt": "<the prompt>", "n_predict": 1, "n_probs": 64, "temperature": 0}'
```

Then read `completion_probabilities[0].top_logprobs`, keep the entries whose token is `A`, `B`, …, and
renormalise. Any runtime that exposes next-token probabilities can be used the same way.

## 6. The model file

`JustDecide-q8_0.gguf` is the full model with its weights in 8-bit (q8_0) and its name written inside
the file. 8-bit keeps the answers practically identical to full precision at half the size. It loads in
any llama.cpp build that supports Qwen3 (llama.cpp from May 2025 on, llama-cpp-python 0.3.9 or later).
