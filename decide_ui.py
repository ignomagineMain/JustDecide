"""
JustDecide test window.

    pip install -r requirements.txt
    python decide_ui.py

Uses JustDecide-q8_0.gguf from this folder (offers to download it from Hugging Face the first time).
Write the situation, a question and its options, press Decide: the decision, its confidence, every
option's probability and the time it took.
"""
from justdecide_ui import main

if __name__ == "__main__":
    main()
