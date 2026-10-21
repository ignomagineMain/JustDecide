"""
JustDecide in a few lines.   From the repository folder:  python examples/quickstart.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from justdecide import default_files, download_model, open_engine  # noqa: E402

model = default_files()[0] or download_model()          # downloads once (about 640 MB)
engine = open_engine(model)

state = {"battery_pct": 14, "order_queue": 2, "position": "aisle 3", "lidar": "clear"}
questions = {
    "next": {"type": "choice", "instructions": "What should the AGV do next?",
             "criteria": {"pick next order": None, "go charge": "drive to the charger now", "go to parking": None}},
    "battery": {"type": "score", "instructions": "How much battery is left?",
                "criteria": ["very low", "low", "ok", "full"]},
    "go_charge": {"type": "yes/no"},
}
result = engine.decide(state, questions)

for name, answer in result["answers"].items():
    print(f"{name:10s} {answer}")
print(f"{result['ms']} ms for {len(questions)} questions")

# Act only when the model is sure enough
step = result["answers"]["next"]
p = step["probabilities"][step["choice"]]
print(f"\n-> {step['choice']}" if p >= 0.8 else f"\n-> not sure ({step['choice']} {p:.0%}): ask a person")
engine.close()
