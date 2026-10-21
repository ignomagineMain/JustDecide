"""
Calling the JustDecide HTTP API (start it first:  python justdecide.py). Standard library only.

    python examples/call_api.py
"""
import json
import os
import urllib.request

URL = "http://127.0.0.1:8090/decide"


def decide(state, questions):
    body = json.dumps({"state": state, "questions": questions}).encode("utf-8")
    req = urllib.request.Request(URL, body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.load(r)


if __name__ == "__main__":
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "request.json"), encoding="utf-8") as f:
        request = json.load(f)
    print(json.dumps(decide(request["state"], request["questions"]), indent=2))
