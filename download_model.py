import os, json, sys, traceback
from transformers import pipeline

TASK = os.environ.get("TASK", "question-answering")
MODEL_ID = os.environ.get("MODEL_ID", "deepset/minilm-uncased-squad2")

# Use HF_HOME only (TRANSFORMERS_CACHE is deprecated warning). Keeping both is fine.
os.environ.setdefault("HF_HOME", "/models")
os.environ.setdefault("TRANSFORMERS_CACHE", "/models")

print(f"[download_model] START task={TASK} model={MODEL_ID} HF_HOME={os.environ.get('HF_HOME')}")
try:
    if TASK == "question-answering":
        nlp = pipeline(task=TASK, model=MODEL_ID)
        _ = nlp({
            "question": "Where is Lawrence Township located?",
            "context": "Lawrence Township is in Mercer County, New Jersey."
        })
    else:
        nlp = pipeline(task=TASK, model=MODEL_ID)
        _ = nlp("Hello world")

    meta = {"task": TASK, "model_id": MODEL_ID}
    with open("/models/MODEL_META.json", "w") as f:
        json.dump(meta, f, indent=2)

    print("[download_model] SUCCESS – model cached in /models")
except Exception as e:
    print("[download_model] FATAL – failed to cache model:", e, file=sys.stderr)
    traceback.print_exc()
    sys.exit(10)
