import os
import logging
import traceback
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from transformers import pipeline

# ----------------- Config -----------------
TASK = os.environ.get("TASK", "question-answering")
MODEL_ID = os.environ.get("MODEL_ID", "deepset/minilm-uncased-squad2")

# Abstractive (text-generation) model
GEN_MODEL_ID = os.environ.get("GEN_MODEL_ID", "google/flan-t5-small")

MAX_SEQ_LEN = int(os.environ.get("MAX_SEQ_LEN", 384))
DOC_STRIDE = int(os.environ.get("DOC_STRIDE", 128))
ANSWER_THRESHOLD = float(os.environ.get("ANSWER_THRESHOLD", 0.20))
RETURN_N_BEST = int(os.environ.get("RETURN_N_BEST", 3))

# Offline-safe defaults
os.environ.setdefault("HF_HOME", "/models")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

# ----------------- Logging -----------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger("hf-qa")

# ----------------- App -----------------
app = FastAPI(title="HF QA Service", version="0.3.0")

class QAIn(BaseModel):
    context: str
    question: str

class QABatchIn(BaseModel):
    items: List[QAIn]

class AbstractiveIn(BaseModel):
    context: str
    question: str
    max_new_tokens: Optional[int] = 64
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9

# Pipelines (lazy-loaded)
pipe = None
load_error = None

gen_pipe = None
gen_load_error = None

def load_pipeline_once():
    """Load extractive QA pipeline once."""
    global pipe, load_error
    if pipe is not None or load_error is not None:
        return
    try:
        log.info("Loading extractive pipeline: task=%s model=%s", TASK, MODEL_ID)
        p = pipeline(task=TASK, model=MODEL_ID, tokenizer=MODEL_ID, device=-1)
        # tiny warmup
        if TASK == "question-answering":
            _ = p({"question": "ping?", "context": "pong."})
        else:
            _ = p("ping")
        pipe = p
        log.info("Extractive pipeline loaded OK.")
    except Exception as e:
        load_error = traceback.format_exc()
        log.error("Extractive pipeline load FAILED: %s", e)
        for line in load_error.splitlines():
            log.error(line)

def load_gen_pipeline_once():
    """Load abstractive (text2text-generation) pipeline once."""
    global gen_pipe, gen_load_error
    if gen_pipe is not None or gen_load_error is not None:
        return
    try:
        log.info("Loading generative pipeline: text2text-generation model=%s", GEN_MODEL_ID)
        gp = pipeline(task="text2text-generation", model=GEN_MODEL_ID, tokenizer=GEN_MODEL_ID, device=-1)
        # warmup
        _ = gp("Question: ping? Context: pong. Answer:")
        gen_pipe = gp
        log.info("Generative pipeline loaded OK.")
    except Exception as e:
        gen_load_error = traceback.format_exc()
        log.error("Generative pipeline load FAILED: %s", e)
        for line in gen_load_error.splitlines():
            log.error(line)

@app.on_event("startup")
def _startup():
    banner = f"""
============================================================
HuggingFace Q/A Service (Transformers pipelines)
Extractive:
  Task : {TASK}
  Model: {MODEL_ID}

Abstractive:
  Task : text2text-generation
  Model: {GEN_MODEL_ID}

Port : {os.environ.get("UVICORN_PORT", "9090")}

Endpoints:
  GET  /healthz
  POST /predict               -> {{"context": "...", "question": "..."}}
  POST /predict/batch         -> {{"items":[{{"context":"...","question":"..."}}, ...]}}
  POST /predict_abstractive   -> {{"context": "...", "question": "...", "max_new_tokens": 64}}

UI:
  GET  /ui

Try:
  curl -s http://localhost:{os.environ.get("UVICORN_PORT", "9090")}/healthz | jq
============================================================
"""
    print(banner, flush=True)
    # Preload both so the first user call is fast (optional; remove if you prefer lazy)
    load_pipeline_once()
    load_gen_pipeline_once()

@app.get("/healthz")
def healthz():
    status = "ready" if pipe is not None else ("gen-only" if gen_pipe is not None else "loading")
    resp = {
        "status": status,
        "extractive": {
            "task": TASK,
            "model_id": MODEL_ID,
            "loaded": pipe is not None,
            "error": (load_error.splitlines()[-1] if load_error else None),
        },
        "abstractive": {
            "task": "text2text-generation",
            "model_id": GEN_MODEL_ID,
            "loaded": gen_pipe is not None,
            "error": (gen_load_error.splitlines()[-1] if gen_load_error else None),
        },
        "settings": {
            "max_seq_len": MAX_SEQ_LEN,
            "doc_stride": DOC_STRIDE,
            "answer_threshold": ANSWER_THRESHOLD,
            "return_n_best": RETURN_N_BEST,
        },
        "offline": {
            "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE"),
            "TRANSFORMERS_OFFLINE": os.environ.get("TRANSFORMERS_OFFLINE"),
        },
    }
    return resp

@app.post("/predict")
def predict(payload: QAIn):
    if not payload.context or not payload.question:
        raise HTTPException(status_code=400, detail="Both 'context' and 'question' are required.")
    load_pipeline_once()
    if pipe is None:
        raise HTTPException(status_code=503, detail="Extractive model not loaded. See /healthz for load_error.")

    res = pipe(
        {"question": payload.question, "context": payload.context},
        top_k=RETURN_N_BEST,
        max_seq_len=MAX_SEQ_LEN,
        doc_stride=DOC_STRIDE,
        handle_impossible_answer=True,
    )
    best = res[0] if isinstance(res, list) else res
    ans = best.get("answer", "") or ""
    score = float(best.get("score", 0.0))
    return {
        "task": TASK,
        "model_id": MODEL_ID,
        "answer": ans if score >= ANSWER_THRESHOLD else "",
        "score": score,
        "start": int(best.get("start", -1)),
        "end": int(best.get("end", -1)),
        "no_answer": (ans.strip() == "" or score < ANSWER_THRESHOLD),
    }

@app.post("/predict/batch")
def predict_batch(payload: QABatchIn):
    load_pipeline_once()
    if pipe is None:
        raise HTTPException(status_code=503, detail="Extractive model not loaded. See /healthz for load_error.")
    queries = [{"question": it.question, "context": it.context} for it in (payload.items or [])]
    if not queries:
        return {"results": []}
    results = pipe(
        queries,
        top_k=RETURN_N_BEST,
        max_seq_len=MAX_SEQ_LEN,
        doc_stride=DOC_STRIDE,
        handle_impossible_answer=True,
    )
    normalized = []
    for r in results:
        best = r[0] if isinstance(r, list) else r
        ans = best.get("answer", "") or ""
        score = float(best.get("score", 0.0))
        normalized.append({
            "answer": ans if score >= ANSWER_THRESHOLD else "",
            "score": score,
            "start": int(best.get("start", -1)),
            "end": int(best.get("end", -1)),
            "no_answer": (ans.strip() == "" or score < ANSWER_THRESHOLD),
        })
    return {"task": TASK, "model_id": MODEL_ID, "results": normalized}

@app.post("/predict_abstractive")
def predict_abstractive(payload: AbstractiveIn):
    if not payload.context or not payload.question:
        raise HTTPException(status_code=400, detail="Both 'context' and 'question' are required.")
    load_gen_pipeline_once()
    if gen_pipe is None:
        raise HTTPException(status_code=503, detail="Generative model not loaded. See /healthz for load_error.")

    # Prompt pattern suited for FLAN-T5
    prompt = f"Question: {payload.question}\nContext: {payload.context}\nAnswer:"
    gen = gen_pipe(
        prompt,
        max_new_tokens=payload.max_new_tokens or 64,
        temperature=payload.temperature or 0.7,
        top_p=payload.top_p or 0.9,
        do_sample=True,
    )
    answer = gen[0].get("generated_text", "").strip()
    return {
        "task": "text2text-generation",
        "model_id": GEN_MODEL_ID,
        "answer": answer,
        "params": {
            "max_new_tokens": payload.max_new_tokens or 64,
            "temperature": payload.temperature or 0.7,
            "top_p": payload.top_p or 0.9,
        },
    }

@app.get("/ui", response_class=HTMLResponse)
def ui():
    return """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Hugging Face Q/A Service – UI</title>
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <style>
    :root { --bg:#0f172a; --card:#111827; --muted:#334155; --fg:#e5e7eb; --accent:#22d3ee; }
    *{box-sizing:border-box} body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.45 system-ui,-apple-system,Segoe UI,Roboto,Ubuntu,Cantarell}
    .wrap{max-width:980px;margin:40px auto;padding:0 16px}
    .card{background:var(--card);border:1px solid #1f2937;border-radius:16px;padding:20px;box-shadow:0 5px 25px rgba(0,0,0,.25)}
    h1{margin:0 0 8px;font-size:24px} .muted{color:#94a3b8;font-size:14px}
    label{display:block;margin:16px 0 8px;font-weight:600}
    textarea,input{width:100%;padding:12px 14px;border-radius:12px;border:1px solid #334155;background:#0b1220;color:var(--fg);font:15px}
    textarea{min-height:160px;resize:vertical}
    .row{display:flex;gap:12px;align-items:center;flex-wrap:wrap}
    button{background:var(--accent);border:0;color:#002b31;padding:10px 16px;border-radius:12px;font-weight:700;cursor:pointer}
    button:disabled{opacity:.6;cursor:not-allowed}
    pre{background:#0b1220;border:1px solid #1f2937;border-radius:12px;padding:14px;overflow:auto;max-height:360px}
    .tabs{display:flex;gap:8px;margin-top:10px}
    .tab{background:#0b1220;border:1px solid #1f2937;border-radius:8px;padding:6px 10px;cursor:pointer}
    .tab.active{outline:2px solid var(--accent)}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="card">
      <h1>Hugging Face QA Service</h1>
      <div class="muted">Try either Extractive (<code>/predict</code>) or Abstractive (<code>/predict_abstractive</code>)</div>

      <div class="tabs">
        <div class="tab active" id="tab-ex">Extractive</div>
        <div class="tab" id="tab-ab">Abstractive</div>
      </div>

      <label for="context">Context</label>
      <textarea id="context">Shah Rukh Khan is famous due to a combination of his acting prowess, charisma, and a compelling rags-to-riches success story, amplified by his interviews and global brand endorsements.</textarea>

      <label for="question">Question</label>
      <input id="question" value="Who is Shah Rukh Khan?"/>

      <div class="row" style="margin-top:12px">
        <button id="runBtn">Ask</button>
        <span class="muted" id="hint"></span>
      </div>

      <label style="margin-top:18px">Result</label>
      <pre id="out">{}</pre>

      <div class="muted" style="margin-top:10px">Endpoints: <code>/healthz</code>, <code>/predict</code>, <code>/predict_abstractive</code>, <code>/predict/batch</code></div>
    </div>
  </div>

  <script>
    const $ = s => document.querySelector(s);
    let mode = 'extractive';
    const hint = $("#hint");
    const tabEx = $("#tab-ex"), tabAb = $("#tab-ab");
    const btn = $("#runBtn"), out = $("#out");

    function setMode(m){
      mode = m;
      tabEx.classList.toggle('active', m==='extractive');
      tabAb.classList.toggle('active', m==='abstractive');
      hint.textContent = m==='extractive'
        ? "POST /predict (copies exact span from context)"
        : "POST /predict_abstractive (generates a natural-language answer)";
    }
    setMode('extractive');

    tabEx.onclick = () => setMode('extractive');
    tabAb.onclick = () => setMode('abstractive');

    async function run(){
      btn.disabled = true;
      out.textContent = "Running...";
      const context = $("#context").value;
      const question = $("#question").value;
      const endpoint = mode === 'extractive' ? '/predict' : '/predict_abstractive';
      const body = mode === 'extractive'
        ? {context, question}
        : {context, question, max_new_tokens: 48};

      try{
        const res = await fetch(endpoint, {
          method:'POST',
          headers:{'Content-Type':'application/json'},
          body: JSON.stringify(body)
        });
        const json = await res.json();
        out.textContent = JSON.stringify(json, null, 2);
      }catch(e){
        out.textContent = 'Error: ' + e;
      }finally{
        btn.disabled = false;
      }
    }
    btn.onclick = run;
  </script>
</body>
</html>
    """
