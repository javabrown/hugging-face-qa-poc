## 🧠 Hugging Face QA POC (Offline Dockerized)

This project demonstrates a **self-contained FastAPI app** running two Hugging Face Transformer models **entirely offline** inside Docker.  
It supports both *Extractive* and *Abstractive* question-answering from any text input.

---

### 🚀 Features
- **Runs 100 % offline** — models are downloaded at build time.  
- **Two QA modes:**
  - 🟩 **Extractive:** finds and copies exact words from the context.  
  - 🟦 **Abstractive:** writes a natural-language answer in its own words.  
- **REST APIs + simple web UI** at `/ui`.  
- Built with **FastAPI**, **Transformers**, and **Docker**.

---

### ⚙️ Models Used
| Type | Model | Purpose |
|------|--------|----------|
| Extractive | `deepset/minilm-uncased-squad2` | Finds exact answers from text |
| Generative | `google/flan-t5-small` | Generates natural-language answers |

---

### 🧩 Build

```bash
# Default build
docker build -t hf-qa-poc:cpu .

# (Optional) use a different generative model
docker build   --build-arg GEN_MODEL_ID=google/flan-t5-base   -t hf-qa-poc:cpu .
```

---

### ▶️ Run

```bash
docker run --rm -p 9090:9090   -e HF_HUB_OFFLINE=1   -e TRANSFORMERS_OFFLINE=1   hf-qa-poc:cpu
```

Then open **http://localhost:9090/ui**

---

### 🔗 API Examples

#### 1️⃣ Extractive QA
```bash
curl -s -X POST http://localhost:9090/predict   -H 'Content-Type: application/json'   -d '{
    "context": "Shah Rukh Khan is an Indian actor and global icon.",
    "question": "Who is Shah Rukh Khan?"
  }' | jq
```

#### 2️⃣ Abstractive QA
```bash
curl -s -X POST http://localhost:9090/predict_abstractive   -H 'Content-Type: application/json'   -d '{
    "context": "Shah Rukh Khan is famous due to his acting prowess, charisma, and global brand endorsements.",
    "question": "Who is Shah Rukh Khan?",
    "max_new_tokens": 48
  }' | jq
```

---

### 📂 Folder Structure
```
.
├── Dockerfile
├── app.py
├── download_model.py
├── requirements.txt
└── run.sh   (optional helper script)
```

---

### 💡 Notes
- Both models are downloaded once during the Docker build; runtime is completely offline.  
- You can switch models anytime using `--build-arg`.  
- Works well on CPU with small models (`flan-t5-small` / `flan-t5-base`).  

---

**Purpose:** A lightweight demo for learning how to deploy Hugging Face Transformers with FastAPI inside Docker — supporting both extractive and generative QA.
