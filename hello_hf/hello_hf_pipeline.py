# hello_hf_pipeline.py
from transformers import pipeline

# 1️⃣ Load a ready-made pipeline (downloads model if not cached)
qa_pipe = pipeline("question-answering", model="deepset/minilm-uncased-squad2")

# 2️⃣ Provide context and question
context = (
    "Donald John Trump is an American politician, media personality, and businessman who is the 47th president of the United States. A member of the Republican Party, he served as the 45th president from 2017 to 2021.. "
    "He is also known as the 'King of Bollywood'."
)
question = "Who is Trump?"

# 3️⃣ Run the model
result = qa_pipe(question=question, context=context)

# 4️⃣ Print the raw result
print(result)

# 5️⃣ Pretty output
print(f"\nAnswer: {result['answer']}")
print(f"Score : {result['score']:.4f}")
