import sys

import torch
from transformers import pipeline

model = sys.argv[1]  # "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

pipe = pipeline(
    "text-generation", model=model, torch_dtype=torch.bfloat16, device_map="auto"
)

# We use the tokenizer's chat template to format each message - see https://huggingface.co/docs/transformers/main/en/chat_templating
messages = [
    {
        "role": "system",
        "content": "You are a helpful chatbot who pretends to be Richard Feynman",
    },
    {"role": "user", "content": "Could you tell me about the challenger disaster ?"},
]
prompt = pipe.tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
outputs = pipe(
    prompt, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95
)
print(outputs[0]["generated_text"])
