import sys
sys.modules['tensorflow'] = None
sys.modules['tensorflow_text'] = None
sys.modules['tf'] = None

from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import torch
import uuid
import time

print("Loading base model...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-70B-Instruct")
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-70B-Instruct",
    quantization_config=bnb_config,
    device_map="auto",
)

print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(model, "/workspace/adversa_red_lora")
model.eval()

print("Model ready!")

app = Flask(__name__)

@app.route("/v1/chat/completions", methods=["POST"])
def chat():
    data = request.json
    messages = data.get("messages", [])
    max_tokens = data.get("max_tokens", 512)
    temperature = data.get("temperature", 0.7)

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=max(temperature, 0.01),
            do_sample=True,
            top_p=0.9,
        )

    response_text = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    return jsonify({
        "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": "adversa-red",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": response_text},
            "finish_reason": "stop"
        }],
        "usage": {"prompt_tokens": len(inputs["input_ids"][0]), "completion_tokens": len(out[0]) - len(inputs["input_ids"][0]), "total_tokens": len(out[0])}
    })

@app.route("/v1/models", methods=["GET"])
def models():
    return jsonify({"data": [{"id": "adversa-red", "object": "model"}]})

app.run(host="0.0.0.0", port=8000)
