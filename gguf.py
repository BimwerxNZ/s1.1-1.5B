import os
import subprocess
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path

# ✅ Step 1: Define Paths
model_path = "./s1-1"  # Your fine-tuned model
merged_model_path = "./GGUF"  # Where merged model will be saved
output_gguf_path = "./s1-1.gguf"  # Final GGUF file

Path(merged_model_path).mkdir(parents=True, exist_ok=True)

# ✅ Step 2: Load Fine-Tuned Model Directly
print("🔄 Loading fine-tuned model...")
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# ✅ Step 3: Save Model to Merged Directory (if needed)
print("💾 Saving model to merged directory...")
model.save_pretrained(merged_model_path)
tokenizer.save_pretrained(merged_model_path)

# ✅ Step 4: Locate `convert_hf_to_gguf.py` Script
gguf_script = os.path.join("llama.cpp", "convert_hf_to_gguf.py")

if not os.path.exists(gguf_script):
    print("❌ ERROR: `convert_hf_to_gguf.py` not found! Download it manually.")
    exit(1)

# ✅ Step 5: Convert Model to GGUF
quantization_format = "f16"  # Use "f16" for an unquantized model

print(f"🚀 Converting to GGUF format using {quantization_format}...")
try:
    subprocess.run([
        "python", gguf_script,
        merged_model_path,  # positional argument: directory of the fine-tuned model
        "--outfile", output_gguf_path,
        "--outtype", quantization_format
    ], check=True)
except subprocess.CalledProcessError as e:
    print(f"❌ ERROR: GGUF conversion failed: {e}")

print(f"✅ GGUF Model saved at: {output_gguf_path}")
