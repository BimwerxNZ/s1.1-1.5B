from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
# Load fine-tuned model
model_path = "./s1-1"
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)

def generate_response(prompt, system_prompt="", max_length=2048, temperature=0.1):
    # Build the prompt according to the Qwen2.5 template:
    # 1. System block (guiding instructions)
    # 2. User block (the user's prompt)
    # 3. Assistant block (model's response starts here; we open a <think> block for the model to generate into)
    formatted_prompt = (
        f"<|im_start|>system\n"
        f"{system_prompt}\n"
        f"<|im_end|>\n"
        f"<|im_start|>user\n"
        f"{prompt}\n"
        f"<|im_end|>\n"
        f"<|im_start|>assistant\n"
        f"<think>\n"  # Leave this block open so that the model can generate the closing </think> and answer.
    )
    
    # Tokenize the prompt and move it to GPU (if available)
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to("cuda")
    
    # Generate the output from the model
    output = model.generate(
        **inputs,
        max_length=max_length,
        temperature=temperature,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True
    )
    
    # Decode the generated tokens to a string (preserving special tokens)
    full_response = tokenizer.decode(output[0], skip_special_tokens=False)
    
    # Only keep text before the end marker (<|im_end|>)
    # response = full_response.split("<|im_end|>")[0]
    
    return full_response


# Test the generation
user_input = "What is the third derivative of -25*o - 11*o**2 - 33*o + 58*o + 23*o**5 wrt o?"
# 1380*o**2
print("Bot:", generate_response(user_input))