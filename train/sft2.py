import os
import argparse
import torch
from dataclasses import dataclass, field, asdict
import warnings
import logging
from datasets import load_dataset
import transformers
import trl

warnings.filterwarnings("ignore", category=FutureWarning)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Function to get GPU count
def get_gpu_count():
    return torch.cuda.device_count()

@dataclass
class TrainingConfig:
    model_name: str = field(default="FlofloB/100k_fineweb_continued_pretraining_Qwen2.5-0.5B-Instruct_Unsloth_merged_16bit")
    block_size: int = field(default=32768)
    wandb_project: str = field(default="s1.1b-0.5B")
    wandb_entity: str = field(default="chrisvnz-bimwerx")
    train_file_path: str = field(default="simplescaling/s1K-1.1_tokenized")
    dagger: bool = field(default=False)
    output_dir: str = field(default="s1.1b-0.5B")
    
    def __post_init__(self):
        os.environ['WANDB_PROJECT'] = self.wandb_project
        os.environ['WANDB_ENTITY'] = self.wandb_entity

def format_example(example):
    """
    Given a dataset example with columns: 'question', 'deepseek_thinking_trajectory', and 'solution',
    this function returns a formatted string with the desired structure.
    """
    system_prompt = (
        "<|im_start|>system\n"
        "You are a helpful assistant that explains your reasoning step by step.\n"
        "<|im_end|>\n"
    )
    user_prompt = (
        "<|im_start|>user\n"
        f"Question: {example['question']}\n"
        "<|im_end|>\n"
    )
    assistant_prompt = (
        "<|im_start|>assistant\n"
        "<think>\n"
        f"{example['deepseek_thinking_trajectory']}\n"
        "</think>\n"
        f"Solution: {example['solution']}\n"
        "<|im_end|>\n"
    )
    return system_prompt + user_prompt + assistant_prompt

def preprocess_dataset(dataset):
    """
    Applies the format_example function to each example in the dataset.
    """
    # If the dataset is a DatasetDict, we work on the 'train' split.
    # Modify the split names as needed.
    def _preprocess(example):
        example["new_text"] = format_example(example)
        return example

    # Apply on train (and eval/test if needed)
    dataset = dataset.map(_preprocess)
    return dataset

def train():
    # Parse script arguments
    parser = argparse.ArgumentParser(description="Train LLM with fine-tuning using chain-of-thought examples")
    
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--min_lr", type=float, default=0)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=1)  # Matches micro_batch_size
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--push_to_hub", type=bool, default=False)

    args = parser.parse_args()

    # Get the number of GPUs
    gpu_count = get_gpu_count()
    logging.info(f"Detected {gpu_count} GPUs.")

    # Load configurations
    config = TrainingConfig()
    log_config = {**asdict(config), **vars(args)}
    logging.info(f"Training config: {log_config}")

    # Set max_seq_length in args (will be picked up by SFTTrainer)
    args.max_seq_length = config.block_size

    # Load model
    model = transformers.AutoModelForCausalLM.from_pretrained(config.model_name)

    # Load dataset and preprocess it
    dataset = load_dataset(config.train_file_path)
    dataset = preprocess_dataset(dataset["train"])  # Adjust if using different splits

    # Load tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(config.model_name, use_fast=True)
    tokenizer.padding_side = "right"  # Fix for BF16 training

    # For Qwen models, set special tokens and response template that now includes the chain-of-thought markers.
    if "Qwen" in config.model_name:
        # Note: the response_template here is not used directly if you are formatting the text in the dataset.
        # We include it to keep consistency.
        instruction_template = "<|im_start|>user"
        response_template = "<|im_start|>assistant\n<think>\n"  # The training examples already include this.
        tokenizer.pad_token = "<|fim_pad|>"
    else:
        # Fallback defaults (if needed)
        instruction_template = "<|im_start|>user"
        response_template = "<|im_start|>assistant\n"
    
    # Define a simple data collator that just returns the "text" field as the model input.
    # Here we assume that each example now has a "text" field produced by format_example.
    class SimpleTextCollator:
        def __call__(self, examples):
            texts = [ex["text"] for ex in examples]
            return tokenizer(texts, truncation=True, max_length=config.block_size, padding="max_length", return_tensors="pt")

    collator = SimpleTextCollator()

    # Setup training arguments (removed max_seq_length from here)
    training_args = transformers.TrainingArguments(
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=0.05,
        bf16=True,
        evaluation_strategy="no",
        logging_steps=1,
        save_strategy="no",
        lr_scheduler_type="cosine",
        output_dir=config.output_dir,
        push_to_hub=args.push_to_hub,
        save_only_model=True,
        gradient_checkpointing=True
    )

    # Define trainer, explicitly passing the tokenizer.
    # Note: We use the preprocessed dataset's "text" field as training data.
    trainer = trl.SFTTrainer(
        model=model,
        train_dataset=dataset,  # Each example now has the "text" field with our complete prompt.
        eval_dataset=dataset if "test" not in dataset else dataset["test"],
        args=training_args,
        data_collator=collator,
        tokenizer=tokenizer,
        max_seq_length=config.block_size
    )

    # Start training
    trainer.train()

    # Save the model and tokenizer
    trainer.save_model(output_dir=config.output_dir)
    tokenizer.save_pretrained(config.output_dir)
    trainer.accelerator.wait_for_everyone()

if __name__ == "__main__":
    train()
