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
    model_name: str = field(default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    block_size: int = field(default=32768)
    wandb_project: str = field(default="s1")
    wandb_entity: str = field(default="chrisvnz-bimwerx")
    train_file_path: str = field(default="simplescaling/s1K-1.1_tokenized")
    dagger: bool = field(default=False)
    output_dir: str = field(default="s1-1")

    def __post_init__(self):
        os.environ['WANDB_PROJECT'] = self.wandb_project
        os.environ['WANDB_ENTITY'] = self.wandb_entity


def train():
    # Parse script arguments
    parser = argparse.ArgumentParser(description="Train LLM with fine-tuning")
    
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

    # Load model
    model = transformers.AutoModelForCausalLM.from_pretrained(config.model_name)

    # Load dataset
    dataset = load_dataset(config.train_file_path)

    # Load tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(config.model_name, use_fast=True)

    # Define tokenization strategy based on model type
    if "Llama" in config.model_name:
        instruction_template = "<|start_header_id|>user<|end_header_id|>"
        response_template = "<|start_header_id|>assistant<|end_header_id|>\n\n"
        tokenizer.pad_token = "<|reserved_special_token_5|>"
    elif "Qwen" in config.model_name:
        instruction_template = "<|im_start|>user"
        response_template = "<|im_start|>assistant\n"
        tokenizer.pad_token = "<|fim_pad|>"

    # Define data collator
    collator = trl.DataCollatorForCompletionOnlyLM(
        instruction_template=instruction_template,
        response_template=response_template,
        tokenizer=tokenizer,
        mlm=False
    )

    # Setup training arguments
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

    # Define trainer
    trainer = trl.SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"] if "test" in dataset else dataset["train"],
        args=training_args,
        data_collator=collator
    )

    # Start training
    trainer.train()

    # Save the model and tokenizer
    trainer.save_model(output_dir=config.output_dir)
    tokenizer.save_pretrained(config.output_dir)
    trainer.accelerator.wait_for_everyone()


if __name__ == "__main__":
    train()
