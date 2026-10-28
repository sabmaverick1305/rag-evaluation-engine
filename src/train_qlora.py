import torch
from datasets import load_dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTConfig, SFTTrainer


def train_qlora():
    model_id = "Qwen/Qwen2.5-1.5B-Instruct"
    dataset_path = "data/sft_dataset.jsonl"
    use_cuda = torch.cuda.is_available()

    print("Loading tokenizer and dataset...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    dataset = load_dataset("json", data_files=dataset_path, split="train")

    # 1. 4-bit QLoRA Configuration (NF4 format)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    print(f"Loading base model ({model_id}) with 4-bit quantization...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto"
    )

    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)

    # 2. LoRA Configuration (Targeting attention matrices)
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    # 3. Training Arguments
    training_args = SFTConfig(
        output_dir="./qwen-graphrag-adapter",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=10,
        num_train_epochs=3,
        max_steps=100,
        save_strategy="epoch",
        fp16=False,
        bf16=True,
        # bitsandbytes' 8-bit optimizer has no MPS implementation.
        optim="paged_adamw_8bit" if use_cuda else "adamw_torch",
        dataloader_pin_memory=use_cuda,
        max_length=1024,
    )

    # 4. Initialize SFTTrainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        # TRL recognizes conversational datasets through the messages column
        # and applies the tokenizer's chat template automatically.
        processing_class=tokenizer,
        args=training_args,
    )

    print("Starting QLoRA fine-tuning...")
    trainer.train()

    print("Saving fine-tuned LoRA adapter...")
    trainer.model.save_pretrained("./qwen-graphrag-adapter-final")
    print("Fine-tuning complete!")

if __name__ == "__main__":
    train_qlora()
