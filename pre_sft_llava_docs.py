import torch
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration

from torch.utils import tensorboard

from trl import SFTConfig,SFTTrainer,DataCollatorForCompletionOnlyLM

from datasets import load_dataset

# raw_datasets = load_dataset("HuggingFaceH4/llava-instruct-mix-vsft")
# train_dataset = raw_datasets["train"]
# eval_dataset = raw_datasets["test"]

# raw_datasets = load_dataset("HuggingFaceH4/llava-instruct-mix-vsft")
train_dataset = load_dataset("HuggingFaceH4/llava-instruct-mix-vsft", split="train[:15000]")
# eval_dataset = load_dataset("HuggingFaceH4/llava-instruct-mix-vsft", split="test[:1000]")
print('-=1=-*'*100)
processor = LlavaNextProcessor.from_pretrained('/root/model_playground/cache/qwen2_5vl_8B')
model = LlavaNextForConditionalGeneration.from_pretrained('/root/model_playground/cache/qwen2_5vl_8B', torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
# model.to("cuda:0")

def preprocess_dataset(data):
    text = processor.apply_chat_template(
        data['messages'], tokenize=False, add_generation_prompt=False
    )
    image = (data["images"][0])

    return {"text": text, "image": image}

train_dataset = train_dataset.map(preprocess_dataset, remove_columns=train_dataset.column_names)
# eval_dataset = eval_dataset.map(preprocess_dataset, remove_columns=eval_dataset.column_names)
train_dataset = train_dataset.shuffle()


_training_args = SFTConfig(
    output_dir=r"C:\Users\lenov\PycharmProjects\DRL\llm\cache\checkpoints\SFT",
    report_to="tensorboard",
    dataset_text_field="text",
    max_seq_length=512,
    per_device_train_batch_size=5,
    gradient_accumulation_steps=1,
    num_train_epochs=250,
    save_steps=1000,
    logging_steps=5,
    adam_beta1=0.85,
    adam_beta2=0.95,
)

_response_template = "<|im_start|>assistant\n"
_collator = DataCollatorForCompletionOnlyLM(_response_template, tokenizer=processor.tokenizer)


# data_collator = LLavaDataCollator(processor)

trainer = SFTTrainer(
    model=model,
    tokenizer=processor.tokenizer,
    # processor=processor,
    args=_training_args,
    train_dataset=train_dataset,
    # eval_dataset=eval_dataset,
    data_collator=_collator
)
print('-=1=-*'*100)

if __name__=='__main__':
    from datasets import load_dataset


    trainer.train()