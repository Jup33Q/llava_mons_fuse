import torch
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration

from torch.utils import tensorboard

from trl import SFTConfig,SFTTrainer,DataCollatorForCompletionOnlyLM

from datasets import load_dataset

ds=load_dataset(r"C:\Users\lenov\.cache\modelscope\hub\datasets\coco_2014_caption",split="train[:500]")

print('-=1=-*'*100)
model_path=r'C:\Users\lenov\PycharmProjects\DRL\llm\cache\qwen2_5vl_8B'
processor = LlavaNextProcessor.from_pretrained(model_path)
model = LlavaNextForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
# model.to("cuda:0")

def preprocess_dataset(data):
    conversation = [
        {
            "role": "system",
            "content": [
                # {"type": "image"},
                {"type": "text", "text": "please recognise the information in the picture"},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "What is shown in this image?"},
            ],
        },
        {
            "role": "assistant",
            "content": [
                # {"type": "image"},
                {"type": "text", "text": f"{data['caption']}"},
            ],
        }
    ]
    text = processor.apply_chat_template(
        conversation, tokenize=False, add_generation_prompt=False
    )
    # print(text)
    image = data["image"]

    return {"text": text, "image": image}
train_dataset = ds.map(preprocess_dataset, remove_columns=ds.column_names)
# eval_dataset = eval_dataset.map(preprocess_dataset, remove_columns=eval_dataset.column_names)
train_dataset = train_dataset.shuffle()


_training_args = SFTConfig(
    output_dir=r"C:\Users\lenov\PycharmProjects\DRL\llm\cache\checkpoints\SFT",
    report_to="tensorboard",
    dataset_text_field="text",
    max_seq_length=512,
    per_device_train_batch_size=3,
    gradient_accumulation_steps=1,
    num_train_epochs=250,
    save_steps=150,
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