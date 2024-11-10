class LLavaDataCollator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, examples):
        texts = []
        images = []
        for example in examples:
            messages = example["messages"]
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            texts.append(text)
            images.append(example["images"][0])

        batch = self.processor(texts, images, return_tensors="pt", padding=True)

        labels = batch["input_ids"].clone()
        if self.processor.tokenizer.pad_token_id is not None:
            labels[labels == self.processor.tokenizer.pad_token_id] = -100
        batch["labels"] = labels

        return batch
    
if __name__=='__main__':
    from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration,GenerationConfig
    import torch
    from datasets import load_dataset

    raw_datasets = load_dataset("HuggingFaceH4/llava-instruct-mix-vsft")
    train_dataset = raw_datasets["train"]

    processor = LlavaNextProcessor.from_pretrained('/root/model_playground/cache/qwen2_5vl_8B')
    data_collator = LLavaDataCollator(processor)

    print(data_collator(train_dataset))