from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration,GenerationConfig
import torch
from PIL import Image
import requests
from datasets import load_dataset

processor = LlavaNextProcessor.from_pretrained(r'C:\Users\lenov\PycharmProjects\DRL\llm\cache\qwen2_5vl_8B')
# print(processor.chat_template)

model = LlavaNextForConditionalGeneration.from_pretrained(r'C:\Users\lenov\PycharmProjects\DRL\llm\cache\qwen2_5vl_8B', torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
model.to("cuda:0")


# prepare image and text prompt, using the appropriate prompt template
# url = "http://images.cocodataset.org/val2017/000000039769.jpg"
# image = Image.open(requests.get(url, stream=True).raw)
_dataset=load_dataset(path=r"C:\Users\lenov\.cache\modelscope\hub\datasets\coco_2014_caption\coco_2014_caption",split='train[:20]')
example=_dataset[2]
image=example['image']

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
]
prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
print(prompt)
inputs = processor(image, prompt, return_tensors="pt").to("cuda:0")

# print(inputs['attention_mask'],inputs['input_ids'])

# # autoregressively complete prompt
output = model.generate(inputs['input_ids'],
                        # attention_mask=inputs['attention_mask'],
                        max_new_tokens=100)
# print(output[0])
print(processor.decode(output[0], skip_special_tokens=False))