import streamlit as st

st.markdown(

'''
# LlaVA/LlaVA_Next Architecture

## Abstract

this document is about using native transfromers funtionals to apply llava architecture 
(with CLIPS by openai) on ready llms like llama and qwen (here we use qwen2.5:7b as an example).

\n as a hint, see huggingface docs for detials on llava and llava_next config functionals (links as bellow):
\nllava:
\nhttps://huggingface.co/docs/transformers/main/en/model_doc/llava#transformers.LlavaConfig
\nllava_next:
\nhttps://huggingface.co/docs/transformers/main/en/model_doc/llava_next#transformers.LlavaNextConfig
''')


st.markdown(
'''
the example datasets we used here can be downloaded via the following python commands

```python
# sample1
from modelscope.msdatasets import MsDataset
ds = MsDataset.load("coco_2014_caption", namespace="modelscope", split="train")

# sample2
from datasets import load_dataset
raw_datasets = load_dataset("HuggingFaceH4/llava-instruct-mix-vsft")


```

'''
)

Step1,Step2,Step3=st.tabs(['Fusing Models','Inference','SFT'])

Step1.markdown(
'''
## Get the models fused into llava structure

### 1.preprations

for preperations, some models should be downloaded and set to be ready-to-use

here are the downloading examples for Clip_models
```python

from transformers import CLIPModel,CLIPProcessor,AutoModel 
#downloading clip model (should supports CLIPModel and ClipVision config)
model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14-336",cache_dir='/root/model_playground/cache')
# as an alternative 'thomas/clip-vit-base-patch32' from modelscope is recommended as the same base model defaults to CLIPConfig()
from modelscope import snapshot_download
model=snapshot_download('thomas/clip-vit-base-patch32',cache_dir='/root/model_playground/cache')


```
in addition, we need qwen2.5:7b and a chat_template from qwen2vl

```python
from modelscope import snapshot_download
model=snapshot_download('Qwen/Qwen2.5-7B-Instruct',cache_dir='/root/model_playground/cache') # you can use 14B_version or higher for better result
from modelscope import snapshot_download
model=snapshot_download('Qwen/Qwen2-VL-2B-Instruct',cache_dir='/root/model_playground/cache')

```
test if your clip works 
```python

device = "cuda"
torch_dtype = torch.float16

model = CLIPModel.from_pretrained(
    "paths to your clip model"
)
processor = CLIPProcessor.from_pretrained("paths to your clip model")

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

inputs = processor(text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="pt", padding=True)
inputs.to(device)

with torch.no_grad():
    with torch.autocast(device):
        outputs = model(**inputs)

logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities
print(probs) # tensor([[0.9946, 0.0052]], device='cuda:0', dtype=torch.float16)

```
Remark: please run the following code in another file

import all packages:

```python
from transformers import CLIPProcessor, CLIPModel
import torch
import requests
from PIL import Image


#packages for establish LlaVANext models
from transformers import LlavaNextConfig,LlavaProcessor,LlavaNextConfig,LlavaNextForConditionalGeneration
from transformers import AutoConfig,AutoProcessor,AutoTokenizer
from transformers import CLIPVisionConfig,CLIPConfig,CLIPImageProcessor,CLIPVisionModel


```
''')

Step1.markdown(
''' 
### 2. get parts ready
#### preperations:
an llm model (text to text)
\n
a clip model
\n
a chat template applied to your llm which supports image input
\n

loading config processor and tokeniser

```python
#this tokeniser from qwen2_vl contains a chat_template attribute applied to qwen2.5
vl_tokeniser=AutoTokenizer.from_pretrained('/root/model_playground/cache/Qwen/Qwen2-VL-2B-Instruct')

llm_config=AutoConfig.from_pretrained('/root/model_playground/cache/Qwen/Qwen2___5-7B-Instruct')
clip_config=CLIPVisionConfig.from_pretrained("openai/clip-vit-large-patch14",cache_dir='/root/model_playground/cache')

llm_tokeniser=AutoTokenizer.from_pretrained('/root/model_playground/cache/Qwen/Qwen2___5-7B-Instruct')
clip_processor=CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14",cache_dir='/root/model_playground/cache')
```


'''
)

Step1.markdown(
'''
### 3.Magics

```python
# 1. get the fused llavaNext model
llava_config=LlavaNextConfig(text_config=llm_config,vision_config=clip_config)

# 2. fuse your clip processor, tokenizer, and 'steal' a chat template
processor=LlavaProcessor(clip_processor,llm_tokeniser)
processor.chat_template=vl_tokeniser.chat_template

# 3.save model and processor, name it anyway you long
model.save_pretrained('/root/model_playground/cache/qwen2_5vl_8B')
processor.save_pretrained('/root/model_playground/cache/qwen2_5vl_8B')
```


'''
)


Step2.markdown(
'''
## the inference 
#### code to give you a clue how your llava-mons are trained

import packages

```python
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration,GenerationConfig
import torch
from PIL import Image
import requests
from datasets import load_dataset

```
### loading models
```python
processor = LlavaNextProcessor.from_pretrained('paths to your models')
# print(processor.chat_template)

model = LlavaNextForConditionalGeneration.from_pretrained(r'paths to your models', torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
model.to("cuda:0")

```
### get a data 
either
```python
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

```
or
```python
_dataset=load_dataset(path='path to coco_2014_caption',split='train[:20]')
example=_dataset[2]
image=example['image']
```

### apply template
```python
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
# make sure you have taken a chat template for a multimodel for your processor
prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
# print(prompt)
inputs = processor(image, prompt, return_tensors="pt").to("cuda:0")


```
### get output
```python
output = model.generate(inputs['input_ids'],
                    # attention_mask=inputs['attention_mask'],
                    max_new_tokens=100)
print(processor.decode(output[0], skip_special_tokens=False))
```

'''
)


Step3.markdown(
'''
you may find the model inference immediately after  model fusing is almostly unusable, so a fine-tuing is suggested
## SFT training
Remark: since the images are not necessarily a part of predictants, SFT/(lora-SFT) is the better candidate for training 

'''
)

Step3.markdown(
'''
import packages and models
```python
import torch
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from torch.utils import tensorboard
from trl import SFTConfig,SFTTrainer,DataCollatorForCompletionOnlyLM
from datasets import load_dataset

processor = LlavaNextProcessor.from_pretrained('/root/model_playground/cache/qwen2_5vl_8B')
model = LlavaNextForConditionalGeneration.from_pretrained('/root/model_playground/cache/qwen2_5vl_8B', torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
```
different dataset are used to different processing functions

'''
)

s1,s2=Step3.tabs(['sample1','sample2'])
s1.code(
'''
# raw_datasets = load_dataset("HuggingFaceH4/llava-instruct-mix-vsft")
train_dataset = load_dataset("HuggingFaceH4/llava-instruct-mix-vsft", split="train[:15000]")
# eval_dataset = load_dataset("HuggingFaceH4/llava-instruct-mix-vsft", split="test[:1000]")    

def preprocess_dataset(data):
    text = processor.apply_chat_template(
        data['messages'], tokenize=False, add_generation_prompt=False
    )
    image = (data["images"][0])

    return {"text": text, "image": image}

train_dataset = train_dataset.map(preprocess_dataset, remove_columns=train_dataset.column_names)
# eval_dataset = eval_dataset.map(preprocess_dataset, remove_columns=eval_dataset.column_names)
train_dataset = train_dataset.shuffle()


'''
)
s2.code(
'''
ds=load_dataset("path to coco_2014_caption",split="train[:500]")
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
train_dataset = train_dataset.shuffle()

'''
)

Step3.markdown(
'''
#### set train configs
''')
Step3.code(
r'''
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
'''
)

Step3.markdown(
'''
#### get trainer
'''
)

Step3.code(
r'''
trainer = SFTTrainer(
    model=model,
    tokenizer=processor.tokenizer,
    # processor=processor,
    args=_training_args,
    train_dataset=train_dataset,
    # eval_dataset=eval_dataset,
    data_collator=_collator
)
'''
)
Step3.markdown(
'''
#### train
'''
)
Step3.code(
r'''
if __name__=='__main__':
    from datasets import load_dataset


    trainer.train()
'''
)
