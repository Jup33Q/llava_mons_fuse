
# packages for using clip demo
from transformers import CLIPProcessor, CLIPModel
import torch
import requests
from PIL import Image


#packages for establish LlaVANext models
from transformers import LlavaNextConfig,LlavaProcessor,LlavaNextConfig,LlavaNextForConditionalGeneration
from transformers import AutoConfig,AutoProcessor,AutoTokenizer
from transformers import CLIPVisionConfig,CLIPConfig,CLIPImageProcessor,CLIPVisionModel


'''
downloading clip (should using open_ai framework, clip-vit-base-patch32 is the default )
'''
# from modelscope import snapshot_download
# model=snapshot_download('thomas/clip-vit-base-patch32',cache_dir='/root/model_playground/cache')

'''
use of CLIP: seethomas/clip-vit-base-patch32
'''
# device = "cuda"
# torch_dtype = torch.float16

# model = CLIPModel.from_pretrained(
#     "/root/model_playground/cache/thomas/clip-vit-base-patch32"
# )
# processor = CLIPProcessor.from_pretrained("/root/model_playground/cache/thomas/clip-vit-base-patch32")

# url = "http://images.cocodataset.org/val2017/000000039769.jpg"
# image = Image.open(requests.get(url, stream=True).raw)

# inputs = processor(text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="pt", padding=True)
# # inputs.to(device)

# with torch.no_grad():
#     with torch.autocast(device):
#         outputs = model(**inputs)

# logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
# probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities
# print(probs) # tensor([[0.9946, 0.0052]], device='cuda:0', dtype=torch.float16)

'''
preperations:
all base model/file needed
an llm model (text to text)
a clip model
a chat template applied to your llm which supports image input (can using chat temp from qwen2_vl) 

loading config processor and tokeniser
//set qwen2.5:3b as an example llm
'''
vl_tokeniser=AutoTokenizer.from_pretrained('/root/model_playground/cache/Qwen/Qwen2-VL-2B-Instruct')
print('-'*100)
llm_config=AutoConfig.from_pretrained('/root/model_playground/cache/Qwen/Qwen2___5-7B-Instruct')
clip_config=CLIPVisionConfig.from_pretrained("openai/clip-vit-large-patch14",cache_dir='/root/model_playground/cache')
print('-'*100)

llm_tokeniser=AutoTokenizer.from_pretrained('/root/model_playground/cache/Qwen/Qwen2___5-7B-Instruct')
clip_processor=CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14",cache_dir='/root/model_playground/cache')
print('-'*100)

'''
get the llavaNext model config
'''
llava_config=LlavaNextConfig(text_config=llm_config,vision_config=clip_config)
print('-'*100)

'''
get the fused llavaNext model 
'''

model=LlavaNextForConditionalGeneration(llava_config)
print('-'*100)

processor=LlavaProcessor(clip_processor,llm_tokeniser)
processor.chat_template=vl_tokeniser.chat_template
print('-'*100)

'''
save model and processor
'''
model.save_pretrained('/root/model_playground/cache/qwen2_5vl_8B')
processor.save_pretrained('/root/model_playground/cache/qwen2_5vl_8B')
print('-'*100)

'''

'''

if __name__=='__main__':
    model = LlavaNextForConditionalGeneration.from_pretrained('/root/model_playground/cache/qwen2_5vl_8B')
    # processor = AutoProcessor.from_pretrained("/root/model_playground/cache/qwen2_5vl")

    prompt = "USER: <image>\nWhat's the content of the image? ASSISTANT:"
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)

    inputs = processor(images=image, text=prompt, return_tensors="pt")

    # # Generate
    # generate_ids = model.generate(**inputs, max_new_tokens=15)
    # processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    print(inputs)