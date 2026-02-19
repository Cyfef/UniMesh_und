import requests
import torch
import os
import io
import pickle

from PIL import Image
import soundfile as sf
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
from urllib.request import urlopen


# caption dict
captions_dict_path="../../Captions/Phi4_6.pkl"

with open(captions_dict_path, 'rb') as f:
    captions_dict = pickle.load(f)

if captions_dict == None:
    captions_dict={}

# Define model path
model_path = "./models/microsoft/Phi-4-multimodal-instruct"

# Load model and processor
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path, 
    device_map="cuda", 
    torch_dtype="auto", 
    trust_remote_code=True,
    # if you do not use Ampere or later GPUs, change attention to "eager"
    _attn_implementation='flash_attention_2',
).cuda()

# Load generation config
generation_config = GenerationConfig.from_pretrained(model_path)

# 6 imgs run
CAPTION_PROMPT="""
You are given 6 renderings of a 3D object, please generate a concise caption that describes it. Captions should typically begin with an article ("a" or "an"), followed by color(s), shape, and the object type.Include distinctive features introduced by "with" when relevant (e.g., parts, textures, accessories). Use simple, everyday vocabulary and mention colors, materials (wooden, metal, plastic, etc.), and any notable details like wheels, windows, eyes, or decorations. Avoid long or complex sentences. The caption should be a short phrase or a simple sentence that captures the essential visual attributes.
"""

# Define prompt structure
user_prompt = '<|user|>'
assistant_prompt = '<|assistant|>'
prompt_suffix = '<|end|>'

# objs
objs_dir=TODO

for obj_name in os.listdir(objs_dir):
    
    if obj_name in list(captions_dict.keys()):
        continue

    obj_path=os.path.join(objs_dir,obj_name)

    paths_list = []
    for i in range(27):
        img_path=os.path.join(obj_path,f"{i:05}.png")
        paths_list.append(img_path)

    # Image Processing
    prompt = f'{user_prompt}<|image_1|><|image_2|><|image_3|><|image_4|><|image_5|><|image_6|>{CAPTION_PROMPT}{prompt_suffix}{assistant_prompt}'

    # open image
    images = [Image.open(p) for p in paths_list]          # 加载为 PIL Image 列表
    inputs = processor(text=prompt, images=images, return_tensors='pt').to('cuda:0')

    # Generate response
    generate_ids = model.generate(
        **inputs,
        max_new_tokens=1000,
        generation_config=generation_config,
    )
    generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
    response = processor.batch_decode(
        generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    print(response)

    caption = response

    print(f"obj {len(captions_dict)+1}:")
    print(caption)

    captions_dict[obj_name]=caption

with open(captions_dict_path, 'wb') as F:  
    pickle.dump(captions_dict, F)  
    print("Captions saved")  

