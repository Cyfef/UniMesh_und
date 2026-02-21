import pickle
import os

from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

def diffurank_select(obj_path):
    '''
    select 6 imgs based on diffurank scores
    '''
    diffu_path = os.path.join(obj_path, "diffurank_scores.pkl")
    with open(diffu_path, 'rb') as f:
        diffu_scores = pickle.load(f)     # numpy array

    indexed = list(enumerate(diffu_scores))
    indexed.sort(key=lambda x: x[1])
        
    lowest_six = indexed[:6]
    indices = [idx for idx, val in lowest_six][::-1] 

    imgs_path=[os.path.join(obj_path,f"{idx:05}.png") for idx in indices] 
        
    return imgs_path

# caption dict
captions_dict_path="../../Captions/Kimi_instruct_6.pkl"

with open(captions_dict_path, 'rb') as f:
    captions_dict = pickle.load(f)

if captions_dict == None:
    captions_dict={}

# model
model_path = "./models/moonshotai/Kimi-VL-A3B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype="auto",
    device_map="auto",
    trust_remote_code=True,
)
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

# 6 imgs run
CAPTION_PROMPT="""
You are given 6 renderings of a 3D object, please generate a concise caption that describes it. Captions should typically begin with an article ("a" or "an"), followed by color(s), shape, and the object type.Include distinctive features introduced by "with" when relevant (e.g., parts, textures, accessories). Use simple, everyday vocabulary and mention colors, materials (wooden, metal, plastic, etc.), and any notable details like wheels, windows, eyes, or decorations. Avoid long or complex sentences. The caption should be a short phrase or a simple sentence that captures the essential visual attributes.
"""

# objs
objs_dir=TODO

for obj_name in os.listdir(objs_dir):
    
    if obj_name in list(captions_dict.keys()):
        continue

    obj_path=os.path.join(objs_dir,obj_name)
    
    """
    paths_list = []
    for i in range(27):
        img_path=os.path.join(obj_path,f"{i:05}.png")
        paths_list.append(img_path)
    """
    
    paths_list=diffurank_select(obj_path)

    images = [Image.open(p) for p in paths_list]          # 加载为 PIL Image 列表

    # 构造多图对话消息：content 中包含多个 image 类型元素
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": paths_list[0]},  
                {"type": "image", "image": paths_list[1]},  
                {"type": "image", "image": paths_list[2]},  
                {"type": "image", "image": paths_list[3]},  
                {"type": "image", "image": paths_list[4]},  
                {"type": "image", "image": paths_list[5]},   
                {"type": "text", "text": CAPTION_PROMPT}
            ]
        }
    ]

    # 应用对话模板，生成包含图像占位符的文本
    text = processor.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")

    # 将图片列表和文本一起传给 processor
    inputs = processor(
        images=images,           # 这里传入图片列表
        text=text,
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(model.device)

    # 生成回答
    generated_ids = model.generate(**inputs, max_new_tokens=512)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    response = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    print(response)

    caption = response

    print(f"obj {len(captions_dict)+1}:")
    print(caption)

    captions_dict[obj_name]=caption

with open(captions_dict_path, 'wb') as F:  
    pickle.dump(captions_dict, F)  
    print("Captions saved")  

