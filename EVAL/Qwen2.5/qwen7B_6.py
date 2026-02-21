import pickle
import os

from modelscope import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

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


# model
model_path="./models/Qwen2.5-VL-7B-Instruct"

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path, torch_dtype="auto", device_map="auto"
        )
processor = AutoProcessor.from_pretrained(model_path)

# caption dict
captions_dict_path="../../Captions/Qwen3B_6.pkl"

with open(captions_dict_path, 'rb') as f:
    captions_dict = pickle.load(f)

if captions_dict == None:
    captions_dict={}

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

    # content
    content_parts = []

    # add imgs
    for image_path in paths_list:
        content_parts.append({
            "type": "image",
            "image": image_path
        })
    
    content_parts.append({
        "type": "text",
        "text": CAPTION_PROMPT
    })

    # messages
    messages = [
        {
            "role": "user",
            "content": content_parts
        }
    ]

    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=256)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    caption = output_text[0].strip()

    print(f"obj {len(captions_dict)+1}:")
    print(caption)

    captions_dict[obj_name]=caption

with open(captions_dict_path, 'wb') as F:  
    pickle.dump(captions_dict, F)  
    print("Captions saved")  
