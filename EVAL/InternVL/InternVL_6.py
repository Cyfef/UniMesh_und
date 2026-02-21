import pickle
import os

from lmdeploy import pipeline, PytorchEngineConfig
from lmdeploy.vl import load_image
from lmdeploy.vl.constants import IMAGE_TOKEN

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
model_path="./models/OpenGVLab/InternVL3_5-4B"

# caption dict
captions_dict_path="../../Captions/InternVL_6.pkl"

with open(captions_dict_path, 'rb') as f:
    captions_dict = pickle.load(f)

if captions_dict == None:
    captions_dict={}

# 6 imgs run
CAPTION_PROMPT="""
You are given 6 renderings of a 3D object, please generate a concise caption that describes it. Captions should typically begin with an article ("a" or "an"), followed by color(s), shape, and the object type.Include distinctive features introduced by "with" when relevant (e.g., parts, textures, accessories). Use simple, everyday vocabulary and mention colors, materials (wooden, metal, plastic, etc.), and any notable details like wheels, windows, eyes, or decorations. Avoid long or complex sentences. The caption should be a short phrase or a simple sentence that captures the essential visual attributes.
"""

# image_tokens
image_tokens = []
for i in range(6):
    image_tokens.append(f"Image-{i+1}: {IMAGE_TOKEN}")

prompt=f"{chr(10).join(image_tokens)}"+CAPTION_PROMPT

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
    
    images = [load_image(path) for path in paths_list]

    pipe = pipeline(model_path, backend_config=PytorchEngineConfig(session_len=32768, tp=1))
        
    response = pipe((prompt, images))

    caption = response.text.strip()

    print(f"obj {len(captions_dict)+1}:")
    print(caption)

    captions_dict[obj_name]=caption

with open(captions_dict_path, 'wb') as F:  
    pickle.dump(captions_dict, F)  
    print("Captions saved")  
