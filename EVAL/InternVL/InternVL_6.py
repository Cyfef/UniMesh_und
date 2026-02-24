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

if __name__ == '__main__':   # 必须加这行保护
    # 模型路径
    model_path = "./models/OpenGVLab/InternVL3_5-4B"
    
    # 初始化 pipeline（只做一次！）
    pipe = pipeline(
        model_path,
        backend_config=PytorchEngineConfig(session_len=32768, tp=1),
        fix_mistral_regex=True
    )
    
    # 加载或创建 captions 字典
    captions_dict_path = "../../Captions/InternVL_6.pkl"
    if os.path.exists(captions_dict_path):
        with open(captions_dict_path, 'rb') as f:
            captions_dict = pickle.load(f)
    else:
        captions_dict = {}
    
    # 图像提示模板（保持不变）
    CAPTION_PROMPT="""
    You are given 6 renderings of a 3D object, please generate a concise caption that describes it. Captions should typically begin with an article ("a" or "an"), followed by color(s), shape, and the object type.Include distinctive features introduced by "with" when relevant (e.g., parts, textures, accessories). Use simple, everyday vocabulary and mention colors, materials (wooden, metal, plastic, etc.), and any notable details like wheels, windows, eyes, or decorations. Avoid long or complex sentences. The caption should be a short phrase or a simple sentence that captures the essential visual attributes.
    """
    image_tokens = [f"Image-{i+1}: {IMAGE_TOKEN}" for i in range(6)]
    prompt = f"{chr(10).join(image_tokens)}\n{CAPTION_PROMPT}"
    
    # 物体目录
    objs_dir = TODO
    
    for obj_name in os.listdir(objs_dir):
        if obj_name in captions_dict:
            continue
        
        obj_path = os.path.join(objs_dir, obj_name)
        img_paths = diffurank_select(obj_path)   # 获取6张图片路径
        
        # 加载图片
        images = [load_image(p) for p in img_paths]
        
        # 推理（使用已加载的 pipe）
        response = pipe((prompt, images))
        caption = response.text.strip()
        
        print(f"obj {len(captions_dict)+1}: {caption}")
        captions_dict[obj_name] = caption
        
        # 可选：每处理一个物体保存一次（防止中途中断）
        with open(captions_dict_path, 'wb') as f:
            pickle.dump(captions_dict, f)
    
    print("所有物体处理完成，最终 captions 已保存。")
