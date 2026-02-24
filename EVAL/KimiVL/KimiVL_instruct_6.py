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

    imgs_path = [os.path.join(obj_path, f"{idx:05}.png") for idx in indices] 
    return imgs_path


if __name__ == '__main__':          # 保护主程序入口
    # 加载或初始化 caption 字典
    captions_dict_path = "../../Captions/Kimi_instruct_6.pkl"
    if os.path.exists(captions_dict_path):
        with open(captions_dict_path, 'rb') as f:
            captions_dict = pickle.load(f)
    else:
        captions_dict = {}

    # 加载模型和处理器（只需一次）
    model_path = "./models/moonshotai/Kimi-VL-A3B-Instruct"
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    # 提示词
    CAPTION_PROMPT = """
    You are given 6 renderings of a 3D object, please generate a concise caption that describes it. Captions should typically begin with an article ("a" or "an"), followed by color(s), shape, and the object type.Include distinctive features introduced by "with" when relevant (e.g., parts, textures, accessories). Use simple, everyday vocabulary and mention colors, materials (wooden, metal, plastic, etc.), and any notable details like wheels, windows, eyes, or decorations. Avoid long or complex sentences. The caption should be a short phrase or a simple sentence that captures the essential visual attributes.
    """

    # 物体根目录
    objs_dir = "/data/group/zhaolab/project/UniMesh/lab/UniMesh_und/glbs_4"

    for obj_name in os.listdir(objs_dir):
        if obj_name in captions_dict:
            continue

        obj_path = os.path.join(objs_dir, obj_name)
        paths_list = diffurank_select(obj_path)

        # 加载图片为 PIL Image 对象
        images = [Image.open(p) for p in paths_list]

        # 构造对话消息（注意：这里 "image" 字段只需占位符，实际图片通过 images 参数传入）
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},   # 仅用于占位，不需要路径
                    {"type": "image"},
                    {"type": "image"},
                    {"type": "image"},
                    {"type": "image"},
                    {"type": "image"},
                    {"type": "text", "text": CAPTION_PROMPT}
                ]
            }
        ]

        # 应用对话模板，得到包含特殊占位符的文本
        text = processor.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")

        # 将图片列表和文本一起传入 processor
        inputs = processor(
            images=images,               # 传入 PIL Image 列表
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

        caption = response.strip()
        print(f"obj {len(captions_dict)+1}: {caption}")
        captions_dict[obj_name] = caption

        # 每处理一个物体立即保存，防止中断丢失
        with open(captions_dict_path, 'wb') as f:
            pickle.dump(captions_dict, f)
            print("Captions saved")