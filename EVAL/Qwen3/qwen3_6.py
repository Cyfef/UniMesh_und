import pickle
import os
from transformers import AutoModelForImageTextToText, AutoProcessor

# caption dict
captions_dict_path="../../Captions/Qwen3B_6.pkl"

with open(captions_dict_path, 'rb') as f:
    captions_dict = pickle.load(f)

if captions_dict == None:
    captions_dict={}

# model
model = AutoModelForImageTextToText.from_pretrained(
    "./models/Qwen/Qwen3-VL-8B-Instruct", dtype="auto", device_map="auto"
)

processor = AutoProcessor.from_pretrained("./models/Qwen/Qwen3-VL-8B-Instruct")

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

    paths_list = []
    for i in range(27):
        img_path=os.path.join(obj_path,f"{i:05}.png")
        paths_list.append(img_path)

    # Messages containing multiple images and a text query
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": f"file://{paths_list[0]}"},
                {"type": "image", "image": f"file://{paths_list[1]}"},
                {"type": "image", "image": f"file://{paths_list[2]}"},
                {"type": "image", "image": f"file://{paths_list[3]}"},
                {"type": "image", "image": f"file://{paths_list[4]}"},
                {"type": "image", "image": f"file://{paths_list[5]}"},
                {"type": "text", "text": CAPTION_PROMPT},
            ],
        }
    ]

    # Preparation for inference
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    )
    inputs = inputs.to(model.device)

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    print(output_text)

    caption = output_text

    print(f"obj {len(captions_dict)+1}:")
    print(caption)

    captions_dict[obj_name]=caption

with open(captions_dict_path, 'wb') as F:  
    pickle.dump(captions_dict, F)  
    print("Captions saved")  
