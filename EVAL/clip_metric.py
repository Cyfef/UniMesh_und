import os
import torch
import pickle
import clip
from PIL import Image

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

model_captions_path=TODO

with open(model_captions_path, 'rb') as f:
    model_captions = pickle.load(f)     #dict

def cal_clipscore(objs_dir, model_local_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        model, preprocess = clip.load(model_local_path, device=device)
    except Exception as e:
        print(f"Error: {e}")
        return

    object_scores = []

    for obj_name in os.listdir(objs_dir):
        obj_path = os.path.join(objs_dir, obj_name)

        images=diffurank_select(obj_path)
        caption=model_captions[obj_name]


        with torch.no_grad():
            text_tokens = clip.tokenize([caption]).to(device)
            text_features = model.encode_text(text_tokens)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            img_similarities = []
            for img_p in images:
                image = preprocess(Image.open(img_p)).unsqueeze(0).to(device)
                image_features = model.encode_image(image)
                image_features /= image_features.norm(dim=-1, keepdim=True)

                # cos
                score = (text_features @ image_features.T).item()
                img_similarities.append(score)
            
            obj_avg = sum(img_similarities) / len(img_similarities)
            object_scores.append(obj_avg)
            print(f"Object: {len(object_scores)} | Score: {obj_avg:.4f}")

    if object_scores:
        final_mean = sum(object_scores) / len(object_scores)
        print(f"\n CLIPScore average: {final_mean:.4f}")

if __name__ == "__main__":
    MY_MODEL_PATH = "/你的路径/ViT-B-16.pt" 
    DATASET_PATH = "./my_3d_data"
    cal_clipscore(DATASET_PATH, MY_MODEL_PATH)