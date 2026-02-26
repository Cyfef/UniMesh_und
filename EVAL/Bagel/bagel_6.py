import os
import torch
import random
import pickle
import sys
import numpy as np

from PIL import Image
from accelerate import infer_auto_device_map, load_checkpoint_and_dispatch, init_empty_weights

from data.transforms import ImageTransform
from data.data_utils import add_special_tokens
from modeling.bagel import (
    BagelConfig, Bagel, Qwen2Config, Qwen2ForCausalLM, SiglipVisionConfig, SiglipVisionModel
)
from modeling.qwen2 import Qwen2Tokenizer
from modeling.autoencoder import load_ae
from inferencer import InterleaveInferencer

if not hasattr(np, '_core'):
    # 如果环境是 1.x，但数据需要 2.0 的结构
    from numpy.core import multiarray
    # 模拟新版的模块结构
    class MockCore:
        pass
    _mock_core = MockCore()
    _mock_core._reconstruct = multiarray._reconstruct
    _mock_core.multiarray = multiarray
    
    # 注入到全局模块，让 pickle 能找到 numpy._core._reconstruct
    sys.modules['numpy._core'] = _mock_core
    sys.modules['numpy._core.multiarray'] = multiarray

#if not hasattr(np, '_core'):
 #   sys.modules['numpy._core'] = np.core

def diffurank_select(obj_path):
    '''
    select 6 imgs based on diffurank scores
    '''
    diffu_path = os.path.join(obj_path, "diffurank_scores.pkl")
    with open(diffu_path, 'rb') as f:
        diffu_scores = pickle.load(f)     # numpy array
    #diffu_scores = np.load(diffu_path, allow_pickle=True)
    

    indexed = list(enumerate(diffu_scores))
    indexed.sort(key=lambda x: x[1])
        
    lowest_six = indexed[:6]
    indices = [idx for idx, val in lowest_six][::-1] 

    imgs_path=[os.path.join(obj_path,f"{idx:05}.png") for idx in indices] 
        
    return imgs_path

#-----------------------Model Initialization-----------------------#

model_path = "./models/ByteDance-Seed/BAGEL-7B-MoT"  # Download from https://huggingface.co/ByteDance-Seed/BAGEL-7B-MoT

# LLM config preparing
llm_config = Qwen2Config.from_json_file(os.path.join(model_path, "llm_config.json"))
llm_config.qk_norm = True
llm_config.tie_word_embeddings = False
llm_config.layer_module = "Qwen2MoTDecoderLayer"

# ViT config preparing
vit_config = SiglipVisionConfig.from_json_file(os.path.join(model_path, "vit_config.json"))
vit_config.rope = False
vit_config.num_hidden_layers = vit_config.num_hidden_layers - 1

# VAE loading
vae_model, vae_config = load_ae(local_path=os.path.join(model_path, "ae.safetensors"))

# Bagel config preparing
config = BagelConfig(
    visual_gen=True,
    visual_und=True,
    llm_config=llm_config, 
    vit_config=vit_config,
    vae_config=vae_config,
    vit_max_num_patch_per_side=70,
    connector_act='gelu_pytorch_tanh',
    latent_patch_size=2,
    max_latent_size=64,
)

with init_empty_weights():
    language_model = Qwen2ForCausalLM(llm_config)
    vit_model      = SiglipVisionModel(vit_config)
    model          = Bagel(language_model, vit_model, config)
    model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config, meta=True)

# Tokenizer Preparing
tokenizer = Qwen2Tokenizer.from_pretrained(model_path)
tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)

# Image Transform Preparing
vae_transform = ImageTransform(1024, 512, 16)
vit_transform = ImageTransform(980, 224, 14)

#-----------------------Model Loading and Multi GPU Infernece Preparing-----------------------#

max_mem_per_gpu = "80GiB"  # Modify it according to your GPU setting. On an A100, 80 GiB is sufficient to load on a single GPU.

device_map = infer_auto_device_map(
    model,
    max_memory={i: max_mem_per_gpu for i in range(torch.cuda.device_count())},
    no_split_module_classes=["Bagel", "Qwen2MoTDecoderLayer"],
)
print(device_map)

same_device_modules = [
    'language_model.model.embed_tokens',
    'time_embedder',
    'latent_pos_embed',
    'vae2llm',
    'llm2vae',
    'connector',
    'vit_pos_embed'
]

if torch.cuda.device_count() == 1:
    first_device = device_map.get(same_device_modules[0], "cuda:0")
    for k in same_device_modules:
        if k in device_map:
            device_map[k] = first_device
        else:
            device_map[k] = "cuda:0"
else:
    first_device = device_map.get(same_device_modules[0])
    for k in same_device_modules:
        if k in device_map:
            device_map[k] = first_device

# Thanks @onion-liu: https://github.com/ByteDance-Seed/Bagel/pull/8
model = load_checkpoint_and_dispatch(
    model,
    checkpoint=os.path.join(model_path, "ema.safetensors"),
    device_map=device_map,
    offload_buffers=True,
    dtype=torch.bfloat16,
    force_hooks=True,
    offload_folder="/tmp/offload"
)

model = model.eval()
print('Model loaded')


#-----------------------Inferencer Preparing-----------------------#
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


inference=InterleaveInferencer(
    model=model, 
    vae_model=vae_model, 
    tokenizer=tokenizer, 
    vae_transform=vae_transform, 
    vit_transform=vit_transform, 
    new_token_ids=new_token_ids
)

#-----------------------Understanding-----------------------#

inference_hyper=dict(
    max_think_token_n=1000,
    do_sample=False,
    # text_temperature=0.3,
)

# caption dict
import pickle

captions_dict_path="../../Captions/Bagel_6.pkl"

# 6 imgs run
#CAPTION_PROMPT="""
#You are given 6 renderings of a 3D object, please generate a concise caption that describes it. Captions should typically begin with an article ("a" or "an"), followed by color(s), shape, and the object type.Include distinctive features introduced by "with" when relevant (e.g., parts, textures, accessories). Use simple, everyday vocabulary and mention colors, materials (wooden, metal, plastic, etc.), and any notable details like wheels, windows, eyes, or decorations. Avoid long or complex sentences. The caption should be a short phrase or a simple sentence that captures the essential visual attributes.
#"""

CAPTION_PROMPT="""
You are given 6 rendered images of the same 3D object from different viewpoints.

Please write one short English caption that best describes the 3D object shown in these images.

Follow these rules:
1. Write **only one simple sentence**.
2. Begin with “a” or “an”.
3. Mention the **main object category** (e.g., airplane, car, person, animal, chair, etc.).
4. Include **dominant colors** and optionally **materials or simple attributes** (e.g., standing, flying, sitting).
5. Be concise and factual — do not use complex phrases or subjective language.
"""

prompt="<img><|image_1|></img><img><|image_2|></img><img><|image_3|></img><img><|image_4|></img><img><|image_5|></img><img><|image_6|></img>"+CAPTION_PROMPT

objs_dir="/data/group/zhaolab/project/UniMesh/lab/UniMesh_und/data/cap"

for obj_name in os.listdir(objs_dir):
    with open(captions_dict_path, 'rb') as f:
        captions_dict = pickle.load(f)

    if captions_dict == None:
        captions_dict={}

    if obj_name in list(captions_dict.keys()):
        continue

    obj_path=os.path.join(objs_dir,obj_name)

    """
    imgs_list=[]
    for i in range(27):
        img_path=os.path.join(obj_path,f"{i:05}.png")
        imgs_list.append(Image.open(img_path))
    """

    imgs_list=diffurank_select(obj_path)

    input_list=imgs_list+[prompt]

    output_list=inference.interleave_inference(input_lists=input_list,
                                   understanding_output=True,
                                   max_think_token_n=1000,
                                   do_sample=False,
                                   # text_temperature=0.3,
                                   )
    print(f"obj {len(captions_dict)+1}:")
    print(output_list[0])

    captions_dict[obj_name]=output_list[0]

    with open(captions_dict_path, 'wb') as F:  
        pickle.dump(captions_dict, F)
