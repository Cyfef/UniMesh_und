from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

class Qwen2_5Predictor:
    def __init__(self, model_path="./models/Qwen/Qwen2.5-VL-7B-Instruct"):
        print(f"Loading model from {model_path}...")
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path, torch_dtype="auto", device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.device = self.model.device

    def predict(self, messages, max_new_tokens=128):
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        inputs = inputs.to(self.device)

        generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )
        
        return output_text[0] 