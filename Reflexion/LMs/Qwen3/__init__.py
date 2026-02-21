from transformers import AutoModelForImageTextToText, AutoProcessor

class Qwen3Predictor:
    def __init__(self, model_path="./models/Qwen/Qwen3-VL-8B-Instruct"):
        print(f"Loading model from {model_path}...")
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path, torch_dtype="auto", device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.device = self.model.device
        self.model.eval()

    def predict(self, messages, max_new_tokens=128):
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
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