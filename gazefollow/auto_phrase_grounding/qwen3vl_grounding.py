from transformers import AutoTokenizer, AutoModelForCausalLM
import torch, base64
from PIL import Image
import json, io

model_id = "Qwen/Qwen3-VL-8B-Instruct"
tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
)

def pil_to_b64(img):
    buf = io.BytesIO(); img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

img = Image.open("test.jpg")
img_b64 = pil_to_b64(img)

# Ask for structured boxes
messages = [
  {"role":"system","content":"You are a vision assistant. Output ONLY JSON."},
  {"role":"user","content":[
      {"type":"image_url","image_url":{"url":"data:image/png;base64,"+img_b64}},
      {"type":"text","text":
       "Detect all mugs. Return JSON: {\"objects\":[{\"label\":\"mug\",\"bbox\":[x1,y1,x2,y2],\"confidence\":float}]} \
        Coordinates must be absolute pixel ints, origin at top-left. No extra text."}
  ]}
]

inputs = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_tensors="pt").to(model.device)
out = model.generate(**inputs, max_new_tokens=300)
pred = tok.decode(out[0], skip_special_tokens=True)
boxes = json.loads(pred)  # boom
print(boxes)
