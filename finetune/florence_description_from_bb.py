#%% Prepare Florence 2 model for each coordinate description
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image
import requests
import copy
import torch
model_id = 'microsoft/Florence-2-large'
model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, torch_dtype='auto').eval().cuda()
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)


def run_example(task_prompt, image, text_input=None):
    if text_input is None:
        prompt = task_prompt
    else:
        prompt = task_prompt + text_input
    inputs = processor(text=prompt, images=image, return_tensors="pt").to('cuda', torch.float16)
    generated_ids = model.generate(
      input_ids=inputs["input_ids"].cuda(),
      pixel_values=inputs["pixel_values"].cuda(),
      max_new_tokens=1024,
      early_stopping=False,
      do_sample=False,
      num_beams=3,
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = processor.post_process_generation(
        generated_text,
        task=task_prompt,
        image_size=(image.width, image.height)
    )

    return parsed_answer


from PIL import Image, ImageDraw, ImageFont
import random
import numpy as np

colormap = ['blue', 'orange', 'green', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan', 'red',
            'lime', 'indigo', 'violet', 'aqua', 'magenta', 'coral', 'gold', 'tan', 'skyblue']


def draw_polygons(image, prediction, fill_mask=False):
    """
    Draws segmentation masks with polygons on an image.

    Parameters:
    - image_path: Path to the image file.
    - prediction: Dictionary containing 'polygons' and 'labels' keys.
                  'polygons' is a list of lists, each containing vertices of a polygon.
                  'labels' is a list of labels corresponding to each polygon.
    - fill_mask: Boolean indicating whether to fill the polygons with color.
    """
    # Load the image

    draw = ImageDraw.Draw(image)

    # Set up scale factor if needed (use 1 if not scaling)
    scale = 1

    # Iterate over polygons and labels
    for polygons, label in zip(prediction['polygons'], prediction['labels']):
        color = random.choice(colormap)
        fill_color = random.choice(colormap) if fill_mask else None

        for _polygon in polygons:
            _polygon = np.array(_polygon).reshape(-1, 2)
            if len(_polygon) < 3:
                print('Invalid polygon:', _polygon)
                continue

            _polygon = (_polygon * scale).reshape(-1).tolist()

            # Draw the polygon
            if fill_mask:
                draw.polygon(_polygon, outline=color, fill=fill_color)
            else:
                draw.polygon(_polygon, outline=color)

                # Draw the label text
            draw.text((_polygon[0] + 8, _polygon[1] + 2), label, fill=color)
    return image


#%% Lets run on all images, and extract descriptions for every gaze target
task_prompt = '<REGION_TO_DESCRIPTION>'
# task_prompt = '<REGION_TO_SEGMENTATION>'
# task_prompt = '<REGION_TO_CATEGORY>'

# convert df columns to float
df['gaze_x'] = df['gaze_x'].astype(float)
df['gaze_y'] = df['gaze_y'].astype(float)

# first load the image
row = df.iloc[0]
img_path = base_data_dir_path / row['image_path']
# if not img_path.exists():
#     continue
img = Image.open(img_path).convert("RGB")

h = img.height
w = img.width
increase_margin = 0.2
gaze_target_x = int((1-increase_margin)*(row['gaze_x']) * 1000)
gaze_target_y = int((1-increase_margin)*(row['gaze_y']) * 1000)
gaze_target_x_end = int((1+increase_margin)*row['gaze_x'] * 1000)
gaze_target_y_end = int((1+increase_margin)*row['gaze_y'] * 1000)

results = run_example(task_prompt, img, text_input=f"<loc_{gaze_target_x}><loc_{gaze_target_y}>"
                                              f"<loc_{gaze_target_x_end}><loc_{gaze_target_y_end}>")
#%% draw the polygons on the image
img = draw_polygons(img, results['<REGION_TO_SEGMENTATION>'], fill_mask=True)
# save the image
img_save_path = base_data_dir_path / f"{Path(img_path).stem}_gaze_target.png"
img.save(img_save_path)
print(f"Saved image to {img_save_path}")
#%% Get the bounding box from polygon
poly = np.array(results['<REGION_TO_SEGMENTATION>']['polygons'][0][0]).reshape(-1, 2)
x_min = np.min(poly[:, 0])
x_max = np.max(poly[:, 0])
y_min = np.min(poly[:, 1])
y_max = np.max(poly[:, 1])
bbox = np.array([x_min, y_min, x_max, y_max])

# get description for the resulted bbox
task_prompt = '<REGION_TO_DESCRIPTION>'
results = run_example(task_prompt, img, text_input=f"<loc_{x_min}><loc_{y_min}>"
                                              f"<loc_{x_max}><loc_{y_max}>")
