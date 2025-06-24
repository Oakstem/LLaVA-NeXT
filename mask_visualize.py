import cv2
import numpy as np
from pathlib import Path
from docs.extract_attention import fix_wsl_paths
# lets visualize a binary mask by applying it to an image
img_path = Path(r"D:\Projects\data\gazefollow\train\00000000\00000002.jpg")
mask_path = Path(r"D:\Projects\data\gazefollow\train_gaze_segmentations\masks\gaze__00000002_masks.npy")

img_path = fix_wsl_paths(str(img_path))
mask_path = fix_wsl_paths(str(mask_path))
#%%
img = cv2.imread(str(img_path))
mask = np.load(str(mask_path), allow_pickle=True).item()['masks'][0]

# Ensure mask is binary
mask = (mask > 0).astype(np.uint8) * 255
# Create a black image
result_img = np.zeros_like(img)
# Copy original colors where mask is present
result_img[mask == 255] = img[mask == 255]
# Save the result image
output_path = Path("masked_image.jpg")
cv2.imwrite(str(output_path), result_img)
print(f"Masked image saved to {output_path}")
