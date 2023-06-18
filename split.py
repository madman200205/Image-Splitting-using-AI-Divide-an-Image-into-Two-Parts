import cv2
import numpy as np
import torch
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn

# Load the pre-trained Mask R-CNN model
model = maskrcnn_resnet50_fpn(pretrained=False)
model.load_state_dict(torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True).state_dict())
model.eval()

# Load the screenshot image
image_path = 'Resources/repi.jpg'
image = cv2.imread(image_path)

# Convert the image to RGB format
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Convert the image to a tensor
image_tensor = torch.from_numpy(image_rgb / 255.0).permute(2, 0, 1).unsqueeze(0).float()

# Run the image through the model
with torch.no_grad():
    prediction = model(image_tensor)

# Retrieve the segmented regions from the prediction
masks = prediction[0]['masks']
masks = masks.ge(0.5).squeeze().cpu().numpy()

# Split the image based on the segmented regions
split_images = []
for i, mask in enumerate(masks):
    segmented_region = cv2.bitwise_or(image, image, mask=mask.astype(np.uint8))
    split_images.append(segmented_region)

# Split the image into two parts
height, width = image.shape[:2]
half_width = width // 2

split_images = []
split_images.append(image[:, :half_width, :])  # Left half
split_images.append(image[:, half_width:, :])  # Right half

# Display the split images
for i, split_image in enumerate(split_images):
    cv2.imshow(f'Split Image {i+1}', split_image)

# Wait for a key press and then close the windows
cv2.waitKey(0)
cv2.destroyAllWindows()
