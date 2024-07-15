import os
import json
import random
from PIL import Image, ImageDraw

# Directory structure
data_dir = 'data'
os.makedirs(data_dir, exist_ok=True)
image_dir = os.path.join(data_dir, 'images')
os.makedirs(image_dir, exist_ok=True)

# Parameters
num_images = 2000
image_size = (64, 64)
questions = ["What shape is this?", "What color is this?"]

# Create synthetic images and questions
dataset = []

for i in range(num_images):
    # Create an image with a random shape and color
    img = Image.new('RGB', image_size, color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    shape = random.choice(['circle', 'square'])
    color = random.choice(['red', 'green', 'blue'])
    
    if shape == 'circle':
        draw.ellipse((16, 16, 48, 48), fill=color)
    else:
        draw.rectangle((16, 16, 48, 48), fill=color)
    
    img_path = os.path.join(image_dir, f'image_{i}.png')
    img.save(img_path)
    
    # Generate questions and answers
    for question in questions:
        answer = shape if "shape" in question else color
        dataset.append({
            'image': img_path,
            'question': question,
            'answer': answer
        })

# Save dataset to JSON file
with open(os.path.join(data_dir, 'dataset.json'), 'w') as f:
    json.dump(dataset, f)
