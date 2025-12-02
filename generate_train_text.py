import os
import random
import string
from PIL import Image, ImageDraw, ImageFont, ImageFilter

FONT_PATH = "Supercell-Magic Regular.ttf"
OUTPUT_DIR = "train_data"
COUNT = 2000

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def create_clash_text_image(text, index):
    # 1. Setup Canvas
    font_size = random.randint(28, 42) # Vary size slightly
    font = ImageFont.truetype(FONT_PATH, font_size)
    
    # Calculate size
    bbox = font.getbbox(text)
    text_w, text_h = bbox[2], bbox[3]
    w, h = text_w + 20, text_h + 20
    
    # 2. Create Background (Simulate game darkness/transparency)
    # Using a dark gray generic background helps model learn contrast
    bg_color = (random.randint(30, 60), random.randint(30, 60), random.randint(30, 60))
    img = Image.new('RGB', (w, h), color=bg_color)
    draw = ImageDraw.Draw(img)
    
    # 3. Draw Heavy Black Outline (Stroke)
    # Tesseract hates outlines unless trained on them.
    stroke_width = 3
    x, y = 10, 5
    
    # Draw stroke by drawing text multiple times in black
    draw.text((x, y), text, font=font, fill="black", stroke_width=stroke_width, stroke_fill="black")
    
    # 4. Draw White Inner Text
    draw.text((x, y), text, font=font, fill="white")
    
    # 5. Save in Tesseract format
    # Filename format: [lang].[fontname].exp[num].tif
    base_name = f"clash.Supercell.exp{index}"
    img.save(f"{OUTPUT_DIR}/{base_name}.tif")
    
    # 6. Create Box File (Ground Truth)
    # This is a simplified single-line box file format for Tesseract 5 LSTM training
    with open(f"{OUTPUT_DIR}/{base_name}.gt.txt", "w") as f:
        f.write(text)

# GENERATE LOOPS
print("Generating training data...")

# 1. Digits (Priority for Health/Elixir)
for i in range(1000):
    # Generate numbers like "2534", "10", "4.5"
    if random.random() > 0.5:
        txt = str(random.randint(0, 4000))
    else:
        txt = f"{random.randint(0, 9)}:{random.randint(10, 59)}" # Timer format
    create_clash_text_image(txt, i)

# 2. Words (For occasional game text)
words = ["Sudden", "Death", "Overtime", "x2", "Elixir"]
for i in range(1000, 1500):
    create_clash_text_image(random.choice(words), i)

# 3. Random Noise (To make it robust)
for i in range(1500, COUNT):
    txt = ''.join(random.choices(string.ascii_uppercase + string.digits, k=5))
    create_clash_text_image(txt, i)

print("Done. Data ready in /train_data")