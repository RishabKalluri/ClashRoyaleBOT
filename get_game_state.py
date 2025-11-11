import cv2
import re
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np

processor = TrOCRProcessor.from_pretrained('microsoft/trocr-small-printed')
model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-small-printed')
IMAGE_PATH = "IMG_4297.PNG" # IMG_4297, IMG_4376

# Format: (x1, y1, x2, y2)
TIMER_REGION = (990, 220, 1170, 290) 
TIMER_REGION_ALT = (990, 180, 1170, 250) # Alternate timer region for different layouts
ELIXIR_REGION = (320, 2400, 380, 2470)
ELIXIR_BAR_REGION = (300, 2380, 1170, 2510)

LEFT_ALLY_TOWER_REGION = (190, 1540, 340, 1590)
LEFT_ALLY_TOWER_REGION_ALT = (195, 1575, 345, 1625) # Alternate left tower region
RIGHT_ALLY_TOWER_REGION = (885, 1540, 1035, 1590)
RIGHT_ALLY_TOWER_REGION_ALT = (880, 1575, 1030, 1625) # Alternate right tower region

LEFT_ENEMY_TOWER_REGION = (190, 365, 340, 425)
LEFT_ENEMY_TOWER_REGION_ALT = (195, 480, 345, 540) # Alternate left enemy tower region
RIGHT_ENEMY_TOWER_REGION = (885, 365, 1035, 425)
RIGHT_ENEMY_TOWER_REGION_ALT = (880, 480, 1030, 540) # Alternate right enemy tower region

KING_TOWER_REGION = (0, 0, 0, 0)
KING_TOWER_REGION_ALT = (0, 0, 0, 0) # Alternate king tower region
ENEMY_KING_TOWER_REGION = (0, 0, 0, 0)
ENEMY_KING_TOWER_REGION_ALT = (515, 210, 715, 290) # Alternate enemy king tower region

def plot_cropped_image(image):
    """Plot the cropped image region. Accepts a PIL.Image or a BGR numpy array."""
    if isinstance(image, Image.Image):
        arr = np.array(image)
        bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    else:
        bgr = image
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    ax.imshow(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    ax.set_title("Image Region")
    ax.axis("off")
    plt.tight_layout()
    plt.show()

def _ensure_ocr_available():
    if "processor" not in globals() or "model" not in globals():
        raise RuntimeError("OCR processor/model not initialized. Uncomment or initialize them at the top of the file.")

def _ocr_pil_image(pil_img):
    _ensure_ocr_available()
    pixel_values = processor(pil_img, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values)
    return processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

def _crop_to_pil(img, region):
    x1, y1, x2, y2 = region
    crop = img[y1:y2, x1:x2]
    if crop is None or crop.size == 0:
        return None
    return Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))

def _extract_text_from_region(img, region):
    pil = _crop_to_pil(img, region)
    if pil is None:
        return ""
    return _ocr_pil_image(pil)

def extract_timer(image_path):
    """Extract timer text from the given image using modular OCR helpers.
    Returns the timer string like '1:23' or None if not found/recognized."""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    text = _extract_text_from_region(img, TIMER_REGION)
    if not text or not re.match(r'^\d+:\d{2}$', text):
        # Try alternate region if format is incorrect
        text = _extract_text_from_region(img, TIMER_REGION_ALT)

    return text.strip() if text and text.strip() else None

def extract_elixir(image_path):
    """Extract elixir count using modular OCR helpers.
    Returns an int if numeric elixir found, otherwise the raw text or None."""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Prefer the small numeric region first, fallback to elixir bar region
    text = _extract_text_from_region(img, ELIXIR_REGION)
    if not text:
        text = _extract_text_from_region(img, ELIXIR_BAR_REGION)

    if not text:
        return None

    # Try to parse as integer (reuse _parse_health for robust parsing)
    parsed = _parse_health(text)
    if parsed is not None:
        return parsed

    return text.strip() or None

def _parse_health(text):
    if not text:
        return None
    # Find first sequence of digits (allow commas/periods) and convert to int
    m = re.search(r'\d+', text.replace(" ", ""))
    if not m:
        return None
    num_str = m.group(0).replace(",", "").split(".")[0]
    try:
        return int(num_str)
    except ValueError:
        return None

def extract_tower_health(image_path, region, alt_region=None, try_plot=False):
    """Extract health for a single tower. Returns int health or None.
    If try_plot is True, displays the chosen crop (useful for debugging)."""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    text = _extract_text_from_region(img, region)
    health = _parse_health(text)
    used_region = region

    if health is None and alt_region:
        text_alt = _extract_text_from_region(img, alt_region)
        health = _parse_health(text_alt)
        if health is not None:
            used_region = alt_region

    if try_plot:
        pil = _crop_to_pil(img, used_region)
        if pil is not None:
            plot_cropped_image(cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR))

    return health

def extract_all_tower_healths(image_path):
    """Return dict with healths for all towers (None if not found)."""
    img_healths = {
        "left_ally": extract_tower_health(image_path, LEFT_ALLY_TOWER_REGION, LEFT_ALLY_TOWER_REGION_ALT),
        "right_ally": extract_tower_health(image_path, RIGHT_ALLY_TOWER_REGION, RIGHT_ALLY_TOWER_REGION_ALT),
        "left_enemy": extract_tower_health(image_path, LEFT_ENEMY_TOWER_REGION, LEFT_ENEMY_TOWER_REGION_ALT),
        "right_enemy": extract_tower_health(image_path, RIGHT_ENEMY_TOWER_REGION, RIGHT_ENEMY_TOWER_REGION_ALT),
        "ally_king": extract_tower_health(image_path, KING_TOWER_REGION, KING_TOWER_REGION_ALT),
        "enemy_king": extract_tower_health(image_path, ENEMY_KING_TOWER_REGION, ENEMY_KING_TOWER_REGION_ALT),
    }
    return img_healths

if __name__ == "__main__":
    time_remaining = extract_timer(IMAGE_PATH)
    print(f"🕒 Time Remaining: {time_remaining}")

    elixir_count = extract_elixir(IMAGE_PATH)
    print(f"💧 Elixir Count: {elixir_count}")

    tower_healths = extract_all_tower_healths(IMAGE_PATH)
    for tower, health in tower_healths.items():
        print(f"🏰 {tower.replace('_', ' ').title()} Tower Health: {health}")
