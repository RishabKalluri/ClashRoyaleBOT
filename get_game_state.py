import cv2
import re
import pytesseract
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np

IMAGE_PATH = "IMG_4376.PNG" # IMG_4297, IMG_4376

# Format: (x1, y1, x2, y2)
TIMER_REGION = (1000, 180, 1160, 250)
TIMER_REGION_ALT = (990, 220, 1170, 290) # Alternate timer region for different layouts
ELIXIR_REGION = (320, 2400, 380, 2470)
ELIXIR_BAR_REGION = (300, 2380, 1170, 2510)

LEFT_ALLY_TOWER_REGION = (190, 1540, 340, 1590)
LEFT_ALLY_TOWER_REGION_2 = (190, 1550, 300, 1590)
LEFT_ALLY_TOWER_REGION_ALT = (195, 1575, 345, 1625) # Alternate left tower region
RIGHT_ALLY_TOWER_REGION = (885, 1540, 1035, 1590)
RIGHT_ALLY_TOWER_REGION_2 = (885, 1550, 995, 1590)
RIGHT_ALLY_TOWER_REGION_ALT = (880, 1575, 1030, 1625) # Alternate right tower region

LEFT_ENEMY_TOWER_REGION = (190, 365, 340, 425)
LEFT_ENEMY_TOWER_REGION_2 = (190, 365, 290, 415)
LEFT_ENEMY_TOWER_REGION_ALT = (195, 480, 345, 540) # Alternate left enemy tower region
RIGHT_ENEMY_TOWER_REGION = (885, 365, 1035, 425)
RIGHT_ENEMY_TOWER_REGION_2 = (885, 365, 995, 415)
RIGHT_ENEMY_TOWER_REGION_ALT = (880, 480, 1030, 540) # Alternate right enemy tower region

KING_TOWER_REGION = (0, 0, 0, 0)
KING_TOWER_REGION_ALT = (0, 0, 0, 0) # Alternate king tower region
ENEMY_KING_TOWER_REGION = (0, 0, 0, 0)
ENEMY_KING_TOWER_REGION_ALT = (515, 210, 715, 290) # Alternate enemy king tower region

# Map of tower identifiers to their primary region constants used by ClashRoyaleOCR._get_towers
TOWER_REGIONS = {
    "left_ally": LEFT_ALLY_TOWER_REGION_2,
    "right_ally": RIGHT_ALLY_TOWER_REGION_2,
    "left_enemy": LEFT_ENEMY_TOWER_REGION_2,
    "right_enemy": RIGHT_ENEMY_TOWER_REGION_2,
    # "ally_king": KING_TOWER_REGION,
    # "enemy_king": ENEMY_KING_TOWER_REGION,
}

def plot_cropped_image(image):
    """Plot the cropped image region. Accepts a PIL.Image or a BGR numpy array."""
    if isinstance(image, Image.Image):
        arr = np.array(image)
        bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    else:
        bgr = image
    _, ax = plt.subplots(1, 1, figsize=(4, 4))
    ax.imshow(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    ax.set_title("Image Region")
    ax.axis("off")
    plt.tight_layout()
    plt.show()

class ClashRoyaleOCR:
    def __init__(self):
        # Tesseract Config:
        # --psm 7: Treat the image as a single text line (crucial for digits)
        self.config = r'-l clash --psm 7 -c tessedit_char_whitelist=0123456789:'

    def get_game_state(self, image_path):
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Image not found at {image_path}")

        state = {
            "time": self._get_text(img, TIMER_REGION),
            "elixir": self._get_elixir(img),
            "towers": self._get_towers(img)
        }
        return state

    def _get_text(self, img, region):
        """Generic helper to crop and read text using the custom model."""
        x1, y1, x2, y2 = region
        # Crop: numpy uses img[y:y+h, x:x+w]
        crop = img[y1:y2, x1:x2]
        plot_cropped_image(crop)
        
        if crop.size == 0:
            return ""
        
        scale = 3
        w = int(crop.shape[1] * scale)
        h = int(crop.shape[0] * scale)
        processed = cv2.resize(crop, (w, h), interpolation=cv2.INTER_CUBIC)

        # Simple Preprocessing: just grayscale is enough for custom models
        gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        _, thresh = cv2.threshold(blurred, 190, 255, cv2.THRESH_BINARY)

        kernel = np.ones((2,2), np.uint8)
        cleaned = cv2.erode(thresh, kernel, iterations=1)
        cleaned = cv2.dilate(cleaned, kernel, iterations=1)

        inverted = cv2.bitwise_not(cleaned)
        padded = cv2.copyMakeBorder(inverted, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        plot_cropped_image(padded)
        
        # Run Tesseract with custom config
        text = pytesseract.image_to_string(padded, config=self.config)
        return text.strip()

    def _get_elixir(self, img):
        # Try to read the number
        text = self._get_text(img, ELIXIR_REGION)
        
        # Clean up result (sometimes it reads '4.5' as '4,5' or '4 5')
        if not text:
            return 0
        try:
            # Filter non-digits just in case
            digits = "".join(filter(str.isdigit, text))
            return int(digits) if digits else 0
        except ValueError:
            return 0

    def _get_towers(self, img):
        healths = {}
        for name, region in TOWER_REGIONS.items():
            text = self._get_text(img, region)
            
            if not text:
                healths[name] = None
            else:
                try:
                    # Remove any accidental chars and parse int
                    clean_text = text.replace(" ", "").replace(",", "").replace(".", "")
                    healths[name] = int(clean_text)
                except ValueError:
                    healths[name] = None
                    
        return healths

def _ensure_ocr_available():
    if "processor" not in globals() or "model" not in globals():
        raise RuntimeError("OCR processor/model not initialized. Uncomment or initialize them at the top of the file.")

def _ocr_pil_image(pil_img):
    _ensure_ocr_available()
    pixel_values = processor(pil_img, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values)
    return processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

def _ocr_pil_image_paddle(pil_img):
    cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    result = ocr.ocr(cv_img)
    if not result or not isinstance(result, list):
        return ""
    
    # PaddleOCR v3 returns list of dicts
    item = result[0]
    if isinstance(item, dict) and "transcription" in item:
        return item["transcription"].strip()
    
    # fallback for older versions
    try:
        return item[1][0].strip()
    except Exception:
        return ""

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
    return _ocr_pil_image_paddle(pil)

def extract_timer(image_path, try_plot=False):
    """Extract timer text from the given image using modular OCR helpers.
    Returns the timer string like '1:23' or None if not found/recognized."""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    text = _extract_text_from_region(img, TIMER_REGION)
    if try_plot and text:
        pil = _crop_to_pil(img, TIMER_REGION)
        if pil is not None:
            plot_cropped_image(cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR))
    if not text or not re.match(r'^\d:\d{2}$', text):
        # Try alternate region if format is incorrect
        text = _extract_text_from_region(img, TIMER_REGION_ALT)
        if try_plot:
            pil = _crop_to_pil(img, TIMER_REGION_ALT)
            if pil is not None:
                plot_cropped_image(cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR))

    return text.strip() if text and text.strip() else None

def extract_elixir(image_path, try_plot=False):
    """Extract elixir count using modular OCR helpers.
    Returns an int if numeric elixir found, otherwise the raw text or None."""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Prefer the small numeric region first, fallback to elixir bar region
    text = _extract_text_from_region(img, ELIXIR_REGION)
    if try_plot and text:
        pil = _crop_to_pil(img, ELIXIR_REGION)
        if pil is not None:
            plot_cropped_image(cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR))
    if not text:
        text = _extract_text_from_region(img, ELIXIR_BAR_REGION)
        if try_plot:
            pil = _crop_to_pil(img, ELIXIR_BAR_REGION)
            if pil is not None:
                plot_cropped_image(cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR))

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
    # time_remaining = extract_timer(IMAGE_PATH, try_plot=True)
    # print(f"🕒 Time Remaining: {time_remaining}")

    # elixir_count = extract_elixir(IMAGE_PATH)
    # print(f"💧 Elixir Count: {elixir_count}")

    # tower_healths = extract_all_tower_healths(IMAGE_PATH)
    # for tower, health in tower_healths.items():
    #     print(f"🏰 {tower.replace('_', ' ').title()} Tower Health: {health}")
    bot = ClashRoyaleOCR()
    print(bot.get_game_state("IMG_4376.PNG"))
