import os
import json
import time
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

GITHUB_URL = "https://api.github.com/repos/smlbiobot/cr-assets-png/contents/assets/sc/"
CLASSES_PATH = "classes.json"
SAVE_DIR = "sprites"
MAX_RETRIES = 3
MAX_WORKERS = 8


def safe_request(url, retries=MAX_RETRIES):
    """Safely fetch data from a URL with retry logic."""
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=10)
            if r.status_code == 200:
                return r
            elif r.status_code == 403:
                print("⚠️ GitHub API rate limit reached, sleeping 60s...")
                time.sleep(60)
            else:
                print(f"⚠️ HTTP {r.status_code} for {url}")
                break
        except requests.RequestException as e:
            print(f"⚠️ Error fetching {url}: {e}")
            time.sleep(2 ** attempt)
    return None


def download_file(url, save_path):
    """Download a single file with retry logic."""
    r = safe_request(url)
    if r:
        with open(save_path, "wb") as f:
            f.write(r.content)
        return True
    return False


def download_class_sprites(local_name, github_name):
    """Download all sprites for a single class."""
    base_url = f"{GITHUB_URL}chr_{github_name}_out/"
    save_dir = os.path.join(SAVE_DIR, local_name)
    os.makedirs(save_dir, exist_ok=True)

    r = safe_request(base_url)
    if not r:
        print(f"❌ Failed to fetch directory listing for {github_name}")
        return 0

    files = r.json()
    png_files = [f for f in files if f["name"].endswith(".png")]
    count = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = []
        for file in png_files:
            save_path = os.path.join(save_dir, file["name"])
            futures.append(executor.submit(download_file, file["download_url"], save_path))

        for f in tqdm(as_completed(futures), total=len(futures), desc=f"⬇️ {local_name}", ncols=80):
            if f.result():
                count += 1

    print(f"✅ {local_name}: Downloaded {count} sprites → {save_dir}")
    return count


def main():
    if not os.path.exists(CLASSES_PATH):
        print(f"❌ {CLASSES_PATH} not found.")
        return

    classes = json.load(open(CLASSES_PATH))
    total = 0

    for local_name, github_name in classes.items():
        total += download_class_sprites(local_name, github_name)

    print(f"\n🎉 Done! Downloaded {total} total sprites across {len(classes)} classes.")


if __name__ == "__main__":
    main()
