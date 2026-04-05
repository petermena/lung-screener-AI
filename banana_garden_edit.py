#!/usr/bin/env python3
"""
Banana Garden Edit -- Run this locally on your machine.
Edits your garden photo via Gemini API, adding all 21 plant varieties.

Usage:
    python3 banana_garden_edit.py --image ~/Desktop/garden.jpg

Requires: Python 3.8+ (no pip installs needed)
"""

import argparse
import base64
import json
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path

API_KEY = "AIzaSyDvscytn6G-5wkXBFQnkqV705WNXwWJKJk"
MODEL = "gemini-3.1-flash-image-preview"
OUTPUT_DIR = Path.home() / "Documents" / "nanobanana_generated"

EDIT_PROMPT = """Preserve the entire structural scene exactly as photographed: \
the cedar wood post and black wire mesh fence enclosure, central black iron gate, \
suburban house with blue-grey siding and dark shingle roof on the left, \
deciduous trees in background, paved driveway beyond the fence, \
green grass lawn in foreground. ONLY modify the interior of the raised garden beds.

Fill all raised beds with lush peak-summer vegetable growth — 2 of each variety:

- Calabrian chili pepper plants with clusters of small vivid red-orange chilis
- Napa cabbage plants with large pale-green tightly wrapped heads
- Sage plants with silvery-green oval leaves and purple flower spikes
- London tomato plants staked tall against the wire mesh with clusters of red tomatoes
- Cilantro plants with delicate white umbel flowers
- Zucchini plants with broad dark-green leaves and golden trumpet blossoms
- Cucumber vines climbing the wire mesh with hanging cucumbers
- Tall dill plants with feathery yellow-green fronds
- Butternut squash vines sprawling across the bed with beige-tan oblong fruits
- Deep-purple eggplant plants with large glossy oval fruit
- Habanero pepper plants with small orange lantern-shaped peppers
- Jalapeño plants with slender dark-green peppers
- Kale plants with deep blue-green heavily ruffled leaves
- Cherry tomato plants heavy with small bright-red cherry tomatoes
- Bushy basil plants with large glossy green leaves and white flower tips
- Flat-leaf Italian parsley plants
- Low-growing thyme plants with tiny purple flowers
- Watermelon vines with striped green oval fruits peeking through foliage
- Rosemary shrubs with upright needle-like blue-grey leaves
- Marjoram plants with small rounded aromatic leaves
- Tomatillo plants with papery pale-green husk-lantern fruits

Rich dark soil visible between plant bases. Maintain the original photograph's \
natural daylight, perspective, color accuracy, and photorealistic style. \
Sony A7R IV aesthetic, Better Homes and Gardens editorial quality."""


def edit_image(image_path: str):
    image_path = Path(image_path).expanduser().resolve()
    if not image_path.exists():
        print(f"Error: Image not found at {image_path}")
        sys.exit(1)

    print(f"Reading image: {image_path}")
    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode("utf-8")

    # Detect mime type
    suffix = image_path.suffix.lower()
    mime = {"jpg": "image/jpeg", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
            ".png": "image/png", ".webp": "image/webp"}.get(suffix, "image/jpeg")

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent?key={API_KEY}"

    body = {
        "contents": [{
            "parts": [
                {"inline_data": {"mime_type": mime, "data": image_data}},
                {"text": EDIT_PROMPT}
            ]
        }],
        "generationConfig": {
            "responseModalities": ["TEXT", "IMAGE"],
            "imageConfig": {"aspectRatio": "4:3"}
        }
    }

    print("Sending to Gemini API... (this may take 30-60 seconds)")
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                result = json.loads(resp.read().decode("utf-8"))
            break
        except urllib.error.HTTPError as e:
            body_text = e.read().decode("utf-8")
            if e.code == 429 and attempt < 2:
                wait = 2 ** (attempt + 1)
                print(f"Rate limited. Waiting {wait}s...")
                time.sleep(wait)
                req = urllib.request.Request(url, data=data,
                    headers={"Content-Type": "application/json"}, method="POST")
                continue
            print(f"API Error {e.code}: {body_text[:300]}")
            sys.exit(1)

    candidates = result.get("candidates", [])
    if not candidates:
        print("No candidates returned:", json.dumps(result)[:300])
        sys.exit(1)

    parts = candidates[0].get("content", {}).get("parts", [])
    image_out = None
    text_out = ""
    for part in parts:
        if "inlineData" in part:
            image_out = part["inlineData"]["data"]
        elif "text" in part:
            text_out = part.get("text", "")

    if not image_out:
        finish = candidates[0].get("finishReason", "UNKNOWN")
        print(f"No image returned. finishReason: {finish}")
        if text_out:
            print("Model said:", text_out[:300])
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = OUTPUT_DIR / f"garden_edit_{timestamp}.png"
    with open(out_path, "wb") as f:
        f.write(base64.b64decode(image_out))

    print(f"\nSuccess! Saved to: {out_path}")
    if text_out:
        print(f"Model note: {text_out[:200]}")
    return str(out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Edit garden photo with Gemini")
    parser.add_argument("--image", required=True, help="Path to your garden photo")
    args = parser.parse_args()
    edit_image(args.image)
