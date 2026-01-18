from PIL import Image
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR))  # script is in project root in your case

INPUT_ROOT = os.path.join(PROJECT_ROOT, "data", "output")
OUTPUT_ROOT = os.path.join(INPUT_ROOT, "converted_images")

for root, dirs, files in os.walk(INPUT_ROOT):
    # Avoid converting already converted images
    if os.path.abspath(root).startswith(os.path.abspath(OUTPUT_ROOT)):
        continue

    for file in files:
        if not file.lower().endswith(".ppm"):
            continue

        ppm_path = os.path.join(root, file)

        rel_path = os.path.relpath(root, INPUT_ROOT)
        out_dir = os.path.join(OUTPUT_ROOT, rel_path)
        os.makedirs(out_dir, exist_ok=True)

        png_name = os.path.splitext(file)[0] + ".png"
        png_path = os.path.join(out_dir, png_name)

        try:
            img = Image.open(ppm_path)
            img.save(png_path)
            print("Converted:", png_path)
        except Exception as e:
            print("Failed:", ppm_path, e)
