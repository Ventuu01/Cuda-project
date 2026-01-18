
from PIL import Image
import os

def save_ppm_p3(img: Image.Image, out_path: str, maxval: int = 255) -> None:
    img = img.convert("RGB")
    w, h = img.size
    pixels = list(img.getdata())

    with open(out_path, "w", newline="\n") as f:
        f.write("P3\n")
        f.write(f"{w} {h}\n")
        f.write(f"{maxval}\n")
        for (r, g, b) in pixels:
            f.write(f"{r} {g} {b}\n")

def convert_folder_jpg_to_p3(input_dir: str, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)

    for name in os.listdir(input_dir):
        if not name.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        in_path = os.path.join(input_dir, name)
        base = os.path.splitext(name)[0]
        out_path = os.path.join(output_dir, base + ".ppm")

        img = Image.open(in_path)
        save_ppm_p3(img, out_path)
        print("Saved P3:", out_path)

if __name__ == "__main__":
    convert_folder_jpg_to_p3("data/input/archive/", "data/input/dataset_1/")
