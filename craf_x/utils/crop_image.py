from PIL import Image
import sys

def crop_top_half(img_path, out_path):
    with Image.open(img_path) as img:
        width, height = img.size
        # Crop the top half
        cropped_img = img.crop((0, 0, width, height // 2))
        cropped_img.save(out_path)
        print(f"Cropped image saved to {out_path}")

if __name__ == "__main__":
    crop_top_half(sys.argv[1], sys.argv[2])
