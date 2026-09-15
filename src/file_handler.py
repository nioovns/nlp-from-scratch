import re
import os
OUTPUT_DIR = "./processed_output"
def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8', errors="ignore") as f:
        return f.read()


def write_file(text, file_path=None, output_dir=OUTPUT_DIR, base_name="output"):
    if file_path is None:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        file_path = os.path.join(output_dir, f"{base_name}.txt")
    else:
        dir_name = os.path.dirname(file_path)
        if dir_name and not os.path.exists(dir_name):
            os.makedirs(dir_name)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(text)
