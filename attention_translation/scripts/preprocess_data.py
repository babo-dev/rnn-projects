import sys
import os

project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_dir)

from data.preprocess import preprocess_data

if __name__ == "__main__":
    file_path = project_dir + "/data/raw/eng-tk.txt"
    output_path = project_dir + "/data/processed/eng-tuk_processed.txt"
    preprocess_data(file_path, output_path)
