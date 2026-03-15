import os
import urllib.request

def download_data():
    # Provide the absolute path to the main project directory (.data will be created here)
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_dir = os.path.join(base_dir, '.data')
    
    # Create the .data directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)
    
    # Correct HuggingFace raw URLs (changing blob/main to resolve/main)
    datasets = {
        'train_df.csv': 'https://huggingface.co/datasets/Sp1786/multiclass-sentiment-analysis-dataset/resolve/main/train_df.csv',
        'test_df.csv': 'https://huggingface.co/datasets/Sp1786/multiclass-sentiment-analysis-dataset/resolve/main/test_df.csv',
        'val_df.csv': 'https://huggingface.co/datasets/Sp1786/multiclass-sentiment-analysis-dataset/resolve/main/val_df.csv'
    }
    
    for filename, url in datasets.items():
        filepath = os.path.join(data_dir, filename)
        if not os.path.exists(filepath):
            print(f"Downloading {filename}...")
            urllib.request.urlretrieve(url, filepath)
            print(f"Successfully downloaded {filename} to {filepath}")
        else:
            print(f"File {filename} already exists at {filepath}. Skipping download.")

if __name__ == '__main__':
    download_data()
