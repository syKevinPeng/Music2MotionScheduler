from email.mime import audio
import torch 
from pathlib import Path
import numpy as np
import re

class Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path:str, label_path:str):
        dataset_path = Path(dataset_path)
        label_path = Path(label_path)

        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset path {dataset_path} does not exist")
        if not label_path.exists():
            raise FileNotFoundError(f"Label path {label_path} does not exist")
        
        # get all the files in the dataset
        self.data_list = [
            file for file in dataset_path.iterdir() if file.suffix == ".npy"
        ]
        self.all_label  = np.load(label_path, allow_pickle=True)
        self.audio_chunk_length = 5


    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        file_name = self.data_list[idx]
        audio_feature = np.load(file_name)
        audio_file_name = file_name.stem
        chunk_num_match = re.search(r'chunk(\d+)', audio_file_name.split("_")[-2])
        if chunk_num_match:
            chunk_num = int(chunk_num_match.group(1))
        else:
            raise ValueError(f"Chunk number not found in {audio_file_name}")
        label_file_name = "_".join(audio_file_name.split("_")[:-2]) + ".mp4"
        label_sequence = self.all_label[label_file_name]
        print(f'label sequence:{label_sequence.shape}')
        label = label_sequence[:, chunk_num:chunk_num+self.audio_chunk_length]
        print(f'audio shape:{audio_feature.shape}, label shape:{label.shape}')
        return audio_feature, label
    
if __name__ == "__main__":
    data_path = "/fs/nexus-projects/PhysicsFall/data/motorica_dance_dataset/sliced_audio_features"
    label_path = "/fs/nexus-projects/PhysicsFall/data/motorica_dance_dataset/all_label.pkl"
    dataset = Dataset(data_path, label_path)

    audio_feature, label = dataset[0]
    