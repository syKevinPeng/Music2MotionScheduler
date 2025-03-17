from email.mime import audio
import torch 
from pathlib import Path
import numpy as np
import pickle
import re
import lightning as pl
import sys
sys.path.append(str(Path(__file__).parent.parent))
from utils.utils import LabelConverter

class Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path:str, audio_chunk_length:int=5, dataset_fps:int=60, label_format = "one_hot"):
        dataset_path = Path(dataset_path)
        self.label_format = label_format
        self.audio_file_path = dataset_path/"sliced_audio_features417"
        self.sliced_label_path = dataset_path/"sliced_labels"

        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset path {dataset_path} does not exist")
        if not self.sliced_label_path.exists():
            raise FileNotFoundError(f"Sliced label path {self.sliced_label_path} does not exist")
        if self.label_format not in ["one_hot", "number"]:
            raise ValueError(f"Label format {self.label_format} not supported. Choose from 'one_hot' or 'number'")
        # get all the files in the dataset
        self.data_list = [
            file for file in self.audio_file_path.iterdir() if file.suffix == ".npy"
        ]
        self.label_list = [
            file for file in self.sliced_label_path.iterdir() if file.suffix == ".npy"
        ]
        print(f"Found {len(self.data_list)} audio files and {len(self.label_list)} label files in the dataset")

        data_list_stem = [file.stem[:-6] for file in self.data_list]
        label_list_stem = [file.stem[:-6] for file in self.label_list]
        # find the common files
        common_files = set(data_list_stem).intersection(set(label_list_stem))
        self.data_list = sorted([file for file in self.data_list if file.stem[:-6] in common_files])
        self.label_list = sorted([file for file in self.label_list if file.stem[:-6] in common_files])
        print(f"Found {len(self.data_list)} common files in the dataset")
        print(f'Data list size: {len(self.data_list)}')
        print(f'Label list size: {len(self.label_list)}')

        # self.all_label  = np.load(label_path, allow_pickle=True)
        # filtered_data_list = []
        # pattern = re.compile(r"^(.*?)_chunk\d+_audio$")

        # for data in self.data_list:
        #     match = pattern.match(data.stem)
        #     if match:
        #         file_name = match.group(1)
        #         if file_name + ".mp4" in self.all_label:
        #             filtered_data_list.append(data)
        # self.data_list = filtered_data_list


        self.audio_chunk_length = audio_chunk_length
        self.dataset_fps = dataset_fps
        self.label_converter = LabelConverter()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        audio_file = self.data_list[idx]
        label_file = self.label_list[idx]
        assert audio_file.stem[:-6] == label_file.stem[:-6], f"File name mismatch: {audio_file.stem[:-6]} != {label_file.stem[:-6]}"
        audio_feature = np.load(audio_file, allow_pickle=True).item()["audio"]
        label = np.load(label_file, allow_pickle=True).item()["data"]
        # print(f'audio_feature shape: {audio_feature.shape}') # 500 x 417
        # print(f'label shape: {label.shape}') # 500 x 2
       
        # convert label to number
        if self.label_format == "one_hot":
            vectorized_label_converter = np.frompyfunc(self.label_converter.subclass_label_to_one_hot, 1, 1)
        else:
            vectorized_label_converter = np.frompyfunc(self.label_converter.subclass_label_to_number, 1, 1)
        subclass_label =vectorized_label_converter(label[:,1])
        subclass_label = np.stack(subclass_label).astype(np.int16)
        audio_feature = torch.tensor(audio_feature).float().to(self.device)
        subclass_label = torch.tensor(subclass_label, dtype=torch.float32).to(self.device)
        return audio_feature, subclass_label
    

class SchdulerDataModule(pl.LightningDataModule):
    def __init__(self, dataset_path:str, batch_size:int=32):
        super().__init__()
        self.dataset = Dataset(dataset_path)
        self.batch_size = batch_size

    def setup(self, stage=None):
        self.train_dataset = self.dataset
        self.val_dataset = self.dataset
        return self.batch_size, self.train_dataset[0], self.val_dataset[0]

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size)
    


if __name__ == "__main__":
    data_path = "/fs/nexus-projects/PhysicsFall/data/motorica_beats"
    dataset = Dataset(data_path)

    audio_feature, label = dataset[0]
    