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
from tqdm import tqdm

class Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path:str, audio_chunk_length:int=5, dataset_fps:int=60, label_format = "one_hot", class_list_path = "None"):
        dataset_path = Path(dataset_path)
        self.label_format = label_format
        self.audio_file_path = dataset_path/"sliced_audio_features"
        self.sliced_label_path = dataset_path/"sliced_labels_SE"

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

        self.audio_chunk_length = audio_chunk_length
        self.dataset_fps = dataset_fps
        self.label_converter = LabelConverter(label_list_path=class_list_path)
        self.len_of_label = self.label_converter.get_length_labels()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.first_chunk_list, self.last_chunk_list = self.preprocess_chunks()

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        audio_file = self.data_list[idx]
        label_file = self.label_list[idx]
        assert audio_file.stem[:-6] == label_file.stem[:-6], f"File name mismatch: {audio_file.stem[:-6]} != {label_file.stem[:-6]}"
        audio_feature = np.load(audio_file, allow_pickle=True).item()["audio"]
        label = np.load(label_file, allow_pickle=True).item()["data"]
        if label_file.stem in self.first_chunk_list:
            # if the file is the first chunk, repalce the first label with SOS
            label[0,:] = ['Start of Sequence', 'SOS']
        if label_file.stem in self.last_chunk_list:
            # if the file is the last chunk, replace the last label with EOS
            label[-1,:] = ["End of Sequence", "EOS"]
       
        # convert label to number
        if self.label_format == "one_hot":
            vectorized_label_converter = np.frompyfunc(self.label_converter.subclass_label_to_one_hot, 1, 1)
        else:
            vectorized_label_converter = np.frompyfunc(self.label_converter.subclass_label_to_number, 1, 1)
        subclass_label =vectorized_label_converter(label[:,1])
        subclass_label = np.stack(subclass_label).astype(np.int16)
        audio_feature = torch.tensor(audio_feature).float()
        subclass_label = torch.tensor(subclass_label, dtype=torch.int64)
        if label_file.stem in self.first_chunk_list:
            assert subclass_label[0] == 0, f"First label is not SOS: {subclass_label[0]}"
        if label_file.stem in self.last_chunk_list:
            assert self.label_converter.subclass_number_to_label(int(subclass_label[-1])) == "EOS", f"Last label is not EOS: {subclass_label[-1]}"
        return audio_feature, subclass_label
    
    def preprocess_chunks(self):
        # find the first and last chunk of the audios
        if not self.label_list:
            raise Exception("please load the dataset first")
        chunk_list = [path.stem for path in self.label_list]
        chunk_list = sorted(list(set(chunk_list)))
        # 1. Group files by their base name
        grouped_files = {}
        for filename in chunk_list:
            base_name, chunk_number = self.get_base_name_and_chunk(filename)
            if base_name not in grouped_files:
                grouped_files[base_name] = []
            grouped_files[base_name].append(filename)

        # 2. Sort files within each group by chunk number and extract first/last
        first_chunk_list = []
        last_chunk_list = []
        for base_name, file_list in grouped_files.items():
            # Sort the list using a custom key that extracts the chunk number
            file_list.sort(key=lambda fname: self.get_base_name_and_chunk(fname)[1])

            if file_list: # Ensure the list is not empty
                first_chunk_file = file_list[0]
                last_chunk_file = file_list[-1]
                first_chunk_list.append(first_chunk_file)
                last_chunk_list.append(last_chunk_file)
        return first_chunk_list, last_chunk_list

    def get_base_name_and_chunk(self, filename):
        """
        Extracts the base name (before _chunk) and the chunk number from a filename.
        Returns a tuple (base_name, chunk_number) or (filename, -1) if no chunk found.
        """
        match = re.search(r'_chunk(\d+)_label$', filename)
        if match:
            base_name = filename[:match.start()]
            chunk_number = int(match.group(1))
            return base_name, chunk_number
        else:
            # Return the filename itself and -1 as chunk if pattern doesn't match
            return filename, -1


     
    

class SchdulerDataModule(pl.LightningDataModule):
    def __init__(self, dataset_path:str, batch_size:int=32, label_format = "number", class_list_path = "None", num_workers:int=4):
        super().__init__()
        self.dataset = Dataset(dataset_path, label_format=label_format, class_list_path=class_list_path)
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        self.train_dataset = self.dataset
        self.val_dataset = self.dataset
        return self.batch_size, self.train_dataset[0], self.val_dataset[0]

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)
    
    def get_label_length(self):
        return self.dataset.len_of_label
    


if __name__ == "__main__":
    data_path = "/ihchomes/peng2000/editdance/editable_dance_project/data/motorica_beats"
    dataset = Dataset(data_path, label_format = "number", class_list_path = "/ihchomes/peng2000/editdance/Music2MotionScheduler/src/data_processing/class_list.txt")
    audio_size = torch.Size([1000, 417])
    subclass_size = torch.Size([1000, 233])

    for i in tqdm(range(len(dataset)), desc="Processing dataset"):
        audio_feature, subclass_label = dataset[i]
        print(f'subclass_label shape: {subclass_label})')
        exit()
    # labels = Path("/ihchomes/peng2000/editdance/data/motorica_dance_dataset/label/kthjazz_gCH_sFM_cAll_d02_mCH_ch01_beatlestreetwashboardbandfortyandtight_003.pkl")
    # with open(labels, "rb") as f:
    #     data = pickle.load(f)
    # print(data)
    
    