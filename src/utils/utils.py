from pathlib import Path
import numpy as np


# class that convert label to number and number to label
class LabelConverter:
    def __init__(self, label_list_path = "None"):
        if label_list_path == "None":
            self.label_list_path = Path("/fs/nexus-projects/PhysicsFall/Music2MotionScheduler/src/data_processing/class_list.txt")
            if not self.label_list_path.exists():
                raise FileNotFoundError(f"Default Label list file not found at {self.label_list_path}")
        else:
            self.label_list_path = Path(label_list_path)
            if not self.label_list_path.exists():
                raise FileNotFoundError(f"Label list file not found at {self.label_list_path}")
        self.load_label_list()
            
    def load_label_list(self):
        with open(self.label_list_path, "r") as f:
            self.label_list = [eval(line.strip()) for line in f.readlines()]
        # append [None, None] to the label list
        self.label_list.append([None, None])
        self.subclass_label_dict = {label[1]: i for i, label in enumerate(self.label_list)}
        self.subclass_number_dict = {i: label[1] for i, label in enumerate(self.label_list)}

        # get unique class labels while keeping the order
        self.unique_labels = []
        for label in self.label_list:
            if label[1] not in self.unique_labels:
                self.unique_labels.append(label[1])
        self.class_label_dict = {label: i for i, label in enumerate(self.unique_labels)}
        self.class_number_dict = {i: label for i, label in enumerate(self.unique_labels)}

        print(f"Loaded {len(self.label_list)} labels from {self.label_list_path}")
        return self.label_list
    
    def subclass_label_to_number(self, label):
        if label:
            if label in self.subclass_label_dict:
                return self.subclass_label_dict[label]
            else:
                raise ValueError(f"Label {label} not found in label list")
        else:
            return -1
    
    def subclass_number_to_label(self, number):
        if number in self.subclass_number_dict:
            return self.subclass_number_dict[number]
        else:
            raise ValueError(f"Number {number} not found in label list")
        
    def subclass_label_to_one_hot(self, label):
        # convert label to one hot vector
        if label in self.subclass_label_dict:
            one_hot = np.zeros(len(self.label_list))
            one_hot[self.subclass_label_dict[label]] = 1
            return one_hot
        else:
            raise ValueError(f"Label {label} not found in label list")
        
    def subclass_one_hot_to_label(self, one_hot):
        # convert one hot vector to label
        if isinstance(one_hot, np.ndarray) and len(one_hot) == len(self.label_list):
            index = np.argmax(one_hot)
            return self.subclass_number_dict[index]
        else:
            raise ValueError(f"One hot vector {one_hot} not valid")
        
    
    def class_label_to_number(self, label):
        if label in self.class_label_dict:
            return self.class_label_dict[label]
        else:
            raise ValueError(f"Label {label} not found in label list")
    
    def class_number_to_label(self, number):
        if number in self.class_number_dict:
            return self.class_number_dict
        else:
            raise ValueError(f"Number {number} not found in label list")
        
    def class_label_to_one_hot(self, label):
        if label in self.class_label_dict:
            one_hot = np.zeros(len(self.unique_labels))
            one_hot[self.class_label_dict[label]] = 1
            return one_hot
        else:
            raise ValueError(f"Label {label} not found in label list")
    
    def class_one_hot_to_label(self, one_hot):
        if isinstance(one_hot, np.ndarray) and len(one_hot) == len(self.unique_labels):
            index = np.argmax(one_hot)
            return self.class_number_dict
        else:
            raise ValueError(f"One hot vector {one_hot} not valid")
        
        
        



if __name__ == "__main__":
    print(f'Initializing LabelConverter...')
    label_converter = LabelConverter()
    print(f"Label to number: {label_converter.subclass_label_to_number}")
    print(f"Number to label: {label_converter.subclass_number_to_label}")
            