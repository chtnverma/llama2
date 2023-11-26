import torch
import os
import pandas as pd
from torchvision.io import read_image
from torchvision.transforms import Resize
from torch.utils.data import Dataset
from transformers import AutoImageProcessor, ResNetModel, AutoTokenizer


label_to_ignore = -100
image_size = (256, 256)
sequence_length = 4096
image_processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
image_repr_model = ResNetModel.from_pretrained("microsoft/resnet-50")
image_emb_size = 2048
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
resize_tf = Resize(image_size, antialias=None)
eos_token_id = tokenizer.eos_token_id

def transform_raw_image(image):
    # resize and embed image.
    image = resize_tf(image)
    processed_img = image_processor(image.transpose(0, 2).transpose(0,1), return_tensors="pt")
    outputs = image_repr_model(**processed_img)
    return outputs.pooler_output.squeeze().detach()

def transform_raw_text(text):
    return text + tokenizer.eos_token

def tokenize_and_collate(data):
    """
    Input: list of tuples of (image_embedding, text, img_path)
    Returns: (image, tokenized_text, target_ids), where each is a tensor.
    """
    texts = [d[1] for d in data]
    images = [d[0] for d in data]
    tokenized_texts = tokenizer(
        texts, 
        return_tensors='pt', 
        padding=True,
        truncation=True,
        max_length=sequence_length,
        add_special_tokens=False)
    images = torch.stack(images)
    target = torch.where(tokenized_texts["attention_mask"] != 0, tokenized_texts['input_ids'], label_to_ignore)
    return (images, tokenized_texts, target)

class FlickrDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=transform_raw_image, target_transform=transform_raw_text):
        self.img_labels = pd.read_csv(annotations_file, delimiter='|').sample(frac=1).reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 2]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(str(label))
        return image, img_path, label


class FakeDataset(Dataset):
    def __init__(self):
        self.x = 1
    def __len__(self):
        return 1000
    def __getitem__(self, idx):
        image = torch.rand(8192, 2048)
        label = torch.rand(2048, 8192)
        return image, label
