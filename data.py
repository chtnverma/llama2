import torch
import os
import pandas as pd
from torchvision.io import read_image
from torchvision.transforms import Resize
from torch.utils.data import Dataset, random_split, DataLoader
from transformers import AutoImageProcessor, ResNetModel, AutoTokenizer
import tiktoken


image_size = (256, 256)
# content_length = 1024
# padding_idx = -100
sequence_length = 4096
image_processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
image_repr_model = ResNetModel.from_pretrained("microsoft/resnet-50")
image_emb_size = 2048
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
padding_idx = tokenizer.pad_token_id
eos_token_idx = tokenizer.eos_token_id

# encode = lambda s: tokenizer(s, return_tensors="pt") # for HF's gpt
# eos_token_id = enc.eos_token_id


def transform_raw_image(image):
    resize_tf = Resize(image_size, antialias=None)
    image = resize_tf(image)
    processed_img = image_processor(image.transpose(0, 2).transpose(0,1), return_tensors="pt")
    outputs = image_repr_model(**processed_img)
    return outputs.pooler_output.squeeze().detach()


def collate_with_tokenize(data):
    texts = [d[1] for d in data]
    images = [d[0] for d in data]
    # print("CHETAN DEBUG:  texts = ", texts)
    # print("------------ ")
    tokenized_texts = tokenizer(
        texts, 
        return_tensors='pt', 
        padding=True,
        truncation=True,
        max_length=sequence_length)
    images = torch.stack(images)
    return (images, tokenized_texts)

    # try:
    #     ids = encode(text)
    # except ValueError as err:
    #     print(f"Catching an error at ids = encode(text). The text ={text}. Error ={err}.")
    #     print(f'Other details: type={type(text)}. len={len(text)}')
    #     raise Exception("Boo!")        
    
    # # return torch.Tensor([])
    # # unpadded = (torch.tensor(ids['input_ids'], dtype=torch.long)[None, ...])
    # unpadded = ids['input_ids'].long()[None, ...]
    # unpadded = unpadded[0, :content_length-1]
    # padding = torch.full((1, content_length - 1 - unpadded.shape[1]), padding_idx)
    # return torch.cat([unpadded, padding], dim=1).squeeze()



class FlickrDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
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
            label = self.target_transform(label)
        return image, label



class FakeDataset(Dataset):
    def __init__(self):
        self.x = 1
    def __len__(self):
        return 1000
    def __getitem__(self, idx):
        # image = torch.rand([2048])
        # label = torch.randint(10683, [1023])
        
        # Below data is to be used with FakeModel:
        image = torch.rand(8192, 2048)
        label = torch.rand(2048, 8192)
        return image, label
