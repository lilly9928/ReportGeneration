import os  # when loading file paths
import pandas as pd  # for lookup in annotation file
import torch
from torch.nn.utils.rnn import pad_sequence  # pad batch
from torch.utils.data import DataLoader, Dataset
from PIL import Image  # Load img
import torchvision.transforms as transforms
from pytorch_transformers import BertTokenizer

class XrayDataset(Dataset):
    def __init__(self, image_file,fourier_image_file, annotation_file, transform=None):
        self.image_file = image_file
        self.fourier_image_file = fourier_image_file
        self.df = pd.read_csv(annotation_file)
        self.transform = transform
        self.data_length = len(self.df)

        for idx in range(self.data_length):
            if pd.isna(self.df["caption"][idx]):
                print(f"{idx}delete")
                self.df.drop([idx], axis=0)

        # Get img, caption
        self.image_path_1 = self.df["image_name_0"]
        self.image_path_2 = self.df["image_name_1"]
        self.keywords = self.df["keyword"]
        self.captions = self.df["caption"]



    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):


        img1_id = self.image_path_1[index]
        img2_id = self.image_path_2[index]

        caption = self.captions[index]
        if pd.isna(caption):
            caption = self.keywords[index]
        keyword = self.keywords[index]

        img_1 = Image.open(os.path.join(self.image_file, img1_id))
        if pd.isna(img2_id):
            img_2 = Image.open(os.path.join(self.image_file, img1_id))
        else:
            img_2 = Image.open(os.path.join(self.image_file, img2_id))

        if self.transform is not None:
            img_1 = self.transform(img_1)
            img_2 = self.transform(img_2)

        return img_1,img_2, caption,keyword

class MyCollate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        img1 = [item[0].unsqueeze(0) for item in batch]
        img1 = torch.cat(img1, dim=0)
        img2 = [item[1].unsqueeze(0) for item in batch]
        img2 = torch.cat(img2, dim=0)
        caption = [item[2] for item in batch]
        caption = pad_sequence(caption, batch_first=False, padding_value=self.pad_idx)
        keyword = [item[3] for item in batch]
        keyword = pad_sequence(keyword, batch_first=False, padding_value=self.pad_idx)

        return img1,img2,caption, keyword

def get_loader(
        fourier_image_file,
        image_file,
        annotation_file,
        transform,
        batch_size=32,
        num_workers=4,
        shuffle=True,
        pin_memory=True,
):

    dataset = XrayDataset(image_file,fourier_image_file,annotation_file,transform)

    pad_idx = '[PAD]'

    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        #collate_fn=MyCollate(pad_idx=pad_idx),
    )

    return loader, dataset


if __name__ == "__main__":
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Resize((224,224))]
    )

    # loader, dataset = get_loader(
    #     fourier_image_file="D:/data/iuct/preprocess/images/fourier",
    #     image_file="D:/data/iuct/origin/images_normalized",
    #     annotation_file='D:/data/iuct/preprocess/new_caption.csv',
    #     transform=transform
    # )

    loader,dataset = get_loader( image_file="D:/data/iuct/origin/images_normalized",fourier_image_file="D:/data/iuct/preprocess/images/fourier",
        annotation_file='D:/data/iuct/preprocess/new_caption.csv',
        transform=transform)

    loader = DataLoader(dataset,batch_size=2)


    for idx, (img1,img2, captions, keyword) in enumerate(loader):
        print(img1)
        print(img2)
        print(captions)
        print(keyword)
        exit()

