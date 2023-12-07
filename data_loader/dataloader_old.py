import os  # when loading file paths
import pandas as pd  # for lookup in annotation file
import spacy  # for tokenizer
import torch
from torch.nn.utils.rnn import pad_sequence  # pad batch
from torch.utils.data import DataLoader, Dataset
from PIL import Image  # Load img
import torchvision.transforms as transforms
import numpy as np
from collections import Counter

# We want to convert text -> numerical values
# 1. We need a Vocabulary mapping each word to a index
# 2. We need to setup a Pytorch dataset to load the data_aug
# 3. Setup padding of every batch (all examples should be
#    of same seq_len and setup dataloader)
# Note that loading the image is very easy compared to the text!

# Download with: python -m spacy download en
spacy_eng = spacy.load("en_core_web_sm")


class Vocabulary:
    def __init__(self, freq_threshold):
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>",4:"<SEP>"}
        self.stoi = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3,"<SEP>":4}
        self.freq_threshold = freq_threshold

    def __len__(self):
        return len(self.itos)

    @staticmethod
    def tokenizer_eng(text):
        return [tok.text.lower() for tok in spacy_eng.tokenizer(str(text))]

    def build_vocabulary(self, sentence_list):
        frequencies = {}
        idx = 4

        for sentence in sentence_list:
            for word in self.tokenizer_eng(sentence):
                if word not in frequencies:
                    frequencies[word] = 1

                else:
                    frequencies[word] += 1

                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1

    def numericalize(self, text):
        tokenized_text = self.tokenizer_eng(text)

        return [
            self.stoi[token] if token in self.stoi else self.stoi["<UNK>"]
            for token in tokenized_text
        ]


class XrayDataset(Dataset):
    def __init__(self, root_dir,oroot_dir, captions_file, all_caption, transform=None, otransform = None, freq_threshold=5):
        self.root_dir = root_dir
        self.oroot_dir = oroot_dir
        self.df = pd.read_csv(captions_file)
        self.allcaption = pd.read_csv(all_caption)
        self.transform = transform
        self.otransform = otransform

        # Get img, caption coflumns
        self.oimgs = self.df["oimage"]
        self.imgs = self.df["image"]
        self.captions = self.df["caption"]
        self.label = self.df["label"]
        self.all_captions = self.allcaption["caption"]
        self.all_label = self.allcaption["label"]

        for idx in range(len(self.imgs)):
            if ' ' == self.captions[idx]:
                print(f"{idx}delete")
                self.df.drop([idx], axis=0)

        # Initialize vocabulary and build vocab
        self.vocab = Vocabulary(freq_threshold)
        self.vocab.build_vocabulary(self.all_captions.tolist())
        self.vocab.build_vocabulary(self.label.tolist())

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):

        caption = self.captions[index]
        img_id = self.imgs[index]
        oimg_id = self.oimgs[index]
        label = self.label[index]

        img = Image.open(os.path.join(self.root_dir, img_id))
        oimg = Image.open(os.path.join(self.oroot_dir, oimg_id))

        if self.transform is not None:
            img = self.transform(img)
            oimg = self.otransform(oimg)

        numericalized_caption = [self.vocab.stoi["<SOS>"]]
        numericalized_caption += self.vocab.numericalize(caption)
        numericalized_caption.append(self.vocab.stoi["<EOS>"])

        numbericalized_label = [self.vocab.stoi["<SEP>"]]
        numbericalized_label += self.vocab.numericalize(label)
        numbericalized_label.append(self.vocab.stoi["<SEP>"])

        caption = torch.tensor(numericalized_caption)
        label = torch.tensor(numbericalized_label)

        return oimg,img, caption, label


class MyCollate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        oimgs = [item[0].unsqueeze(0) for item in batch]
        oimgs = torch.cat(oimgs, dim=0)
        imgs = [item[1].unsqueeze(0) for item in batch]
        imgs = torch.cat(imgs, dim=0)
        targets = [item[2] for item in batch]
        targets = pad_sequence(targets, batch_first=False, padding_value=self.pad_idx)
        label = [item[3] for item in batch]
        label = pad_sequence(label, batch_first=False, padding_value=self.pad_idx)

        return oimgs,imgs, targets, label


def get_loader(
        root_folder,
        oroot_folder,
        annotation_file,
        all_caption,
        transform,
        otransform,
        batch_size=64,
        num_workers=4,
        shuffle=True,
        pin_memory=True,
):
    dataset = XrayDataset(root_folder,oroot_folder, annotation_file,all_caption, transform=transform,otransform=otransform)

    pad_idx = dataset.vocab.stoi["<PAD>"]

    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=pin_memory,
        collate_fn=MyCollate(pad_idx=pad_idx),
    )

    return loader, dataset


# if __name__ == "__main__":
    # transform = transforms.Compose(
    #     [transforms.ToTensor(),
    #      transforms.Resize((224,224))]
    # )
    #
    # loader, dataset = get_loader(
    #     "D:/data/iuct/images/fourier", "D:/data/iuct/images/images_normalized", "D:/data/iuct/caption_train.txt","D:/data/iuct/caption.txt",
    #     transform=transform
    # )
    #
    # for idx, (oimgs,imgs, captions, label) in enumerate(loader):
    #     print(oimgs.shape)
    #     print(imgs.shape)
    #     print(captions.shape)
    #     print(label.shape)
    #
    # # with open("D:/data/iuct/caption.txt",encoding='utf-8') as f :
    # #     lines = f.readlines()
    # # print(len(lines))
    # #
    # # len_list = []
    # # for _ in range(len(lines)):
    # #     print(str(_)+":"+str(len(lines[_].split(','))))
    # #     len_list.append(len(lines[_].split(',')))
    # #
    # # Counter(len_list)