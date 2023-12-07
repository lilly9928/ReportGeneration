import torch,gc
gc.collect()
torch.cuda.empty_cache()
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from data_loader.dataloader_old import get_loader
from utils.rg_utils import save_checkpoint
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch
from PIL import Image


def train():
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((224,224)),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406],
            #                      std=[0.229, 0.224, 0.225]),
            # transforms.RandomHorizontalFlip(p=0.8),
            # transforms.RandomRotation(degrees=(-30, 30), interpolation=transforms.InterpolationMode.BILINEAR, fill=0),

        ]
    )

    otransform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((224,224)),
            # transforms.RandomHorizontalFlip(p=0.8),
            # transforms.RandomRotation(degrees=(-30, 30), interpolation=transforms.InterpolationMode.BILINEAR, fill=0),

        ]
    )


#iuxray 데이터 셋
    train_loader, dataset = get_loader(
        root_folder = "D:/data/iuct/preprocess/images/fourier", oroot_folder="D:/data/iuct/origin/images_normalized",all_caption = "D:/data/iuct/preprocess/caption.txt", annotation_file="D:/data/iuct/preprocess/caption_train.txt",transform=transform,otransform=otransform
    )


    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    load_model = False
    save_model = True

    step = 0

    total_loss = []

    model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

    criterion = nn.CrossEntropyLoss(ignore_index=dataset.vocab.stoi["<PAD>"])
    optimizer = optim.Adam(model.parameters(), lr = 0.001)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer = optimizer, lr_lambda=lambda epoch:0.95 ** epoch,last_epoch=-1,verbose=False)

    # if load_model:
    #     step = load_checkpoint(torch.load("D:\GitHub\-WIL-Expression-Recognition-Study\Study\Imagecaption\checkpoint_coco_30.pth.tar"),model,optimizer)

    model.train()

    start_token = torch.tensor([1], dtype=torch.long)
    average_loss = 0
    for epoch in range(3):
       # print_ixray_examples(model,start_token,device,dataset)
        if save_model:
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step":step,
            }
            save_checkpoint(checkpoint)

        for idx,(_, imgs,captions,labels) in enumerate(train_loader):

            imgs = imgs.to(device)
            captions = captions.permute(1,0).to(device)

            caption_input = captions[:,:-1]
            caption_expected = captions[:,1:]

            sequence_length = caption_input.size(1)

            tgt_mask = model.get_tgt_mask(sequence_length).to(device)

            optimizer.zero_grad()
            outputs,_,_ = model(imgs, caption_input, tgt_mask)

            outputs = outputs.permute(1,2,0)

            loss = criterion(outputs,caption_expected)

            step += 1

            loss.backward()
            optimizer.step()
        scheduler.step()

        print("epochs",epoch,"Training loss", loss.item())
        average_loss += loss.item()

    average_loss = average_loss/num_epochs
    total_loss.append(average_loss)
    print(total_loss)



if __name__ == "__main__":
    train()
