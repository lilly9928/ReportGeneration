import torch,gc
gc.collect()
torch.cuda.empty_cache()
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from data_loader.dataloader import get_loader
from models.model import CT2captionModel
#from model2 import CT2captionModel
from utils.rg_utils import save_checkpoint,print_ixray_examples
from transformers import AutoTokenizer



def train():
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((224,224))

        ]
    )

    otransform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((224,224)),
            transforms.RandomHorizontalFlip(p=0.8),
            transforms.RandomRotation(degrees=(-30, 30), interpolation=transforms.InterpolationMode.BILINEAR, fill=0),

        ]
    )


#iuxray 데이터 셋
    train_loader,dataset = get_loader( image_file="D:/data/iuct/origin/images_normalized",fourier_image_file="D:/data/iuct/preprocess/images/fourier",
        annotation_file='D:/data/iuct/preprocess/new_caption.csv',
        transform=transform)

    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = torch.device("cpu")
    load_model = False
    save_model = True

    #하이퍼파라미터
    dim_size = 512
    vocab_size = 50265
    num_heads = 8
    num_decoder_layers = [5]
    dropout_p = 0.01
    in_channels= 1
    patch_size = 16
    depth = 3
    learning_rate = 5e-4
    num_epochs = 30

    img_size = 224

    step = 0

    total_loss = []
    train_loss_list = []
    valid_loss_list = []
    global_step_list = []
    for layer in num_decoder_layers:

        #initialize model , loss etc
        print(f"===========decoder_layer : {layer}==============")
        model = CT2captionModel(vocab_size,dim_size,num_heads,layer,dropout_p,in_channels,patch_size,img_size,depth).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr = learning_rate)
        scheduler = optim.lr_scheduler.LambdaLR(optimizer = optimizer, lr_lambda=lambda epoch:0.95 ** epoch,last_epoch=-1,verbose=False)
        tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

        model.train()

        start_token = torch.tensor([1], dtype=torch.long)
        average_loss = 0
        for epoch in range(num_epochs):
            print_ixray_examples(model,start_token,device,dataset)
            if save_model:
                checkpoint = {
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step":step,
                }
                save_checkpoint(checkpoint)


            for idx,(_, imgs,captions,keywords) in enumerate(train_loader):

                imgs = imgs.to(device)
                captions= tokenizer(list(captions),padding=True)
                keywords= tokenizer(list(keywords),padding=True)

                captions = torch.tensor(captions['input_ids'])
                keywords = torch.tensor(keywords['input_ids'])

                # caption = torch.tensor(captions['input_ids']).permute(1,0).to(device)
                # keyword = torch.tensor(keywords['input_ids']).permute(1,0).to(device)

                caption_input = captions[:, :-1].to(device)
                caption_expected = captions[:, 1:].to(device)

                keywords_input = keywords[:, :-1].to(device)
                keywords_expected = keywords[:,1:].to(device)

                sequence_length = caption_input.size(1)
                label_sequence_length = keywords_input.size(1)

                tgt_mask = model.get_tgt_mask(sequence_length).to(device)
                label_mask = model.get_tgt_mask(label_sequence_length).to(device)

                optimizer.zero_grad()

                label_outputs,outputs= model(imgs, caption_input,keywords_input, tgt_mask,label_mask)

                outputs = outputs.permute(1,2,0)

                labels_outputs = label_outputs.permute(1, 2, 0)

                caption_loss = criterion(outputs,caption_expected)
                label_loss = criterion(labels_outputs, keywords_expected)

                loss = caption_loss+label_loss

                step += 1

                loss.backward()
                optimizer.step()
            scheduler.step()

            print("epochs",epoch,"Training loss", loss.item())
            average_loss += loss.item()

        metrics = {
            "train_loss_list": train_loss_list,
            "valid_loss_list": valid_loss_list,
            "global_step_list": global_step_list,
        }
        #save_metrics(metrics)



if __name__ == "__main__":
    train()
