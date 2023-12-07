import torch
import torchvision.transforms as transforms
from PIL import Image
import time

now =time.localtime()
def print_examples(model, start_token,device, dataset):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    model.eval()
    test_img1 = transform(Image.open("D:/data/brain/test_image/1.jpg").convert("L")).unsqueeze(0)
    print("Example 1 CORRECT: Small SAH  right parietal lobe sulci Diffuse brain atrophy with ventriculomegaly Suggestive of small vessel disease  both cerebral white matter and basal ganglia ")
    print(
        "Example 1 OUTPUT: "
        + " ".join(model.example_images(test_img1.to(device), start_token.to(device),dataset.vocab,device)))
    test_img2 = transform(
        Image.open("D:/data/brain/test_image/2.jpg").convert("L")).unsqueeze(0)
    print("Example 2 CORRECT: Acute SDH along bilateral convexities and falx Associated SAH along the basal cisterns and cerebral sulci")
    print(
        "Example 2 OUTPUT: "
        + " ".join(model.example_images(test_img2.to(device), start_token.to(device),dataset.vocab,device)))
    test_img3 = transform(Image.open("D:/data/brain/test_image/3.jpg").convert("L")).unsqueeze(0)
    print("Example 3 CORRECT: Multi cisternal acue SAH  along basal sylvian ciserns and cortical cisterns Associated IVH hydrocephalus Skull vault is unremarkable Conclusion Acute SAH IVH  hydrocephalus")
    print(
        "Example 3 OUTPUT: "
        + " ".join(model.example_images(test_img3.to(device), start_token.to(device),dataset.vocab,device)))
    test_img4 = transform(Image.open("D:/data/brain/test_image/4.jpg")).unsqueeze(0)
    print("Example 4 CORRECT:Clinical information  trauma Acute SAH along the basal cisterns and both cerebral sulci  right left small acute SDH along both frontal convexities and falx Small amount of pneumocephalus noted in left side cavernous sinus and T S sinuses  rather likely IV related air than trauma induced Recommend   Clinical correlation")
    print("Example 4 OUTPUT: "+ " ".join(model.example_images(test_img4.to(device), start_token.to(device),dataset.vocab,device)))
    model.train()


def print_ixray_examples(model, start_token,device, dataset):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    model.eval()
    test_img1: object = transform(Image.open("D:/data/iuct/preprocess/images/fourier/3228_IM-1526-2001.dcm.png.jpg").convert("L")).unsqueeze(0)
    test_oimg1 = transform(Image.open("D:/data/iuct/origin/images_normalized/3228_IM-1526-2001.dcm.png").convert("L")).unsqueeze(0)
    print("Example 1 CORRECT: The heart is normal in size. The mediastinum is unremarkable. There is again biapical scarring. Small stable calcified left lower lobe granuloma. The lungs are otherwise clear. Keyword : Cicatrix Calcified Granuloma")
    print(
        "Example 1 OUTPUT: "
        + " "+model.example_images(test_oimg1.to(device),test_img1.to(device), start_token.to(device),device))
    test_img2 = transform(
        Image.open("D:/data/iuct/preprocess/images/fourier/2_IM-0652-1001.dcm.png.jpg").convert("L")).unsqueeze(0)
    test_oimg2 = transform(
        Image.open("D:/data/iuct/origin/images_normalized/2_IM-0652-1001.dcm.png").convert("L")).unsqueeze(0)
    print("Example 2 CORRECT: Borderline cardiomegaly. Midline sternotomy XXXX. Enlarged pulmonary arteries. Clear lungs. Inferior XXXX XXXX XXXX.")
    print(
        "Example 2 OUTPUT: "
        + " "+model.example_images(test_oimg2.to(device),test_img2.to(device), start_token.to(device),device))
    test_img3 = transform(Image.open("D:/data/iuct/preprocess/images/fourier/3479_IM-1690-2001.dcm.png.jpg").convert("L")).unsqueeze(0)
    test_oimg3 = transform(
        Image.open("D:/data/iuct/origin/images_normalized/3479_IM-1690-2001.dcm.png").convert("L")).unsqueeze(0)
    print("Example 3 CORRECT: TSoft tissue neck. The airway is XXXX. No laryngeal edema. Laryngeal XXXX intact. Cervical spine intact. Chest. The heart is large. Diffuse parahilar and alveolar consolidations are present. Bilateral costophrenic XXXX blunting is present. Keyword: Cardiomegaly Consolidation Consolidation Costophrenic Angle Heart Failure Pulmonary Edema Pleural Effusion")
    print(
        "Example 3 OUTPUT: "
        + " "+model.example_images(test_oimg3.to(device),test_img3.to(device), start_token.to(device),device))
    test_img4 = transform(Image.open("D:/data/iuct/preprocess/images/fourier/3668_IM-1825-2001.dcm.png.jpg")).unsqueeze(0)
    test_oimg4 = transform(
        Image.open("D:/data/iuct/origin/images_normalized/3668_IM-1825-2001.dcm.png").convert("L")).unsqueeze(0)
    print("Example 4 CORRECT:keyword: Infiltrate")
    print("Example 4 OUTPUT: "+ " "+model.example_images(test_oimg4.to(device),test_img4.to(device), start_token.to(device),device))
    model.train()

def save_checkpoint(state, filename=f"D:\github\ReportGeneration\weights\my_checkpoint_{time.strftime('%Y%m%d', now)}.pth"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    step = checkpoint["step"]
    return step

def save_metrics(state,filename=f"my_train_metrics_{time.strftime('%Y%m%d', now)}.pt"):
    print("=> Saving metrics")
    torch.save(state, filename)



