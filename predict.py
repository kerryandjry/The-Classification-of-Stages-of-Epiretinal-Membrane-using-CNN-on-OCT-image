import os
import json
from pathlib import Path
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from models.vit_model import vit_base_patch16_224_in21k as create_model
# from swin_model import swin_base_patch4_window7_224_in22k
# from models.resnet import resnet34
from models.efficient import efficientnetv2_s
from models.mobilenet import mobilenet_v3_small
from models import mlp


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # model = swin_base_patch4_window7_224_in22k(num_classes=4, has_logits=False).to(device)
    # model = mlp.linear_base().to(device)
    model = mobilenet_v3_small(num_classes=4).to(device)

    model_weight_path = "weights/mobile/mob_1.pth"
    model.load_state_dict(torch.load(model_weight_path, map_location=device))

    data_transform = transforms.Compose([transforms.Resize([224, 375]),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                                         ])
    path = Path("./test")
    average_acc = 0
    for i in path.iterdir():
        all_image, class_num = 0, 0
        num = str(i)[-1:]

        for img in Path(i).iterdir():
            all_image += 1
            img = Image.open(img)
            plt.imshow(img)

            img = data_transform(img)

            img = torch.unsqueeze(img, dim=0)

            json_path = './class_indices.json'
            assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

            json_file = open(json_path, "r")
            class_indict = json.load(json_file)

            model.eval()
            with torch.no_grad():

                output = torch.squeeze(model(img.to(device)))
                predict = torch.softmax(output.cpu(), dim=0)
                predict_cla = torch.argmax(predict).numpy()

            print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)],
                                                         predict[predict_cla].numpy())

            if class_indict[str(predict_cla)] == num:
                class_num += 1

        average = class_num / all_image
        average_acc += average
        print(f"class {num} = {class_num}, all images = {all_image}")
        print(f'{average}%')

    print(f'average of all = {average_acc / 4}')


if __name__ == '__main__':
    main()
