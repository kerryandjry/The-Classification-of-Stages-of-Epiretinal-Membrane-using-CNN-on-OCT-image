import os
import json
from pathlib import Path
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from models.vit_model import vit_base_patch16_224_in21k as create_model
# from swin_model import swin_base_patch4_window7_224_in22k
from models.mlp import MLPMixer


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # model = swin_base_patch4_window7_224_in22k(num_classes=4, has_logits=False).to(device)
    # model = mobilenet_v3_small(num_classes=4).to(device)
    model = MLPMixer(
        image_size=224,
        channels=3,
        patch_size=16,
        dim=448,
        depth=1,
        num_classes=4,
        dropout=0.3
    ).to(device)

    model_weight_path = "./weights/mlp.pth"
    model.load_state_dict(torch.load(model_weight_path, map_location=device))

    data_transform = transforms.Compose([transforms.Resize([224, 375]),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                                         ])

    folder_path = "./test/4"
    all_image, class_1, class_2, class_3, class_4 = 0, 0, 0, 0, 0

    for img in Path(folder_path).iterdir():
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

        if class_indict[str(predict_cla)] == '1':
            class_1 += 1
        if class_indict[str(predict_cla)] == '2':
            class_2 += 1
        if class_indict[str(predict_cla)] == '3':
            class_3 += 1
            # plt.title(print_res)
            # plt.show()
        if class_indict[str(predict_cla)] == '4':
            class_4 += 1

    print(f"class 1 = {class_1}, class 2 = {class_2}, class 3 = {class_3}, class 4 = {class_4}")
    print(f'{class_4 / all_image}%')


if __name__ == '__main__':
    main()
