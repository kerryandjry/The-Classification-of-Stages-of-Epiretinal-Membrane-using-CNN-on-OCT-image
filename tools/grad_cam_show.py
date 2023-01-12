import os
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import models
from torchvision import transforms
from models import resnet
from grad_cam import GradCAM, show_cam_on_image, center_crop_img
from pathlib import Path


def main():
    model = resnet.resnet34(num_classes=4)
    # target_layers = [model.features[-1]]
    model_weight_path = "../weights/resnet34/res_4.pth"
    model.load_state_dict(torch.load(model_weight_path, map_location="cuda"))

    target_layers = [model.layer4]

    data_transform = transforms.Compose([transforms.Resize([224, 375]),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                                         ])
    # load image
    path = Path("/test/class4")
    for image in path.iterdir():
        assert os.path.exists(image), "file: '{}' dose not exist.".format(image)
        img = Image.open(image).convert('RGB')

        # [C, H, W]
        img_tensor = data_transform(img)
        # expand batch dimension
        # [C, H, W] -> [N, C, H, W]
        input_tensor = torch.unsqueeze(img_tensor, dim=0)

        cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False)
        target_category = 3 # which class -1
        # target_category -= 1
        grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)

        grayscale_cam = grayscale_cam[0, :]
        img = img.resize((224, 224))
        img = np.array(img, dtype=np.uint8)
        visualization = show_cam_on_image(img.astype(dtype=np.float32) / 255.,
                                          grayscale_cam,
                                          use_rgb=True)
        plt.imshow(visualization)
        plt.axis('off')
        plt.show()


if __name__ == '__main__':
    main()