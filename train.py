import os
import math
import argparse

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from my_dataset import MyDataSet
from utils import read_split_data, train_one_epoch, evaluate
from models.resnet import resnet34
from models.efficient import efficientnetv2_s
from models.mobilenet import mobilenet_v3_small
from models import mlp
from models import vit_model
from models import swin_model


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if os.path.exists("./new_weights") is False:
        os.makedirs("./new_weights")

    tb_writer = SummaryWriter()

    train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(args.data_path)

    data_transform = {
        "train": transforms.Compose([
                                     transforms.Resize([224, 375]),
                                     transforms.CenterCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                                     ]),

        "val": transforms.Compose([
                                   transforms.Resize([224, 375]),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                                   ]),
    }

    train_dataset = MyDataSet(images_path=train_images_path,
                              images_class=train_images_label,
                              transform=data_transform["train"])

    val_dataset = MyDataSet(images_path=val_images_path,
                            images_class=val_images_label,
                            transform=data_transform["val"])

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=nw,
                                               collate_fn=train_dataset.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw,
                                             collate_fn=val_dataset.collate_fn)

    # model = vit_model.vit_base_patch16_224_in21k(num_classes=args.num_classes, has_logits=False).to(device)
    model = resnet34(num_classes=args.num_classes).to(device)
    # model = (num_classes=4).to(device)

    if args.weights != "":
        assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
        # weights_dict = torch.load(args.weights, map_location=device)['model']
        weights_dict = torch.load(args.weights, map_location=device)

        for k in list(weights_dict.keys()):
            if "head" in k:
                print(f"del {k}")
                del weights_dict[k]
        print(model.load_state_dict(weights_dict, strict=False))

    if args.freeze_layers:
        for name, para in model.named_parameters():
            if "head" not in name and "pre_logits" not in name:
            # if "head" not in name:
                para.requires_grad_(False)
            else:
                print(f"training {name}")

    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(params=pg, lr=args.lr, momentum=0.9, weight_decay=5E-4)
    # optimizer = optim.Adam(params=pg, lr=args.lr, weight_decay=5E-2)

    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    best_weight = 0

    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(model=model,
                                                optimizer=optimizer,
                                                data_loader=train_loader,
                                                device=device,
                                                epoch=epoch)

        scheduler.step()

        val_loss, val_acc = evaluate(model=model,
                                     data_loader=val_loader,
                                     device=device,
                                     epoch=epoch)

        tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        tb_writer.add_scalar(tags[1], train_acc, epoch)
        tb_writer.add_scalar(tags[2], val_loss, epoch)
        tb_writer.add_scalar(tags[3], val_acc, epoch)
        tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)

        metric = val_acc
        if metric > best_weight:
            torch.save(model.state_dict(), f"./new_weights_che/res.pth")
            best_weight = metric
            print(f"metric = {metric}, save model")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lrf', type=float, default=0.01)

    parser.add_argument('--data-path', type=str,
                        default="/home/lee/Work/data/new_oct_4fold_che/")
    parser.add_argument('--model-name', default='', help='create model name')

    # parser.add_argument('--weights', type=str, default='./weights/linear_tiny_checkpoint.pth',
    #                    help='initial weights path')
    parser.add_argument('--weights', type=str, default='',
                        help='initial weights path')

    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda:0', help='device id')

    opt = parser.parse_args()
    main(opt)

    # for val in range(1, 5):
    #     print(f'dataset {val} as val date, others training')
    #     main(opt, val)
