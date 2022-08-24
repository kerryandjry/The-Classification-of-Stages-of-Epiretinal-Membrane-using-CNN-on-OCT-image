from pathlib import Path
import cv2
from torchvision import transforms


transform = transforms.Compose([
                                     transforms.Resize([224, 375]),
                                     # transforms.RandomResizedCrop(224),
                                     transforms.CenterCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                                ])

path = "/home/lee/Work/data/oct_temp/test/4"
name = 0
for image in Path(path).iterdir():
    print(image)
    img = cv2.imread(str(image))
    img = img[:435, 500:]
    cv2.imwrite(f'/home/lee/Work/data/oct_data/test/4/{name}.jpg', img)
    name += 1
    # cv2.imshow("2", img)
    # cv2.waitKey(500)
    # cv2.destroyAllWindows()

# img = Image.open(path, mode='r')
# plt.imshow(img, cmap='gray')
# plt.show()

# for _ in Path(path).iterdir():
#     img = Image.open(_, mode='r')
#     img = transform(img)
#     img = img.swapaxes(0, 1)
#     img = img.swapaxes(1, 2)
#     plt.imshow(img)
#     plt.show()
