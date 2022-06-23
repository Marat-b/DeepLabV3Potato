from multiprocessing import freeze_support

import torch
from PIL import Image
import numpy as np
from sklearn.metrics import confusion_matrix
from torchvision import transforms
from tqdm import tqdm
import cv2
from dataset import datahandler
from dataset.potato_dataset import PotatoDataset

from utilz.cv2_imshow import cv2_imshow


def show_color(output_tensor, frame):
    # frame1 = frame[:, :, [2, 1, 0]]
    output_colors = cv2.applyColorMap(output_tensor, cv2.COLORMAP_JET)
    print(f'output_colors.shape={output_colors.shape}')
    print(f'output_tensor.shape={output_tensor.shape}')
    # output_colors[:, :, output_tensor == 0] = [0]
    # output_colors[output_tensor == 0] = [0, 0, 0]
    return cv2.addWeighted(frame, 1, output_colors, 0.4, 0)


def main():
    # model = torch.load('../weights/potato_20220617_10x.pth').eval()
    model = torch.load('./weights/potato_20220620_norm_2.pth').eval()
    image = Image.open('images/image_256.jpg')
    print(image.getbands())
    print(f'image shape={np.asarray(image).transpose((2, 0, 1)).shape}')

    register_dataset_instances('set16', 'datasets/potato_set16_coco.json', './datasets/set16')
    register_dataset_instances('set37', 'datasets/potato_set37_coco.json', './datasets/set37')
    dataloaders = datahandler.get_dataloader(train_instances=['set37'], test_instances=['set37'])
    y_pred_list = []
    y_true_list = []
    frame = None
    test_loader = dataloaders["Test"]
    print(f'dataloaders={test_loader}')
    with torch.no_grad():
        for sample in tqdm(iter(test_loader)):
            x_batch = sample['image']
            y_batch = sample['mask']
            # print(f'\nx_batch.shape={x_batch.shape}')
            # cv2_imshow(x_batch.numpy().squeeze().transpose(1, 2, 0))
            # res = model(torch.as_tensor(np.asarray(image).transpose(2, 0, 1).astype('float32')))

            y_test_pred = model(x_batch)
            y_test_pred = y_test_pred["out"]
            # print(f'\ny_test_pred.shape={y_test_pred.shape}')
            # print(y_test_pred.numpy().squeeze()[1].shape)
            # for i in range(4):
            #     cv2_imshow((y_test_pred.numpy().squeeze())[i], 'y_test_pred[{}]'.format(i))
            #     print(f'max y_test_pred[{i}]={torch.max(y_test_pred[0][i])}')
            _, y_pred_tag = torch.max(y_test_pred, dim=1)
            y_pred_list.append(y_pred_tag.cpu().numpy())
            y_true_list.append(y_batch.cpu().numpy())
            frame = x_batch
    print(f'y_pred_list.shape={np.asarray(y_pred_list).shape}\ny_true_list.shape={np.asarray(y_true_list).shape}')
    # print(f'y_pred_list={np.asarray(y_pred_list)}')
    # print(f'y_pred_list unique={np.unique(np.asarray(y_pred_list))}')
    y_pred_0 = (np.asarray(y_pred_list[0])).transpose((1, 2, 0))  # .astype('uint8')
    cv2_imshow(y_pred_0, 'y_pred_0')
    print(f'y_pred_0 unique={np.unique(np.asarray(y_pred_0))}')
    print(f'y_pred_0 mean={np.mean(np.asarray(y_pred_0))}')
    # y_pred_1 = (np.asarray(y_pred_list[1])).transpose((1, 2, 0))  # .astype('uint8')
    # cv2_imshow(y_pred_1, 'y_pred_1')
    # print(f'y_pred_1 unique={np.unique(np.asarray(y_pred_1))}')
    # print(f'y_pred_1 mean={np.mean(np.asarray(y_pred_1))}')
    # y_pred_list = [i[0][0][0] for i in y_pred_list]
    # y_true_list = [i[0] for i in y_true_list]
    # print(confusion_matrix(y_true_list, y_pred_list))
    frame = frame.numpy().squeeze().transpose((1, 2, 0)).astype('uint8')
    cv2_imshow(frame, 'frame')
    y_pred_last = y_pred_list[len(y_pred_list) - 1].transpose((1, 2, 0)).astype('uint8')
    cv2_imshow(y_pred_last, 'y_pred_last')
    print(f'y_pred_last.shape={y_pred_last.shape}, type={y_pred_last.dtype}')
    print(f'frame.shape={frame.shape}, type={frame.dtype}')
    colored_frame = show_color(y_pred_last, frame)
    cv2_imshow(colored_frame, 'colored_frame')


def main2():
    # model = torch.load('./weights/potato_model.pth').eval()
    # model = torch.load('./weights/potato_20220617_10x.pth').eval()
    model = torch.load('./weights/potato_20220623_1x.pth', map_location=torch.device('cpu')).eval()
    # image = Image.open('datasets/set37/00000001.jpg')
    image = Image.open('datasets/set6/Image_1.jpg')
    print(image.getbands())
    print(f'image shape={np.asarray(image).transpose((2, 0, 1)).shape}')

    y_pred_list = []
    y_true_list = []
    frame = None
    transform = transforms.Compose(
        [
            transforms.PILToTensor(),
            transforms.Resize(size=(256, 256))
        ]
    )
    # test_loader = dataloaders["Test"]
    # print(f'dataloaders={test_loader}')
    with torch.no_grad():
        # print(f'\nx_batch.shape={x_batch.shape}')
        # cv2_imshow(x_batch.numpy().squeeze().transpose(1, 2, 0))
        # res = model(torch.as_tensor(np.asarray(image).transpose(2, 0, 1).astype('float32')))
        image_t = transform(image)
        print(f'image_t.shape={image_t.shape}')
        # image_tensor = torch.as_tensor(np.asarray(image).transpose((2, 0, 1)).astype('float32')).unsqueeze(dim=0)
        image_tensor = transform(image).unsqueeze(dim=0).type(torch.float32)
        print(f'image_tensor.shape={image_tensor.shape}')
        y_test_pred = model(image_tensor)
        y_test_pred = y_test_pred["out"]
        print(f'\ny_test_pred.shape={y_test_pred.shape}')
        for i in range(4):
            cv2_imshow((y_test_pred.numpy().squeeze())[i], 'y_test_pred[{}]'.format(i))
            print(f'max y_test_pred[{i}]={torch.max(y_test_pred[0][i])}')
        _, y_pred_tag = torch.max(y_test_pred, dim=1)
        print(f'y_pred_tag.shape={y_pred_tag.shape}')
        y_pred_list.append(y_pred_tag.cpu().numpy())
        # y_true_list.append(y_batch.cpu().numpy())

    print(f'y_pred_list.shape={np.asarray(y_pred_list).shape}')
    # print(f'y_pred_list={np.asarray(y_pred_list)}')
    # print(f'y_pred_list unique={np.unique(np.asarray(y_pred_list))}')
    y_pred_0 = (np.asarray(y_pred_list[0])).transpose((1, 2, 0))  # .astype('uint8')
    cv2_imshow(y_pred_0, 'y_pred_0')
    print(f'y_pred_0 unique={np.unique(np.asarray(y_pred_0))}')
    print(f'y_pred_0 mean={np.mean(np.asarray(y_pred_0))}')

    # y_pred_list = [i[0][0][0] for i in y_pred_list]
    # y_true_list = [i[0] for i in y_true_list]
    # print(confusion_matrix(y_true_list, y_pred_list))

    cv2_imshow(image, 'image')

    # colored_frame = show_color(y_pred_0.astype('uint8'), np.asarray(image))
    # cv2_imshow(colored_frame, 'colored_frame')


if __name__ == '__main__':
    freeze_support()
    main2()
