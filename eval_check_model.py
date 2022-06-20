from multiprocessing import freeze_support

import torch
from PIL import Image
import numpy as np
from sklearn.metrics import confusion_matrix
from torchvision import transforms
from tqdm import tqdm

from dataset import datahandler
from dataset.potato_dataset import PotatoDataset
from dataset.register_instances import register_dataset_instances
from utilz.cv2_imshow import cv2_imshow


def main():
    # model = torch.load('../weights/potato_20220617_10x.pth').eval()
    model = torch.load('../weights/potato_20220620_norm_2.pth').eval()
    image = Image.open('images/image_256.jpg')
    print(image.getbands())
    print(f'image shape={np.asarray(image).transpose((2, 0, 1)).shape}')

    register_dataset_instances('set16', 'datasets/potato_set16_coco.json', '../datasets/set16')
    dataloaders = datahandler.get_dataloader(train_instances=['set16'], test_instances=['set16'])
    y_pred_list = []
    y_true_list = []
    test_loader = dataloaders["Test"]
    print(f'dataloaders={test_loader}')
    with torch.no_grad():
        for sample in tqdm(iter(test_loader)):
            x_batch = sample['image']
            y_batch = sample['mask']
            print(f'\nx_batch.shape={x_batch.shape}')
            cv2_imshow(x_batch.numpy().squeeze().transpose(1, 2, 0))
            # res = model(torch.as_tensor(np.asarray(image).transpose(2, 0, 1).astype('float32')))

            y_test_pred = model(x_batch)
            y_test_pred = y_test_pred["out"]
            print(f'\ny_test_pred.shape={y_test_pred.shape}')
            print(y_test_pred.numpy().squeeze()[1].shape)
            for i in range(4):
                cv2_imshow((y_test_pred.numpy().squeeze())[i].astype('uint8'), 'y_test_pred[{}]'.format(i))
                print(f'max y_test_pred[{i}]={torch.max(y_test_pred[0][i])}')
            # cv2_imshow((y_test_pred.numpy().squeeze())[1], 'y_test_pred[1]')
            # cv2_imshow((y_test_pred.numpy().squeeze())[2], 'y_test_pred[2]')
            # cv2_imshow((y_test_pred.numpy().squeeze())[3], 'y_test_pred[3]')
            _, y_pred_tag = torch.max(y_test_pred, dim=1)
            y_pred_list.append(y_pred_tag.cpu().numpy())
            y_true_list.append(y_batch.cpu().numpy())
    print(f'y_pred_list.shape={np.asarray(y_pred_list).shape}\ny_true_list.shape={np.asarray(y_true_list).shape}')
    print(f'y_pred_list={np.asarray(y_pred_list)}')
    print(f'y_pred_list unique={np.unique(np.asarray(y_pred_list))}')
    y_pred = (np.asarray(y_pred_list[0])).transpose((1, 2, 0)).astype('uint8')
    cv2_imshow(y_pred, 'y_pred')
    # y_pred_list = [i[0][0][0] for i in y_pred_list]
    # y_true_list = [i[0] for i in y_true_list]
    # print(confusion_matrix(y_true_list, y_pred_list))


if __name__ == '__main__':
    freeze_support()
    main()
