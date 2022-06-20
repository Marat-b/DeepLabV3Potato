import itertools
import json
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Callable, List, Optional
import numpy as np
import pycocotools.mask as mask_util
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from dataset.register_instances import instances, register_dataset_instances
from utilz.cv2_imshow import cv2_imshow


def _isArrayLike(obj):
    return hasattr(obj, '__iter__') and hasattr(obj, '__len__')


class PotatoDataset:
    def __init__(self, name_instances=[], new_shape=(512, 512), transforms_image: Optional[Callable] = None,
                 transforms_mask: Optional[Callable] = None):
        """
        Constructor of Microsoft COCO helper class for reading and visualizing annotations.
        :param annotation_file (str): location of annotation file
        :param image_folder (str): location to the folder that hosts images.
        :return:
        """
        # load dataset
        self.sample = {'image': [], 'mask': []}
        self.sub_sample = self.sample.copy()
        self.dataset, self.imgs = dict(), dict()
        self.img_to_segments, self.cat_ids = defaultdict(list), defaultdict(list)
        self.images_path = None
        self.new_shape = new_shape
        self.name_instances = [name for name in instances.keys() if name in name_instances]
        self.transforms_image = transforms_image
        self.transforms_mask = transforms_mask
        # print(self.name_instances)
        self.get_sample()

    def __getitem__(self, index):
        if self.transforms_image and self.transforms_mask:
            self.sub_sample["image"] = self.transforms_image(self.sample["image"][index])
            self.sub_sample["mask"] = self.transforms_mask(self.sample["mask"][index])
        else:
            self.sub_sample['image'] = self.sample['image'][index]
            self.sub_sample['mask'] = self.sample['mask'][index]
        return self.sub_sample

    def __len__(self) -> int:
        return len(self.sample['image'])

    def create_index(self) -> None:
        """
        create index from annotation's file
        :return: None
        :rtype:
        """
        imgs = {}
        img_to_segments, cat_ids = defaultdict(list), defaultdict(list)
        if 'images' in self.dataset:
            for img in self.dataset['images']:
                imgs[img['id']] = img

        if 'annotations' in self.dataset:
            for ann in self.dataset['annotations']:
                img_to_segments[ann['image_id']].append(
                    {
                        'segmentation': ann['segmentation'],
                        'category_id': ann['category_id']
                    }
                )
        if 'categories' in self.dataset:
            cat_ids = [cat['id'] for cat in self.dataset['categories']]
        self.imgs = imgs
        self.img_to_segments = img_to_segments
        self.cat_ids = cat_ids

    def _polygons_to_bitmask(self, polygons: List[np.ndarray], height: int, width: int) -> np.ndarray:
        """
        Args:
            polygons (list[ndarray]): each array has shape (Nx2,)
            height, width (int)

        Returns:
            ndarray: a bool mask of shape (height, width)
        """
        if len(polygons) == 0:
            # COCOAPI does not support empty polygons
            return np.zeros((height, width)).astype(np.bool_)
        rles = mask_util.frPyObjects(polygons, height, width)
        rle = mask_util.merge(rles)
        return mask_util.decode(rle).astype(np.bool_)

    def get_image(self, img_key: int) -> Optional[np.ndarray]:
        """
        Get image from annotation file
        :param img_key:
        :type img_key: int
        :return:
        :rtype:
        """
        file_name = self.imgs[img_key]['file_name']
        if Path(os.path.join(self.images_path, file_name)).exists():
            image = Image.open(os.path.join(self.images_path, file_name))
            image = np.asarray(self._scale(np.asarray(image), self.new_shape))
            # shape = image.shape
            # print(f'shape={shape}')
            # image = np.transpose(image, (2, 0, 1))
            return image
        else:
            print(f' Path {os.path.join(self.images_path, file_name)} does not exists')
            return None

    def get_mask(self, sample) -> None:
        """
        Get mask from indices of polynom.
        Mask layers are amount of classes
        :param sample:
        :type sample:
        :return:
        :rtype:
        """
        for img_key in self.imgs.keys():
            # bitmasks = np.zeros((len(pd.cat_ids), self.imgs[img_key]['height'], self.imgs[img_key]['width']))
            image = self.get_image(img_key)
            if image is None:
                continue
            # print(f'image.shape={image.shape}')
            # cv2_imshow(image[0])
            # cv2_imshow(image[1])
            # cv2_imshow(image[2])
            # + 1 for background class
            bitmasks = np.zeros((len(self.cat_ids) + 1, self.new_shape[0], self.new_shape[1]), dtype='bool')
            # print(f'empty bitmasks.shape={bitmasks.shape}')
            for img_to_segment in self.img_to_segments[img_key]:
                for cat in self.cat_ids:
                    if img_to_segment['category_id'] == cat:
                        # print(f'img_to_segment={img_to_segment}')
                        # print(f'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
                        # print(f'cat={cat}')
                        bitmask = self._polygons_to_bitmask(
                            img_to_segment['segmentation'],
                            self.imgs[img_key]['height'], self.imgs[img_key]['width']
                        )
                        # print(f'max old bitmask={np.max(bitmask)}')
                        bitmasks[cat] += self._scale(bitmask, self.new_shape)
                        # print(f'max new bitmask={np.max(self._scale(bitmask, new_shape[0], new_shape[1]))}')
                        # print(f'bitmask.shape={bitmask.shape}')
                        # print(f'max bitmasks={np.max(bitmasks)}')
            bitmasks = np.transpose(bitmasks, (1, 2, 0))
            # print(f'full bitmasks.shape={bitmasks.shape}')
            sample['image'].append(image)
            sample['mask'].append(bitmasks.astype('float'))
            # print(f"image={(image / 255).dtype}")
            # print(f"bitmasks={bitmasks.astype('float').dtype}")

        # print('sample is loaded...')
        self.sample = sample
        # return sample

    def get_sample(self) -> None:
        """
        Get dictionary - {'image': [], 'mask': []}
        :return:
        :rtype:
        """
        print('Start to get instances...')
        for name_inst in tqdm(self.name_instances):
            # tic = time.time()
            self.images_path = instances[name_inst][1]
            if Path(instances[name_inst][0]).exists():
                dataset = json.load(open(instances[name_inst][0], 'r'))
                assert type(dataset) == dict, 'annotation file format {} not supported'.format(type(dataset))
                # print('Done (t={:0.2f}s)'.format(time.time() - tic))
                self.dataset = dataset
                self.create_index()
                self.get_mask(self.sample)
            # else:
            #     raise OSError(f"{instances[name_inst][0]} does not exist.")
        print('End to get instances...')
        # print(f'masks.shape={np.array(self.sample["mask"]).shape}')
        # print(f'image.shape={np.array(self.sample["image"]).shape}')

    def _scale(self, im, n_shape):
        """
        n_shape[0] = n_rows, n_shape[1] = n_columns
        :param im:
        :type im:
        :return:
        :rtype:
        """
        n_rows0 = len(im)  # source number of rows
        n_columns0 = len(im[0])  # source number of columns
        return [[im[int(n_rows0 * r / n_shape[0])][int(n_columns0 * c / n_shape[1])]
                 for c in range(n_shape[1])] for r in range(n_shape[0])]


if __name__ == '__main__':
    register_dataset_instances('set6', '../datasets/potato_set6_coco.json', '../datasets/set6')
    register_dataset_instances('set15', '../datasets/potato_set15_coco.json', '../datasets/set15')
    register_dataset_instances('set16', '../datasets/potato_set16_coco.json', '../datasets/set16')
    data_transforms_image = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(
             mean=[0.485, 0.456, 0.406],
             std=[0.229, 0.224, 0.225]
         )
         ]
    )
    data_transforms_mask = transforms.Compose(
        [transforms.ToTensor()
         ]
    )
    pd = PotatoDataset(['set16'], transforms_image=data_transforms_image, transforms_mask=data_transforms_mask)  #
    # 'set6', 'set15',
    # sample = {'image': [], 'mask': []}
    # sample = pd.get_mask(sample)

    # pd2 = PotatoDataset(
    #     '../datasets/potato_set6_coco.json',
    #     images_path='../datasets/set6'
    #     )
    # sample = pd2.get_mask(sample)
    print(f'len pd={len(pd)}')
    # print(f'pd.img_to_anns={pd.img_to_anns}')
    print(f'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print(f'pd.imgs={pd.imgs}')
    print(f'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    # print(f'pd.cats={pd.cats}')
    # print(f'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    # print(f'pd.cat_to_imgs={pd.cat_to_imgs}')

    for i in range(len(pd)):
        print(pd[i])

    # print(f'pd.imgs.keys()={pd.imgs.keys()}')
    # print(f'cat_ids={pd.cat_ids}')
    # print(f'masks.shape={np.array(sample["mask"]).shape}')
    # print(f'image.shape={np.array(sample["image"]).shape}')

    # masks[masks == 1] = 255
    # print(np.array(masks)[0].shape)
    # cv2_imshow(bitmasks[1].reshape((height, width, 1)), 'bitmasks[1]')
    # cv2_imshow(np.array(sample["mask"])[0][0].reshape((512, 512, 1)), 'masks[0][0]')
    # cv2_imshow(np.array(sample["mask"])[0][1].reshape((512, 512, 1)), 'masks[0][1]')
    # cv2_imshow(np.array(sample["mask"])[0][2].reshape((512, 512, 1)), 'masks[0][2]')
    # print(f'max masks={np.max(sample["mask"])}')
