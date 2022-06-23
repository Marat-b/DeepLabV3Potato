import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Callable, List, Optional
import numpy as np
import pycocotools.mask as mask_util
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from pathlib import Path


class PotatoSample:
    def __init__(self, data_instances=[], new_shape=(512, 512), transforms_image: Optional[Callable] = None,
                 transforms_mask: Optional[Callable] = None):
        # if data_instances is None:
        #     data_instances = []
        # print(f'init data_instances={data_instances}, len={len(data_instances)}')
        self.cat_ids = 0
        self.count_images = 0
        self.data_instances = data_instances
        self.dataset = dict()
        self.img_segments, self.cat_ids = defaultdict(list), defaultdict(list)
        self.new_shape = new_shape
        self.sample = {'image': [], 'mask': []}
        self.sub_sample = self.sample.copy()
        self.transforms_image = transforms_image
        self.transforms_mask = transforms_mask
        self.create_dataset()

    def __getitem__(self, index):
        sub_sample = {}
        smpl = self.get_sample(index)
        # print(f'self.get_sample(index)["image"].shape={smpl["image"].shape}')
        # print(f'self.get_sample(index)["mask"].shape={smpl["mask"].shape}')
        # smpl['image'].shape = (h, w ,c) type - ndarray
        sub_sample["image"] = self.transforms_image(smpl["image"])
        sub_sample["mask"] = self.transforms_mask(smpl["mask"])
        # print(f'self.sub_sample(index)["image"]={sub_sample["image"].shape}')
        # print(f'self.sub_sample(mask)["image"]={sub_sample["mask"].shape}')
        # sub_sample['image'].shape = (c, h, w)
        return sub_sample

    def __len__(self):
        # print(f'len(self.img_segments)={len(self.img_segments)}')
        return len(self.img_segments)

    def create_dataset(self):
        for data_instance in tqdm(self.data_instances):
            # tic = time.time()
            self.images_path = data_instance[1]
            if Path(data_instance[0]).exists():
                # print(f'Load dataset:{data_instance[0]}')
                dataset = json.load(open(data_instance[0], 'r'))
                assert type(dataset) == dict, 'annotation file format {} not supported'.format(type(dataset))
                # print('Done (t={:0.2f}s)'.format(time.time() - tic))
                self.dataset = dataset
                self.create_index(self.images_path)
                # self.get_mask(self.sample)
            else:
                raise OSError(f"{data_instance[0]} does not exist.")
        # print('End to get instances...')

    def create_index(self, path: str) -> None:
        """
        create index from annotation's file
        :return: None
        :rtype:
        """

        def get_annotation(index: int):
            img_to_segments = []
            if 'annotations' in self.dataset:
                for ann in self.dataset['annotations']:
                    if ann['image_id'] == index:
                        img_to_segments.append(
                            {
                                'segmentation': ann['segmentation'],
                                'category_id': ann['category_id']
                            }
                        )
            return img_to_segments

        imgs = {}

        if 'images' in self.dataset:
            for img in self.dataset['images']:
                imgs[img['id']] = img

        for img in imgs.keys():
            # print(f'--> img={img}')
            self.img_segments[self.count_images].append(
                {
                    'path': path,
                    'file_name': imgs[img]['file_name'],
                    'height': imgs[img]['height'],
                    'width': imgs[img]['width'],
                    'annotations': get_annotation(imgs[img]['id'])
                }
            )
            self.count_images += 1
        if 'categories' in self.dataset:
            cat_ids = [cat['id'] for cat in self.dataset['categories']]
        # print(f'imgs={imgs}')
        # print(f'img_segments={self.img_segments}')
        # print(f'cat_ids={cat_ids}')
        # self.imgs = imgs
        # self.img_to_segments = img_to_segments
        self.cat_ids = cat_ids

    def get_image(self, images_path, file_name):
        if Path(os.path.join(images_path, file_name)).exists():
            image = Image.open(os.path.join(images_path, file_name))
            image = np.asarray(self._scale(np.asarray(image), self.new_shape))
            # shape = image.shape
            # print(f'get_image image.shape={shape}')
            # image = np.transpose(image, (2, 0, 1))
            return image
        else:
            print(f' Path {os.path.join(self.images_path, file_name)} does not exists')
            return None

    def get_mask(self, img_segment):
        sample = {}
        img_segment = img_segment[0]
        # print(f'img_segment={img_segment}')
        image = self.get_image(img_segment['path'], img_segment['file_name'])
        if image is not None:
            bitmasks = np.zeros((len(self.cat_ids) + 1, self.new_shape[0], self.new_shape[1]), dtype='bool')
            # print(f'empty bitmasks.shape={bitmasks.shape}')
            for img_to_segment in img_segment['annotations']:
                for cat in self.cat_ids:
                    if img_to_segment['category_id'] == cat:
                        # print(f'img_to_segment={img_to_segment}')
                        # print(f'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
                        # print(f'cat={cat}')
                        bitmask = self._polygons_to_bitmask(
                            img_to_segment['segmentation'],
                            img_segment['height'], img_segment['width']
                        )
                        # print(f'max old bitmask={np.max(bitmask)}')
                        bitmasks[cat] += self._scale(bitmask, self.new_shape)
                        # print(f'max new bitmask={np.max(self._scale(bitmask, new_shape[0], new_shape[1]))}')
                        # print(f'bitmask.shape={bitmask.shape}')
                        # print(f'max bitmasks={np.max(bitmasks)}')
            bitmasks = np.transpose(bitmasks, (1, 2, 0))
            # print(f'full bitmasks.shape={bitmasks.shape}')
            sample['image'] = image
            sample['mask'] = bitmasks.astype('float')
            # print(f'self.sample={self.sample}')
            # print(f"image={(image / 255).dtype}")
            # print(f"bitmasks={bitmasks.astype('float').dtype}")
            # print(f"image.shape={np.asarray(sample['image']).shape}")
            # print(f"mask.shape={np.asarray(sample['mask']).shape}")

            # print('sample is loaded...')
        # self.sample = sample
        return sample

    def get_sample(self, index):
        img_segment = self.img_segments[index]
        # print(f'img_segment={img_segment}')
        sample = self.get_mask(img_segment)
        return sample

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


if __name__ == "__main__":
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
    ps = PotatoSample([
        # [['../datasets/potato_set15_coco.json', '../datasets/set15'],
        ('../datasets/potato_set16_coco.json', '../datasets/set16')],
        transforms_image=data_transforms_image,
        transforms_mask=data_transforms_mask
        )
    # ps.create_dataset()
    print(f'len(ps)={len(ps)}')
    # ps.get_sample(0)
    for i in range(len(ps)):
        print(f'i={i}')
        s = ps[i]
