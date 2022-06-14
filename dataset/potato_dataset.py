import itertools
import json
import time
from collections import defaultdict
from typing import List
import numpy as np
import pycocotools.mask as mask_util

from utilz.cv2_imshow import cv2_imshow


def _isArrayLike(obj):
    return hasattr(obj, '__iter__') and hasattr(obj, '__len__')


class PotatoDataset:
    def __init__(self, annotation_file=None):
        """
        Constructor of Microsoft COCO helper class for reading and visualizing annotations.
        :param annotation_file (str): location of annotation file
        :param image_folder (str): location to the folder that hosts images.
        :return:
        """
        # load dataset
        self.dataset, self.anns, self.cats, self.imgs = dict(), dict(), dict(), dict()
        self.img_to_segments, self.cat_ids = \
            defaultdict(list), defaultdict(list)
        if annotation_file is not None:
            print('loading annotations into memory...')
            tic = time.time()
            dataset = json.load(open(annotation_file, 'r'))
            assert type(dataset) == dict, 'annotation file format {} not supported'.format(type(dataset))
            print('Done (t={:0.2f}s)'.format(time.time() - tic))
            self.dataset = dataset
            self.create_index()

    def create_index(self):
        # create index
        print('creating index...')
        anns, cats, imgs = {}, {}, {}
        img_to_segments, cat_ids = defaultdict(list), defaultdict(list)

        if 'annotations' in self.dataset:
            for ann in self.dataset['annotations']:
                # img_to_anns[ann['image_id']].append(ann)
                anns[ann['id']] = ann

        if 'images' in self.dataset:
            for img in self.dataset['images']:
                imgs[img['id']] = img

        if 'categories' in self.dataset:
            for cat in self.dataset['categories']:
                cats[cat['id']] = cat

        # if 'annotations' in self.dataset and 'categories' in self.dataset:
        #     for ann in self.dataset['annotations']:
        #         cat_to_imgs[ann['category_id']].append(ann['image_id'])

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

        print('index created!')

        # create class members
        self.anns = anns
        # self.img_to_anns = img_to_anns
        # self.cat_to_imgs = cat_to_imgs
        self.imgs = imgs
        self.cats = cats
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

    def get_mask(self, masks=[], new_shape=(512, 512)):
        for img_key in self.imgs.keys():
            # bitmasks = np.zeros((len(pd.cat_ids), self.imgs[img_key]['height'], self.imgs[img_key]['width']))
            bitmasks = np.zeros((len(pd.cat_ids), new_shape[0], new_shape[1]))
            print(f'empty bitmasks.shape={bitmasks.shape}')
            for img_to_segment in self.img_to_segments[img_key]:
                for cat in self.cat_ids:
                    if img_to_segment['category_id'] == cat:
                        # print(f'img_to_segment={img_to_segment}')
                        print(f'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
                        bitmask = self._polygons_to_bitmask(
                            img_to_segment['segmentation'],
                            self.imgs[img_key]['height'], self.imgs[img_key]['width']
                        ).astype('uint8')
                        print(f'max old bitmask={np.max(bitmask)}')
                        bitmasks[cat - 1] += np.resize(bitmask, new_shape=new_shape)
                        print(f'max new bitmask={np.max(np.resize(bitmask, new_shape=new_shape))}')
                        print(f'bitmask.shape={bitmask.shape}')
                        print(f'max bitmasks={np.max(bitmasks)}')
            print(f'full bitmasks.shape={bitmasks.shape}')
            masks.append(bitmasks)
        return masks


if __name__ == '__main__':
    pd = PotatoDataset('../datasets/potato_set16_coco.json')

    # print(f'pd.img_to_anns={pd.img_to_anns}')
    print(f'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print(f'pd.imgs={pd.imgs}')
    print(f'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print(f'pd.cats={pd.cats}')
    print(f'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    # print(f'pd.cat_to_imgs={pd.cat_to_imgs}')

    masks = pd.get_mask()
    print(f'pd.imgs.keys()={pd.imgs.keys()}')
    # np.stack((masks, bitmasks.astype('uint8')), axis=2)
    # cv2_imshow(mask.astype("uint8"))
    # print(f'pb={mask.astype("uint8")}')
    print(f'cat_ids={pd.cat_ids}')
    print(f'masks.shape={np.array(masks).shape}')
    # cv2_imshow(bitmask)
    # print(f'bitmasks.shape={bitmasks.shape}')

    # print(f'bitmasks={bitmasks}')
    # cv2_imshow(bitmasks[0].reshape((height, width, 1)), 'bitmasks[0]')
    # print(np.array(masks)[0])
    # masks[masks == 1] = 255
    print(np.array(masks)[0].shape)
    # cv2_imshow(bitmasks[1].reshape((height, width, 1)), 'bitmasks[1]')
    cv2_imshow(np.array(masks)[0][2].reshape((512, 512, 1)), 'masks[0][0]')
