import itertools
import json
import time
from collections import defaultdict


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
        self.img_to_anns, self.cat_to_imgs, self.img_to_segments, self.cat_ids =\
            defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)
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
        img_to_anns, cat_to_imgs, img_to_segments, cat_ids = defaultdict(list), defaultdict(list), defaultdict(list), \
                                                             defaultdict(list)
        if 'annotations' in self.dataset:
            for ann in self.dataset['annotations']:
                img_to_anns[ann['image_id']].append(ann)
                anns[ann['id']] = ann

        if 'images' in self.dataset:
            for img in self.dataset['images']:
                imgs[img['id']] = img

        if 'categories' in self.dataset:
            for cat in self.dataset['categories']:
                cats[cat['id']] = cat

        if 'annotations' in self.dataset and 'categories' in self.dataset:
            for ann in self.dataset['annotations']:
                cat_to_imgs[ann['category_id']].append(ann['image_id'])

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
        self.img_to_anns = img_to_anns
        self.cat_to_imgs = cat_to_imgs
        self.imgs = imgs
        self.cats = cats
        self.img_to_segments = img_to_segments
        self.cat_ids = cat_ids

    def get_an_ids(self, img_ids=[], cat_ids=[], areaRng=[], iscrowd=None):
        """
        Get ann ids that satisfy given filter conditions. default skips that filter
        :param img_ids  (int array)     : get anns for given imgs
               cat_ids  (int array)     : get anns for given cats
               areaRng (float array)   : get anns for given area range (e.g. [0 inf])
               iscrowd (boolean)       : get anns for given crowd label (False or True)
        :return: ids (int array)       : integer array of ann ids
        """
        img_ids = img_ids if _isArrayLike(img_ids) else [img_ids]
        cat_ids = cat_ids if _isArrayLike(cat_ids) else [cat_ids]

        if len(img_ids) == len(cat_ids) == len(areaRng) == 0:
            anns = self.dataset['annotations']
        else:
            if not len(img_ids) == 0:
                lists = [self.img_to_anns[imgId] for imgId in img_ids if imgId in self.img_to_anns]
                anns = list(itertools.chain.from_iterable(lists))
            else:
                anns = self.dataset['annotations']
            anns = anns if len(cat_ids) == 0 else [ann for ann in anns if ann['category_id'] in cat_ids]
            anns = anns if len(areaRng) == 0 else [ann for ann in anns if
                                                   ann['area'] > areaRng[0] and ann['area'] < areaRng[1]]
        if iscrowd is not None:
            ids = [ann['id'] for ann in anns if ann['iscrowd'] == iscrowd]
        else:
            ids = [ann['id'] for ann in anns]
        return ids

    def get_cat_ids(self, cat_nms=[], sup_nms=[], cat_ids=[]):
        """
        filtering parameters. default skips that filter.
        :param cat_nms (str array)  : get cats for given cat names
        :param sup_nms (str array)  : get cats for given supercategory names
        :param cat_ids (int array)  : get cats for given cat ids
        :return: ids (int array)   : integer array of cat ids
        """
        cat_nms = cat_nms if _isArrayLike(cat_nms) else [cat_nms]
        sup_nms = sup_nms if _isArrayLike(sup_nms) else [sup_nms]
        cat_ids = cat_ids if _isArrayLike(cat_ids) else [cat_ids]

        if len(cat_nms) == len(sup_nms) == len(cat_ids) == 0:
            cats = self.dataset['categories']
        else:
            cats = self.dataset['categories']
            cats = cats if len(cat_nms) == 0 else [cat for cat in cats if cat['name'] in cat_nms]
            cats = cats if len(sup_nms) == 0 else [cat for cat in cats if cat['supercategory'] in sup_nms]
            cats = cats if len(cat_ids) == 0 else [cat for cat in cats if cat['id'] in cat_ids]
        ids = [cat['id'] for cat in cats]
        return ids

    def get_img_ids(self, img_ids=[], cat_ids=[]):
        """
        Get img ids that satisfy given filter conditions.
        :param img_ids (int array) : get imgs for given ids
        :param cat_ids (int array) : get imgs with all given cats
        :return: ids (int array)  : integer array of img ids
        """
        img_ids = img_ids if _isArrayLike(img_ids) else [img_ids]
        cat_ids = cat_ids if _isArrayLike(cat_ids) else [cat_ids]

        if len(img_ids) == len(cat_ids) == 0:
            ids = self.imgs.keys()
        else:
            ids = set(img_ids)
            for i, catId in enumerate(cat_ids):
                if i == 0 and len(ids) == 0:
                    ids = set(self.cat_to_imgs[catId])
                else:
                    ids &= set(self.cat_to_imgs[catId])
        return list(ids)


if __name__ == '__main__':
    pd = PotatoDataset('../datasets/potato_set16_coco.json')
    pd.create_index()
    # print(f'pd.img_to_anns={pd.img_to_anns}')
    print(f'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print(f'pd.imgs={pd.imgs}')
    print(f'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print(f'pd.cats={pd.cats}')
    print(f'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print(f'pd.cat_to_imgs={pd.cat_to_imgs}')
    for cat in pd.cat_ids:
        for img_to_segment in pd.img_to_segments[1]:
            if img_to_segment['category_id'] == cat:
                print(f'img_to_segment={img_to_segment}')
        print(f'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print(f'cat_ids={pd.cat_ids}')

