from pathlib import Path
from typing import List

from torch.utils.data import DataLoader
from torchvision import transforms

from dataset.potato_dataset import PotatoDataset
from dataset.register_instances import register_dataset_instances


def get_dataloader(train_instances: List = [],
                   test_instances: List = [],
                   batch_size: int = 4):
    """Create train and test dataloader

    Args:
        batch_size (int, optional): Dataloader batch size. Defaults to 4.

    Returns:
        dataloaders: Returns dataloaders dictionary containing the
        Train and Test dataloaders.
    """
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

    image_datasets = {
        'Train': PotatoDataset(
            name_instances=train_instances,
            transforms_image=data_transforms_image,
            transforms_mask=data_transforms_mask
        ),
        'Test': PotatoDataset(
            name_instances=test_instances,
            transforms_image=data_transforms_image,
            transforms_mask=data_transforms_mask
        )

    }
    dataloaders = {
        'Train': DataLoader(
            image_datasets['Train'],
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            drop_last=True
        ),
        'Test': DataLoader(
            image_datasets['Test'],
            batch_size=1,
            shuffle=False,
            num_workers=2,
            drop_last=False
        )
        # for x in ['Train', 'Test']
    }
    return dataloaders


if __name__ == '__main__':
    register_dataset_instances('set6', '../datasets/potato_set6_coco.json', '../datasets/set6')
    register_dataset_instances('set15', '../datasets/potato_set15_coco.json', '../datasets/set15')
    register_dataset_instances('set16', '../datasets/potato_set16_coco.json', '../datasets/set16')
    dataloader = get_dataloader(train_instances=['set16'], test_instances=['set16'], batch_size=1)
    # print(f'dataloader={dir(dataloader)}')
    # for i_batch, sample_batched in enumerate(dataloader):
    #     print(i_batch, sample_batched )
    print(iter(dataloader['Test']))
    for sample in iter(dataloader['Test']):
        inputs = sample['image']
        masks = sample['mask']
        print(f'inputs={inputs}')
