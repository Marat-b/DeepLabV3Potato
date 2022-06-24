from pathlib import Path

import click
import torch
from PIL import Image
from sklearn.metrics import f1_score, roc_auc_score
from torch.utils import data
from torchvision import transforms

from dataset import datahandler
from dataset.register_instances import RegisterDataset
from model import createDeepLabv3
from trainer import train_model


def main(exp_directory='weights', epochs=1, batch_size=4, out_name='potato_model', classes=4, new_shape=(512, 512),
         train_inst=None,
         test_inst=None):
    # Create the deeplabv3 resnet101 model which is pretrained on a subset
    # of COCO train2017, on the 20 categories that are present in the Pascal VOC dataset.
    if train_inst is None:
        train_inst = []
    if test_inst is None:
        test_inst = []
    # print(f'train_inst={train_inst}\ntest_inst={test_inst}')
    model = createDeepLabv3(outputchannels=4)
    model.train()
    # data_directory = Path(data_directory)
    # Create the experiment directory if not present
    exp_directory = Path(exp_directory)
    if not exp_directory.exists():
        exp_directory.mkdir()

    # Specify the loss function
    # criterion = torch.nn.MSELoss(reduction='mean')
    criterion = torch.nn.CrossEntropyLoss(reduction='mean')
    # Specify the optimizer with a lower learning rate
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Specify the evaluation metrics
    metrics = {'f1_score': f1_score, 'auroc': roc_auc_score}
    # metrics = {'f1_score': f1_score}

    # Create the dataloader
    dataloaders = datahandler.get_dataloader(
        train_instances=train_inst, new_shape=new_shape, test_instances=test_inst, \
        batch_size=batch_size
    )
    _ = train_model(
        model,
        criterion,
        dataloaders,
        optimizer,
        bpath=exp_directory,
        metrics=metrics,
        num_epochs=epochs
    )

    # Save the trained model
    to_pth = '{}.pth'.format(out_name)
    to_onnx = '{}.onnx'.format(out_name)
    torch.save(model, exp_directory / to_pth)
    # img = Image.open('./images/image_256.jpg')
    # image = transforms.ToTensor()(img).unsqueeze_(0)
    # torch.onnx.export(
    #     model.eval().to('cpu'), (image,), exp_directory / to_onnx, opset_version=12,
    #     do_constant_folding=True
    # )


if __name__ == "__main__":
    rd = RegisterDataset()
    rd.register_dataset_instances('set1', './datasets/potato_set1.json', './datasets/set1')
    rd.register_dataset_instances('set6', './datasets/potato_set6_coco.json', './datasets/set6')
    rd.register_dataset_instances('set15', './datasets/potato_set15_coco.json', './datasets/set15')
    rd.register_dataset_instances('set16', './datasets/potato_set16_coco.json', './datasets/set16')
    rd.register_dataset_instances('set37', './datasets/potato_set37_coco.json', './datasets/set37')
    main(
        epochs=1,
        train_inst=rd.get_instances(['set37']),
        test_inst=rd.get_instances(['set15']),
        new_shape=(256, 256),
        out_name='potato_model_1'
        )
    # main(epochs=1, train_inst=rd.get_instances(['set16']), test_inst=rd.get_instances(['set16']), new_shape=(256,
    # 256))
    # 'set6']),
    # tuple(['set15'])
