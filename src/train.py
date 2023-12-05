import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--det_thres', type=float, default=0.5)
parser.add_argument('--model_path',
                    type=str,
                    default='fasterrcnn_resnet50_fpn2.pth')
parser.add_argument('--es_patience', type=int, default=2)
args = parser.parse_args()

import pandas as pd
import numpy as np
import cv2
import os
import time
import datetime

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

import torch
import torchvision

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import DataLoader, Dataset

from matplotlib import pyplot as plt

import cv2

from utils import calculate_image_precision
from ensemble_boxes import weighted_boxes_fusion

import warnings

warnings.filterwarnings("ignore")

print("Imports done")

DIR_INPUT = 'dataset_inventbar'
DIR_TRAIN = f'{DIR_INPUT}/Training'
DIR_TEST = f'{DIR_INPUT}/Testing'
DIR_VALID = f'{DIR_INPUT}/Validation'
model_path = args.model_path


def preprocess(df):
    df['x'] = df['x'].astype(np.float32)
    df['y'] = df['y'].astype(np.float32)
    df['w'] = df['w'].astype(np.float32)
    df['h'] = df['h'].astype(np.float32)
    return df


train_df = pd.read_csv(f'{DIR_INPUT}/Training.csv')
train_df = preprocess(train_df)
train_ids = train_df['image_id'].unique()

valid_df = pd.read_csv(f'{DIR_INPUT}/Validation.csv')
valid_df = preprocess(valid_df)
valid_ids = valid_df['image_id'].unique()

test_df = pd.read_csv(f'{DIR_INPUT}/Testing.csv')
test_df = preprocess(test_df)
test_ids = test_df['image_id'].unique()

print("Train, Val, Test shapes:", train_ids.shape, valid_ids.shape,
      test_ids.shape)


class BarcodeDS(Dataset):

    def __init__(self, df, image_dir, transforms=None, testmode=False):
        super().__init__()

        self.image_ids = df['image_id'].unique()
        self.df = df
        self.image_dir = image_dir
        self.transforms = transforms
        self.testmode = testmode

    def __getitem__(self, index: int):
        image_id = self.image_ids[index]

        image = cv2.imread(f'{self.image_dir}/{image_id}.jpg',
                           cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0

        if self.testmode:
            sample = {
                'image': image,
            }
            sample = self.transforms(**sample)
            image = sample['image']
            return image, image_id
        else:
            records = self.df[self.df['image_id'] == image_id]
            # changed
            # resize
            image = cv2.resize(image, (1024, 1024),
                               interpolation=cv2.INTER_AREA)

            boxes = records[[
                'x', 'y', 'w', 'h'
            ]].values  # x and y are the centers and w and h are width and height

            # changed

            # rescale
            boxes *= 1024
            boxes[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
            boxes[:, 1] = boxes[:, 1] - boxes[:, 3] / 2

            boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
            boxes[:, 3] = boxes[:, 1] + boxes[:, 3]

            # now boxes are in xmin, ymin, xmax, ymax format

            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            area = torch.as_tensor(area, dtype=torch.float32)

            # there is only one class
            labels = torch.ones((records.shape[0], ), dtype=torch.int64)

            target = {}
            target['boxes'] = boxes
            target['labels'] = labels
            target['image_id'] = torch.tensor([index])
            target['area'] = area
            if self.transforms:
                sample = {
                    'image': image,
                    'bboxes': target['boxes'],
                    'labels': labels
                }
                sample = self.transforms(**sample)
                image = sample['image']

                target['boxes'] = torch.stack(
                    tuple(map(torch.tensor,
                              zip(*sample['bboxes'])))).permute(1, 0)
            return image, target, image_id

    def __len__(self):
        return self.image_ids.shape[0]


#######################
### Training
#######################


# Albumentations
def get_train_transform():
    return A.Compose(
        [
            A.Flip(0.5),
            # A.RandomCropNearBBox(p=1.0, max_part_shift=0.2),
            A.Rotate(limit=45, p=1.0),
            ToTensorV2(p=1.0),
        ],
        bbox_params={
            'format': 'pascal_voc',
            'label_fields': ['labels']
        })


def get_valid_transform():
    return A.Compose([
        A.Rotate(limit=45, p=1.0),
        ToTensorV2(p=1.0),
    ],
                     bbox_params={
                         'format': 'pascal_voc',
                         'label_fields': ['labels']
                     })


# load model
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

num_classes = 2  # barcode + background

# get number of input features for the classifier
in_features = model.roi_heads.box_predictor.cls_score.in_features

# replace the pre-trained head with a new one
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

train_dataset = BarcodeDS(train_df, DIR_TRAIN, get_train_transform())
valid_dataset = BarcodeDS(valid_df, DIR_VALID, get_valid_transform())


def collate_fn(batch):
    return tuple(zip(*batch))


train_data_loader = DataLoader(train_dataset,
                               batch_size=8,
                               shuffle=False,
                               num_workers=4,
                               collate_fn=collate_fn)

valid_data_loader = DataLoader(valid_dataset,
                               batch_size=8,
                               shuffle=False,
                               num_workers=4,
                               collate_fn=collate_fn)

models = [model]


def ensemble_pred(images):
    images = [image.to(device) for image in images]
    result = []
    for net in models:
        net.eval()
        outputs = net(images)
        result.append(outputs)
    return result


def run_wbf(predictions,
            image_index,
            image_size=1024,
            iou_thr=0.55,
            skip_box_thr=0.5,
            weights=None):
    boxes = [
        prediction[image_index]['boxes'].data.cpu().numpy() / (image_size - 1)
        for prediction in predictions
    ]
    scores = [
        prediction[image_index]['scores'].data.cpu().numpy()
        for prediction in predictions
    ]
    labels = [
        np.ones(prediction[image_index]['scores'].shape[0])
        for prediction in predictions
    ]
    boxes, scores, labels = weighted_boxes_fusion(boxes,
                                                  scores,
                                                  labels,
                                                  weights=weights,
                                                  iou_thr=iou_thr,
                                                  skip_box_thr=skip_box_thr)
    boxes = boxes * (image_size - 1)
    return boxes, scores, labels


device = torch.device('cuda') if torch.cuda.is_available() else torch.device(
    'cpu')

model.to(device)
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params,
                            lr=0.005,
                            momentum=0.9,
                            weight_decay=0.0005)

lr_scheduler = None
# lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

num_epochs = args.epochs

# plot some images from dataloader
images, targets, image_ids = next(iter(train_data_loader))

img = images[0].permute(1, 2, 0).cpu().numpy()
img = np.clip(img, 0, 1)
boxes = targets[0]['boxes'].cpu().numpy().astype(np.int32)
fig, ax = plt.subplots(1, 1, figsize=(16, 8))
# print(boxes)
for box in boxes:
    cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (220, 0, 0), 2)

img = np.clip(img * 255, 0, 255)  # proper [0..255] range
img = img.astype(np.uint8)
cv2.imwrite('sample_train.png', img)
# print(targets)

best_val = None
patience = args.es_patience
for epoch in range(num_epochs):
    start_time = time.time()
    iteration = 1
    losses_list = []
    model.train()
    for images, targets, image_ids in train_data_loader:
        images = list(image.to(device) for image in images)
        targets = [{
            k: v.to(device) if k == 'labels' else v.float().to(device)
            for k, v in t.items()
        } for t in targets]
        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()

        losses_list.append(loss_value)

        # update the parameters
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if iteration % 50 == 0:
            print(f"Iter: #{iteration} Loss: {loss_value}")

        iteration += 1

    # update the learning rate
    if lr_scheduler is not None:
        lr_scheduler.step()

    # At every epoch we will also calculate the validation IOU
    validation_image_precisions = []
    iou_thresholds = [x for x in np.arange(0.5, 0.76, 0.05)]
    model.eval()
    for images, targets, imageids in valid_data_loader:  #return image, target, image_id
        images = list(image.to(device) for image in images)
        targets = [{
            k: v.to(device) if k == 'labels' else v.float().to(device)
            for k, v in t.items()
        } for t in targets]

        preds = ensemble_pred(images)

        for i, image in enumerate(images):
            boxes, scores, labels = run_wbf(preds, image_index=i)
            boxes = boxes.astype(np.int32).clip(min=0, max=1023)

            preds = boxes
            preds_sorted_idx = np.argsort(scores)[::-1]
            preds_sorted = preds[preds_sorted_idx]
            gt_boxes = targets[i]['boxes'].cpu().numpy().astype(np.int32)
            image_precision = calculate_image_precision(
                preds_sorted, gt_boxes, thresholds=iou_thresholds, form='coco')

            validation_image_precisions.append(image_precision)
    val_iou = np.mean(validation_image_precisions)
    avg_loss = np.mean(losses_list)
    end_time = time.time()
    diff_time = end_time - start_time
    print(
        f"Epoch #{epoch+1} Loss: {avg_loss} Val IOU: {val_iou:.4f} Time: {diff_time} sec"
    )
    if not best_val:
        best_val = val_iou  # first model is best model initially
        print("Saving first model")
        torch.save(model, model_path)
    if val_iou >= best_val:
        print(f"IOU improved: {best_val:.4f} -> {val_iou:.4f}")
        print("Saving current model")
        best_val = val_iou
        # reset patience since there's improvement
        patience = args.es_patience
        torch.save(model, model_path)
    else:
        patience -= 1
        if patience == 0:
            print(f"Stopped early. Best Val IOU: {best_val:.4f}")
            break

#######################
### Testing
#######################

model = torch.load(model_path)

test_df = pd.DataFrame()

paths = []
for path in os.listdir(DIR_TEST):
    if path.endswith(".jpg"):
        filename = os.path.basename(path)
        paths.append(filename.replace(".jpg", ""))
test_df['image_id'] = np.array(paths)


def get_test_transform():
    return A.Compose([
        A.Rotate(limit=45, p=1.0),
        # A.Resize(512, 512),
        ToTensorV2(p=1.0)
    ])


def collate_fn(batch):
    return tuple(zip(*batch))


test_dataset = BarcodeDS(test_df,
                         DIR_TEST,
                         get_test_transform(),
                         testmode=True)

test_data_loader = DataLoader(test_dataset,
                              batch_size=4,
                              shuffle=False,
                              num_workers=4,
                              drop_last=False,
                              collate_fn=collate_fn)

model.eval()

detection_threshold = args.det_thres
results = []
outputs = []
test_images = []
for images, image_ids in test_data_loader:
    preds = ensemble_pred(images)

    for i, image in enumerate(images):
        test_images.append(image)  #Saving image values
        boxes, scores, labels = run_wbf(preds, image_index=i)

        boxes = boxes.astype(np.int32).clip(min=0, max=1023)

        preds = boxes
        preds_sorted_idx = np.argsort(scores)[::-1]
        preds_sorted = preds[preds_sorted_idx]
        boxes = preds

        output = {'boxes': boxes, 'scores': scores}

        outputs.append(output)  #Saving outputs and scores
        image_id = image_ids[i]

os.makedirs('results', exist_ok=True)

# save images with bounding boxes
for i in range(min(10, len(test_images))):
    sample = test_images[i].permute(1, 2, 0).cpu().numpy()
    boxes = outputs[i]['boxes']
    scores = outputs[i]['scores']
    boxes = boxes[scores >= detection_threshold].astype(np.int32)

    for box in boxes:
        cv2.rectangle(sample, (box[0], box[1]), (box[2], box[3]), (220, 0, 0),
                      2)

    # save images
    from datetime import datetime
    ts = datetime.now().strftime("%Y%m%d%H%M%S")

    sample = np.clip(sample * 255, 0, 255)  # proper [0..255] range
    sample = sample.astype(np.uint8)

    cv2.imwrite(f'./results/{i}_{ts}.jpg', sample)
