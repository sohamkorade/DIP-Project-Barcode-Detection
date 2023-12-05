from numba import jit
import numpy as np
import cv2


@jit(nopython=True)
def calculate_iou(gt, pr, form='pascal_voc') -> float:
    """Calculates the Intersection over Union.

    Args:
        gt: (np.ndarray[Union[int, float]]) coordinates of the ground-truth box
        pr: (np.ndarray[Union[int, float]]) coordinates of the prdected box
        form: (str) gt/pred coordinates format
            - pascal_voc: [xmin, ymin, xmax, ymax]
            - coco: [xmin, ymin, w, h]
    Returns:
        (float) Intersection over union (0.0 <= iou <= 1.0)
    """

    # Calculate overlap area
    dx = min(gt[2], pr[2]) - max(gt[0], pr[0]) + 1

    if dx < 0:
        return 0.0

    dy = min(gt[3], pr[3]) - max(gt[1], pr[1]) + 1

    if dy < 0:
        return 0.0

    overlap_area = dx * dy

    # Calculate union area
    union_area = ((gt[2] - gt[0] + 1) * (gt[3] - gt[1] + 1) +
                  (pr[2] - pr[0] + 1) * (pr[3] - pr[1] + 1) - overlap_area)

    return overlap_area / union_area


@jit(nopython=True)
def find_best_match(gts,
                    pred,
                    pred_idx,
                    threshold=0.5,
                    form='pascal_voc',
                    ious=None) -> int:
    """Returns the index of the 'best match' between the
    ground-truth boxes and the prediction. The 'best match'
    is the highest IoU. (0.0 IoUs are ignored).

    Args:
        gts: (List[List[Union[int, float]]]) Coordinates of the available ground-truth boxes
        pred: (List[Union[int, float]]) Coordinates of the predicted box
        pred_idx: (int) Index of the current predicted box
        threshold: (float) Threshold
        form: (str) Format of the coordinates
        ious: (np.ndarray) len(gts) x len(preds) matrix for storing calculated ious.

    Return:
        (int) Index of the best match GT box (-1 if no match above threshold)
    """
    best_match_iou = -np.inf
    best_match_idx = -1

    for gt_idx in range(len(gts)):

        if gts[gt_idx][0] < 0:
            # Already matched GT-box
            continue

        iou = -1 if ious is None else ious[gt_idx][pred_idx]

        if iou < 0:
            iou = calculate_iou(gts[gt_idx], pred, form=form)

            if ious is not None:
                ious[gt_idx][pred_idx] = iou

        if iou < threshold:
            continue

        if iou > best_match_iou:
            best_match_iou = iou
            best_match_idx = gt_idx

    return best_match_idx


@jit(nopython=True)
def calculate_precision(gts,
                        preds,
                        threshold=0.5,
                        form='coco',
                        ious=None) -> float:
    """Calculates precision for GT - prediction pairs at one threshold.

    Args:
        gts: (List[List[Union[int, float]]]) Coordinates of the available ground-truth boxes
        preds: (List[List[Union[int, float]]]) Coordinates of the predicted boxes,
               sorted by confidence value (descending)
        threshold: (float) Threshold
        form: (str) Format of the coordinates
        ious: (np.ndarray) len(gts) x len(preds) matrix for storing calculated ious.

    Return:
        (float) Precision
    """
    n = len(preds)
    tp = 0
    fp = 0

    # for pred_idx, pred in enumerate(preds_sorted):
    for pred_idx in range(n):

        best_match_gt_idx = find_best_match(gts,
                                            preds[pred_idx],
                                            pred_idx,
                                            threshold=threshold,
                                            form=form,
                                            ious=ious)

        if best_match_gt_idx >= 0:
            # True positive: The predicted box matches a gt box with an IoU above the threshold.
            tp += 1
            # Remove the matched GT box
            gts[best_match_gt_idx] = -1

        else:
            # No match
            # False positive: indicates a predicted box had no associated gt box.
            fp += 1

    # False negative: indicates a gt box had no associated predicted box.
    fn = (gts.sum(axis=1) > 0).sum()

    return tp / (tp + fp + fn)


@jit(nopython=True)
def calculate_image_precision(gts,
                              preds,
                              thresholds=(0.5, ),
                              form='coco') -> float:
    """Calculates image precision.

    Args:
        gts: (List[List[Union[int, float]]]) Coordinates of the available ground-truth boxes
        preds: (List[List[Union[int, float]]]) Coordinates of the predicted boxes,
               sorted by confidence value (descending)
        thresholds: (float) Different thresholds
        form: (str) Format of the coordinates

    Return:
        (float) Precision
    """
    n_threshold = len(thresholds)
    image_precision = 0.0

    ious = np.ones((len(gts), len(preds))) * -1
    # ious = None

    for threshold in thresholds:
        precision_at_threshold = calculate_precision(gts.copy(),
                                                     preds,
                                                     threshold=threshold,
                                                     form=form,
                                                     ious=ious)
        image_precision += precision_at_threshold / n_threshold

    return image_precision


def draw_pred_boxes(image, preds, thres=0.5):
    # Draw boxes, labels, and scores on the image
    for i in range(len(preds[0]['boxes'])):
        boxes = preds[0]['boxes'][i].cpu().numpy()
        score = preds[0]['scores'][i].cpu().numpy()
        label = preds[0]['labels'][i].cpu().numpy()

        if score > thres:
            x1, y1, x2, y2 = boxes.astype(int)
            image = cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, f'{label} {score}', (x1, y1),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
    return image

def crop_pred(image, preds):
    # preds format:
    # [{'boxes': tensor([[ 277.5048,  165.3501, 1190.6555,  982.0909]]), 'labels': tensor([1]), 'scores': tensor([0.8125])}]
    
    # Crop the image
    for i in range(len(preds[0]['boxes'])):
        boxes = preds[0]['boxes'][i].cpu().numpy()
        x1, y1, x2, y2 = boxes.astype(int)
        crop_img = image[y1:y2, x1:x2]
        return crop_img
    
    return None
