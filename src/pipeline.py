from detector import BarcodeDetector
from utils import draw_pred_boxes, crop_pred
from smart_rotate import smart_rotate
from pyzbar.pyzbar import decode
import matplotlib.pyplot as plt
import cv2
import numpy as np

def pipeline_code(img):
    model_path = 'baseline+rot.pth'
    detector = BarcodeDetector(model_path)
     # faster rcnn output
    pred = detector.infer(img)
    print("Faster RCNN output:")
    print(pred)
    print()

    if pred:
        img = draw_pred_boxes(img, pred)
        cropped = crop_pred(img, pred)
    else:
        cropped = img

    # rotate using morphological operations
    rotated = smart_rotate(cropped)

    # barcode decoding using pyzbar
    try:
        barcodes = decode(rotated)
    except:
        barcodes = []

    # Loop over the barcodes and print their data
    print()
    print("pyzbar output:")
    decoded = "No barcode detected"
    if barcodes:
        print(barcodes)
        decoded = barcodes[0].data.decode("utf-8")
    print(decoded)
    print()

    rows = 1
    cols = 3
    plt.subplot(rows, cols, 1)
    plt.title("Original")
    plt.imshow(img)
    plt.axis('off')

    plt.subplot(rows, cols, 2)
    plt.title("Cropped")
    plt.imshow(cropped)
    plt.axis('off')

    plt.subplot(rows, cols, 3)
    plt.title("Barcode")
    plt.imshow(rotated, cmap='gray')
    plt.text(0, rotated.shape[1], decoded, fontsize=8)
    plt.axis('off')

    plt.tight_layout()
    # plt.show()

    # return plt image
    fig = plt.gcf()
    fig.canvas.draw()
    img = np.array(fig.canvas.renderer._renderer)
    plt.close()
    return img

def live_inference(detector):
    live = True

    cap = cv2.VideoCapture(0)
    # set buffer size to 1 to reduce latency
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    while cap.isOpened():
        ret, frame = cap.read()
        key = cv2.waitKey(1)
        if not ret:
            break

        # toggle live variable
        if key & 0xFF == ord('l'):
            live = not live

        # temp live if pressed space
        temp_live = key & 0xFF == ord(' ')

        if live or temp_live:
            # predictions = detector.infer(frame)
            # frame = draw_pred_boxes(frame, predictions)
            frame = pipeline_code(frame)

        cv2.imshow('Live Inference', frame)

        if key & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    model_path = 'baseline+rot.pth'
    detector = BarcodeDetector(model_path)
    live_inference(detector)