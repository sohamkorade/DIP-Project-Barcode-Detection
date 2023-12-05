import cv2
import torch
from torchvision.transforms import functional as F
from utils import draw_pred_boxes


class BarcodeDetector:

    def __init__(self, model_path='baseline.pth'):
        self.device = torch.device(
            'cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model = self.load_model(model_path)

    def load_model(self, model_path):
        model = torch.load(model_path, map_location=self.device)
        model.eval()
        return model

    def infer(self, image):
        with torch.no_grad():
            image = F.to_tensor(image).to(self.device)
            predictions = self.model([image])
            # check if empty
            if len(predictions[0]['boxes']) == 0:
                return None
            return predictions


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
            predictions = detector.infer(frame)
            if predictions:
                frame = draw_pred_boxes(frame, predictions)

        cv2.imshow('Live Inference', frame)

        if key & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    detector = BarcodeDetector()
    live_inference(detector)


if __name__ == '__main__':
    main()
