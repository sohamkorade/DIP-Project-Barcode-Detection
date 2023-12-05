# Barcode Detection and Decoding in Online Fashion Images

## Instructions

### Installing prerequisites

1. opencv-python
2. albumentations
3. ensemble_boxes
4. pyzbar
5. torch
6. torchvision
7. numpy
8. matplotlib

### Training the model

1. Clone the repository
2. Download the dataset from [here](https://www.kaggle.com/datasets/teerawatkamnardsiri/product-barcode)
3. Save the dataset in the following structure:

	```
	dataset_inventbar
	├── Training
	├── Testing
	├── Validation
	├── Training.csv
	├── Testing.csv
	└── Validation.csv
	```

4. Run the following command to train the model

	```bash
	python train.py --epochs 10
	```

5. Run the following command to run the live inference using webcam

	```bash
	python pipeline.py
	```

### Other datasets
https://github.com/BenSouchet/barcode-datasets


## Credits
| Name          | Roll No    | GitHub                                              |
| ------------- | ---------- | --------------------------------------------------- |
| Soham Korade  | 2021101131 | [sohamkorade](https://github.com/sohamkorade)       |
| Anurag Dubey  | 2021102039 | [AnuragDubey123](https://github.com/AnuragDubey123) |
| Arpit Pathak  | 2021102033 | [arpitpathak16](https://github.com/arpitpathak16)   |
| Mohammed Noor | 2021102014 | [isntnoor](https://github.com/isntnoor)             |