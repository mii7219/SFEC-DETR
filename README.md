# SFEC-DETR# SFEC-DETR





## Environment

| Package | Version |
| --- | --- |
| Python | 3.11.14 |
| PyTorch | 2.7.1+cu128 |
| Torchvision | 0.22.1+cu128 |
| CUDA | 12.8 |
| Ultralytics | >= 8.3.0 |

## Structure

```text
├── sfeclc.py    # Core implementation of SFECLC
└── README.md
```

## Dataset Preparation

| Dataset | URL |
| --- | --- |
| VisDrone2019 | [VisDrone-Dataset](https://github.com/VisDrone/VisDrone-Dataset) |
The UAVVaste dataset is available at https://zenodo.org/record/8214061.


Dataset directory structure (example for VisDrone2019; UAVDT and TinyPerson follow the same convention):

```text
VisDrone/
├── train/
│   └── images/
├── valid/
│   └── images/
└── test/
    └── images/
```

## License

This project is released for academic use.
