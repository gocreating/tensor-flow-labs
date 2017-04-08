# Pre-Trained Model for VGG 19

## Installation

``` bash
pip install scikit-image
```

## Download Data

Download [vgg19.npy](https://mega.nz/!xZ8glS6J!MAnE91ND_WyfZ_8mvkuSa2YcA7q-1ehfSm-Q1fxOvvs)

## Up and Running

``` bash
# nothing applied
nohup python -u retrained-VGG-19-model-for-CIFAR-10.py --log "no-aug-logs.csv" &
# augmentation
nohup python -u retrained-VGG-19-model-for-CIFAR-10.py --log "aug-logs.csv" --aug &
# augmentation + batch normalization
nohup python -u retrained-VGG-19-model-for-CIFAR-10.py --log "aug-bn-logs.csv" --aug --bn &
```

## Reference

- <https://github.com/machrisaa/tensorflow-vgg>
