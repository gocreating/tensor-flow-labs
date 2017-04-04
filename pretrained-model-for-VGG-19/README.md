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
nohup python -u pretrained-model-for-VGG-19.py --log "no-aug-logs.csv" &
# augmentation
nohup python -u pretrained-model-for-VGG-19.py --log "aug-logs.csv" --aug &
```

## Reference

- <https://github.com/machrisaa/tensorflow-vgg>
