# Retrained VGG 19 Model for CIFAR 10

## Up and Running

With pre-trained initialization

``` bash
# nothing applied
nohup python -u retrained-VGG-19-model-for-CIFAR-10.py --log "no-aug-logs.csv" &
# augmentation
nohup python -u retrained-VGG-19-model-for-CIFAR-10.py --log "aug-logs.csv" --aug &
# augmentation + batch normalization
nohup python -u retrained-VGG-19-model-for-CIFAR-10.py --log "aug-bn-logs.csv" --aug --bn &
```

With random initialization

``` bash
# nothing applied
nohup python -u retrained-VGG-19-model-for-CIFAR-10.py --random-init --log "ri-no-aug-logs.csv" &
# augmentation
nohup python -u retrained-VGG-19-model-for-CIFAR-10.py --random-init --log "ri-aug-logs.csv" --aug &
# augmentation + batch normalization
nohup python -u retrained-VGG-19-model-for-CIFAR-10.py --random-init --log "ri-aug-bn-logs.csv" --aug --bn &
```

## Reference

- <https://github.com/machrisaa/tensorflow-vgg>
