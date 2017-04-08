# Retrained VGG 19 Model for CIFAR 10

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
