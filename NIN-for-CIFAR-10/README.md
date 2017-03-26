# Report

## Download Data

``` bash
wget http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
tar -xvzf ./cifar-10-python.tar.gz
```

## Implement NIN

## Implement Data Augmentation

## Up and Running

``` bash
source ~/tensorflow/bin/activate
nohup python -u NIN-for-CIFAR-10.py & # to keep program alive
```

## Sample Logs

```
epoch 1, batch 0 (0 ~ 1500), Test accuracy 0.100429, time 0.0s
epoch 1, batch 1 (1500 ~ 3000), Test accuracy 0.100619, time 6.1s
epoch 1, batch 2 (3000 ~ 4500), Test accuracy 0.1, time 9.4s
epoch 1, batch 3 (4500 ~ 6000), Test accuracy 0.100333, time 12.7s
epoch 1, batch 4 (6000 ~ 7500), Test accuracy 0.100333, time 15.9s
epoch 1, batch 5 (7500 ~ 9000), Test accuracy 0.100333, time 19.2s
...
epoch 1, batch 31 (46500 ~ 48000), Test accuracy 0.0998095, time 103.9s
epoch 1, batch 32 (48000 ~ 49500), Test accuracy 0.0998095, time 107.1s
epoch 1, batch 33 (49500 ~ 50000), Test accuracy 0.0998095, time 110.4s
epoch 2, batch 0 (0 ~ 1500), Test accuracy 0.0998095, time 113.8s
epoch 2, batch 1 (1500 ~ 3000), Test accuracy 0.0998095, time 117.0s
epoch 2, batch 2 (3000 ~ 4500), Test accuracy 0.0998095, time 120.3s
epoch 2, batch 3 (4500 ~ 6000), Test accuracy 0.100095, time 123.6s
epoch 2, batch 4 (6000 ~ 7500), Test accuracy 0.100095, time 126.8s
epoch 2, batch 5 (7500 ~ 9000), Test accuracy 0.100095, time 130.1s
...
```

## Reference

- <http://www.cs.toronto.edu/~kriz/cifar.html>
- <https://github.com/JiaRenChang/DLcourse_NCTU/blob/master/NIN_MNIST.py>
