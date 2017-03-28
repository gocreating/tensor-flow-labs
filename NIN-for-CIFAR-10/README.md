# NIN for CIFAR-10

## Installation

``` bash
sudo apt-get install python-matplotlib
```

## Download Data

``` bash
wget http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
tar -xvzf ./cifar-10-python.tar.gz
```

## Up and Running

Setup tensorflow virtual environment:

``` bash
source ~/tensorflow/bin/activate
```

Use `nohup` to keep program alive:

``` bash
# nothing applied
nohup python -u NIN-for-CIFAR-10.py --log "no-aug-logs.csv" &
# augmentation
nohup python -u NIN-for-CIFAR-10.py --log "aug-logs.csv" --aug &
# augmentation + elu
nohup python -u NIN-for-CIFAR-10.py --log "aug-elu-logs.csv" --aug --elu &
# augmentation + weight initialization
nohup python -u NIN-for-CIFAR-10.py --log "aug-wi-logs.csv" --aug --weight-initial &
# augmentation + batch normalization
nohup python -u NIN-for-CIFAR-10.py --log "aug-bn-logs.csv" --aug --bn &
# augmentation + elu + weight initialization
nohup python -u NIN-for-CIFAR-10.py --log "aug-elu-wi-logs.csv" --aug --elu --weight-initial &
# augmentation + elu + batch normalization
nohup python -u NIN-for-CIFAR-10.py --log "aug-elu-bn-logs.csv" --aug --elu --bn &
# augmentation + weight initialization + batch normalization
nohup python -u NIN-for-CIFAR-10.py --log "aug-wi-bn-logs.csv" --aug --weight-initial --bn &
# augmentation + elu + weight initialization + batch normalization
nohup python -u NIN-for-CIFAR-10.py --log "aug-elu-wi-bn-logs.csv" --aug --elu --weight-initial --bn &
```

## Log Visualization

Generate `accuracy.jpg`, `error.jpg` and `loss.jpg`

``` bash
python plot.py --log "<log_file>.csv"
```

## Result

| elu | weight initialization | BN | Final Accuracy |
| --- | --- | --- | --- |
|   |   |   | [0.841](./results/aug) |
| v |   |   | [0.863](./results/aug-elu) |
|   | v |   | Not Converging |
|   |   | v | [0.883](./results/aug-bn) |
| v | v |   | -     |
| v |   | v | [0.886](./results/aug-elu-bn) |
|   | v | v | Not Converging |
| v | v | v | Not Converging |

## Sample Console Logs

```
Epoch 1, Test accuracy 0.174524, Train loss 2.28849807206, Elapsed time 67.6s
Epoch 2, Test accuracy 0.158381, Train loss 2.26725660352, Elapsed time 118.2s
Epoch 3, Test accuracy 0.19581, Train loss 2.18239071089, Elapsed time 168.6s
Epoch 4, Test accuracy 0.19819, Train loss 2.15034614591, Elapsed time 219.1s
Epoch 5, Test accuracy 0.204476, Train loss 2.09548012649, Elapsed time 269.5s
Epoch 6, Test accuracy 0.129333, Train loss 2.16256986646, Elapsed time 319.9s
Epoch 7, Test accuracy 0.164905, Train loss 2.12558329807, Elapsed time 370.2s
Epoch 8, Test accuracy 0.20619, Train loss 2.12630182855, Elapsed time 420.5s
Epoch 9, Test accuracy 0.227095, Train loss 2.09054305273, Elapsed time 470.9s
Epoch 10, Test accuracy 0.166095, Train loss 2.15404876541, Elapsed time 521.1s
Epoch 11, Test accuracy 0.222429, Train loss 2.07544895481, Elapsed time 571.4s
Epoch 12, Test accuracy 0.227286, Train loss 2.04094898701, Elapsed time 621.6s
Epoch 13, Test accuracy 0.236238, Train loss 2.03972777549, Elapsed time 671.9s
Epoch 14, Test accuracy 0.224619, Train loss 2.0527709407, Elapsed time 722.2s
Epoch 15, Test accuracy 0.238524, Train loss 2.03319939445, Elapsed time 772.5s
Epoch 16, Test accuracy 0.256667, Train loss 1.98097934793, Elapsed time 822.9s
Epoch 17, Test accuracy 0.22, Train loss 2.06845567507, Elapsed time 873.1s
Epoch 18, Test accuracy 0.220524, Train loss 2.13206830445, Elapsed time 923.4s
Epoch 19, Test accuracy 0.258714, Train loss 1.96293732349, Elapsed time 973.8s
Epoch 20, Test accuracy 0.253857, Train loss 1.98042768941, Elapsed time 1024.1s
Epoch 21, Test accuracy 0.273571, Train loss 1.92065281027, Elapsed time 1074.5s
Epoch 22, Test accuracy 0.271952, Train loss 1.92781277264, Elapsed time 1124.8s
Epoch 23, Test accuracy 0.290762, Train loss 1.8833787862, Elapsed time 1175.2s
...
```

## Sample File Logs

```
0,0.10061904788,2.30240570798,15.2526872158
1,0.174523811255,2.28849807206,67.5721940994
2,0.158380949071,2.26725660352,118.155235052
3,0.195809515459,2.18239071089,168.612190008
4,0.198190469827,2.15034614591,219.061039209
5,0.20447618195,2.09548012649,269.49122715
6,0.129333328988,2.16256986646,319.861015081
7,0.164904754077,2.12558329807,370.200902224
8,0.206190466881,2.12630182855,420.543041229
9,0.227095225028,2.09054305273,470.865853071
10,0.166095233389,2.15404876541,521.095661163
11,0.222428568772,2.07544895481,571.354311228
12,0.227285702314,2.04094898701,621.642080069
13,0.236238087927,2.03972777549,671.910745144
14,0.224619041596,2.0527709407,722.199918032
15,0.238523800458,2.03319939445,772.522424221
16,0.256666656051,1.98097934793,822.862049103
17,0.219999986035,2.06845567507,873.111236095
18,0.220523804426,2.13206830445,923.434288025
19,0.258714273572,1.96293732349,973.78949523
20,0.253857135773,1.98042768941,1024.12503815
21,0.273571423122,1.92065281027,1074.46490121
22,0.271952382156,1.92781277264,1124.8094461
23,0.2907619008,1.8833787862,1175.18159604
...
```

## Windows

### Installation of CPU Version

Install [Anaconda](https://www.continuum.io/downloads)

```
conda create -n tensorflow python=3.5
activate tensorflow
pip install --ignore-installed --upgrade https://storage.googleapis.com/tensorflow/windows/cpu/tensorflow-1.0.1-cp35-cp35m-win_amd64.whl
```

### Usage

```
activate tensorflow
```

### Sample Console Logs

It's really not recommended to use CPU version.

It took up to 3700 seconds to finish just 1 epoch of the default version(no data augmentation, not using elu, no weight initialization and not using batch normalization)

```
Epoch 0, Test accuracy 0.101266, Train loss 2.30255841782, Elapsed time 1118.2s
Epoch 1, Test accuracy 0.318236, Train loss 1.73775102781, Elapsed time 4867.4s
Epoch 2, Test accuracy 0.439577, Train loss 1.47241931163, Elapsed time 8601.5s
Epoch 3, Test accuracy 0.541634, Train loss 1.21320135651, Elapsed time 12354.0s
Epoch 4, Test accuracy 0.600079, Train loss 1.07322646498, Elapsed time 16140.3s
Epoch 5, Test accuracy 0.655459, Train loss 0.924272480828, Elapsed time 19902.0s
Epoch 6, Test accuracy 0.668809, Train loss 0.863943901056, Elapsed time 23673.7s
Epoch 7, Test accuracy 0.700949, Train loss 0.780749371747, Elapsed time 27458.5s
Epoch 8, Test accuracy 0.720431, Train loss 0.706208645154, Elapsed time 31252.7s
Epoch 9, Test accuracy 0.731903, Train loss 0.66900576289, Elapsed time 35065.9s
Epoch 10, Test accuracy 0.75623, Train loss 0.587163880887, Elapsed time 38960.1s
Epoch 11, Test accuracy 0.752868, Train loss 0.585411774685, Elapsed time 42816.9s
Epoch 12, Test accuracy 0.772547, Train loss 0.532328287964, Elapsed time 46686.8s
Epoch 13, Test accuracy 0.769778, Train loss 0.547780972155, Elapsed time 50546.1s
Epoch 14, Test accuracy 0.779173, Train loss 0.494357929205, Elapsed time 54419.0s
...
```

## Reference

- <http://www.cs.toronto.edu/~kriz/cifar.html>
- <https://github.com/JiaRenChang/DLcourse_NCTU/blob/master/NIN_MNIST.py>
