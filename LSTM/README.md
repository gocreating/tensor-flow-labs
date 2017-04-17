# LSTM

## Up and Running

``` bash
nohup python -u LSTM.py --log "train-20-test-10.csv" --train-length 20 --test-length 10 &
nohup python -u LSTM.py --log "train-20-test-20.csv" --train-length 20 --test-length 20 &
nohup python -u LSTM.py --log "train-20-test-30.csv" --train-length 20 --test-length 30 &
nohup python -u LSTM.py --log "train-30-test-20.csv" --train-length 30 --test-length 20 &
nohup python -u LSTM.py --log "train-30-test-30.csv" --train-length 30 --test-length 30 &
nohup python -u LSTM.py --log "train-30-test-50.csv" --train-length 30 --test-length 50 &
```

## Reference

- <https://github.com/hans/ipython-notebooks/blob/master/tf/TF%20tutorial.ipynb>
