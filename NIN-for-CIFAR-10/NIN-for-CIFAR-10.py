def unpickle(file):
    import cPickle
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict

if __name__ == '__main__':
    print unpickle('./cifar-10-batches-py/data_batch_1')
