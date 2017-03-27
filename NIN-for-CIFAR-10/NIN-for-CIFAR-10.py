import argparse
import time
from random import randint
import numpy as np
import tensorflow as tf

parser = argparse.ArgumentParser()
parser.add_argument('--log', type=str, help='Filename of logs')
parser.add_argument("--aug", dest='aug', action='store_true', help='Apply data augmentation on training data')
args = parser.parse_args()

DROPOUT_RATE = 0.5
BATCH_SIZE = 128
TRAINING_DATA_SIZE = 50000
TESTING_DATA_SIZE = 10000

training_batch_count = TRAINING_DATA_SIZE / BATCH_SIZE
if TRAINING_DATA_SIZE % BATCH_SIZE is not 0:
    training_batch_count = training_batch_count + 1

testing_batch_count = TESTING_DATA_SIZE / BATCH_SIZE
if TESTING_DATA_SIZE % BATCH_SIZE is not 0:
    testing_batch_count = testing_batch_count + 1

trainingData = {
    'raw_x': np.empty(shape=[0, 32 * 32 * 3], dtype=int),
    'raw_y': np.empty(shape=[0], dtype=int),
    'x': np.empty(shape=[0, 32 * 32 * 3]),
    'y': np.empty(shape=[0, 10]),
}
testingData = {
    'raw_x': np.empty(shape=[0, 32 * 32 * 3], dtype=int),
    'raw_y': np.empty(shape=[0], dtype=int),
    'x': np.empty(shape=[0, 32 * 32 * 3]),
    'y': np.empty(shape=[0, 10]),
}

stdR = None
stdG = None
stdB = None
meanR = None
meanG = None
meanB = None

# http://stackoverflow.com/questions/35138131/invalid-argument-error-incompatible-shapes-with-tensorflow
def print_tf_shapes():
    for k, v in locals().items():
        if type(v) is tf.Variable or type(v) is tf.Tensor:
            print("{0}: {1}".format(k, v))

# http://stackoverflow.com/questions/33681517/tensorflow-one-hot-encoder
def one_hot(labels_dense, num_classes=10):
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot

def unpickle(file):
    import cPickle
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict

def normalize(x):
    global stdR
    global stdG
    global stdB
    global meanR
    global meanG
    global meanB

    reshapped_x = x.reshape(-1, 3, 1024)
    r = reshapped_x[:, 0].flatten()
    g = reshapped_x[:, 1].flatten()
    b = reshapped_x[:, 2].flatten()

    if not stdR:
        stdR = np.std(r)
        stdG = np.std(g)
        stdB = np.std(b)
        meanR = np.mean(r)
        meanG = np.mean(g)
        meanB = np.mean(b)

    r = ((r - meanR) / stdR).reshape(-1, 32, 32)
    g = ((g - meanG) / stdG).reshape(-1, 32, 32)
    b = ((b - meanB) / stdB).reshape(-1, 32, 32)

    return np.stack((r, g, b), axis=3).reshape(-1, 3072)

def read_data_sets():
    for i in range(1, 6):
        dataMap = unpickle('./cifar-10-batches-py/data_batch_{0}'.format(i))
        trainingData['raw_x'] = np.concatenate((trainingData['raw_x'], dataMap['data']), axis=0)
        trainingData['raw_y'] = np.concatenate((trainingData['raw_y'], dataMap['labels']), axis=0)
    trainingData['x'] = normalize(trainingData['raw_x'])
    trainingData['y'] = one_hot(np.array(trainingData['raw_y']))

    dataMap = unpickle('./cifar-10-batches-py/test_batch')
    testingData['raw_x'] = dataMap['data']
    testingData['raw_y'] = dataMap['labels']
    testingData['x'] = normalize(testingData['raw_x'])
    testingData['y'] = one_hot(np.array(testingData['raw_y']))

def weight_variable(shape, stddev=0.05):
    initial = tf.random_normal(shape, stddev=stddev, dtype=tf.float32)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0, shape=shape, dtype=tf.float32)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_3x3(x):
    return tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

def eval_loss(sess):
    loss = 0.0
    for i in range(0, training_batch_count):
        startIndex = i * BATCH_SIZE
        endIndex = min(startIndex + BATCH_SIZE, TRAINING_DATA_SIZE)
        loss = loss + cross_entropy.eval(session=sess, feed_dict={
            x: trainingData['x'][startIndex: endIndex],
            y_: trainingData['y'][startIndex: endIndex],
            keep_prob: 1.0,
        })
    return loss / training_batch_count

def eval_accuracy(sess):
    acc = 0.0
    for i in range(0, testing_batch_count):
        startIndex = i * BATCH_SIZE
        endIndex = min(startIndex + BATCH_SIZE, TESTING_DATA_SIZE)
        acc = acc + accuracy.eval(session=sess, feed_dict={
            x: testingData['x'][startIndex: endIndex],
            y_: testingData['y'][startIndex: endIndex],
            keep_prob: 1.0,
        })
    return acc / testing_batch_count

def randomTransform(x):
    result = np.empty(shape=[0, 32, 32, 3])
    reshapped = x.reshape(-1, 32, 32, 3)
    padded = np.pad(reshapped, ((0, 0), (4, 4), (4, 4), (0, 0)), 'constant')

    for i, v in enumerate(padded):
        x_offset = randint(0, 8)
        y_offset = randint(0, 8)
        should_flip = bool(randint(0, 1))
        transformed = v[y_offset:y_offset + 32, x_offset:x_offset + 32, :]

        if should_flip:
            transformed = np.flip(transformed, 2)

        result = np.append(result, [transformed], axis=0)

    return result.reshape(-1, 32 * 32 * 3)

if __name__ == '__main__':
    read_data_sets()

    x = tf.placeholder(tf.float32, [None, 32 * 32 * 3])
    y_ = tf.placeholder(tf.float32, [None, 10])
    keep_prob = tf.placeholder(tf.float32)
    x_image = tf.reshape(x, [-1, 32, 32, 3])

    ## conv-1
    W_conv1 = weight_variable([5, 5, 3, 192])
    b_conv1 = tf.Variable(tf.random_normal([192], stddev=0.01, dtype=tf.float32))
    output = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

    ## mlp-1-1
    W_MLP11 = weight_variable([1, 1, 192, 160])
    b_MLP11 = bias_variable([160])
    output = tf.nn.relu(conv2d(output, W_MLP11) + b_MLP11)

    ## mlp-1-2
    W_MLP12 = weight_variable([1, 1, 160, 96])
    b_MLP12 = bias_variable([96])
    output = tf.nn.relu(conv2d(output, W_MLP12) + b_MLP12)

    ## max pooling
    output = max_pool_3x3(output)

    ## dropout
    output = tf.nn.dropout(output, keep_prob)

    ## conv-2
    W_conv2 = weight_variable([5, 5, 96, 192])
    b_conv2 = tf.Variable(tf.random_normal([192], stddev=0.01, dtype=tf.float32))
    output = tf.nn.relu(conv2d(output, W_conv2) + b_conv2)

    ## mlp-2-1
    W_MLP21 = weight_variable([1, 1, 192, 192])
    b_MLP21 = bias_variable([192])
    output = tf.nn.relu(conv2d(output, W_MLP21) + b_MLP21)

    ## mlp-2-2
    W_MLP22 = weight_variable([1, 1, 192, 192])
    b_MLP22 = bias_variable([192])
    output = tf.nn.relu(conv2d(output, W_MLP22) + b_MLP22)

    ## max pooling
    output = max_pool_3x3(output)

    ## dropout
    output = tf.nn.dropout(output, keep_prob)

    ## conv-3 layer
    W_conv3 = weight_variable([3, 3, 192, 192])
    b_conv3 = tf.Variable(tf.random_normal([192], stddev=0.01, dtype=tf.float32))
    output = tf.nn.relu(conv2d(output, W_conv3) + b_conv3)

    ## mlp-2-1
    W_MLP31 = weight_variable([1, 1, 192, 192])
    b_MLP31 = bias_variable([192])
    output = tf.nn.relu(conv2d(output, W_MLP31) + b_MLP31)

    ## mlp-2-2
    W_MLP32 = weight_variable([1, 1, 192, 10])
    b_MLP32 = bias_variable([10])
    output = tf.nn.relu(conv2d(output, W_MLP32) + b_MLP32)

    ## global average
    output = tf.nn.avg_pool(output, ksize=[1, 8, 8, 1], strides=[1, 1, 1, 1], padding='VALID')

    # [n_samples, 1, 1, 10] ->> [n_samples, 1*1*10]
    output = tf.reshape(output, [-1, 1 * 1 * 10])

    learning_rate = tf.placeholder(tf.float32)

    # the loss function
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=output))

    # optimizer SGD
    train_step = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(cross_entropy)

    # prediction
    correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # start session
    sess = tf.InteractiveSession()

    # initize variable
    sess.run(tf.global_variables_initializer())


    BREAK_POINTS = [{
        'epochs': [1, 80],
        'learning_rate': 0.1,
    }, {
        'epochs': [81, 121],
        'learning_rate': 0.01,
    }, {
        'epochs': [122, 164],
        'learning_rate': 0.001,
    }]

    filename = args.log or './log.csv'
    fd_log = open(filename, 'w')
    start_time = time.time()

    test_accuracy = eval_accuracy(sess)
    train_loss = eval_loss(sess)
    elapsed_time = time.time() - start_time
    # print initial log
    print('Epoch %d, Test accuracy %g, Train loss %s, Elapsed time %.1fs' % (
        0, test_accuracy, train_loss, elapsed_time
    ))
    fd_log.write('{0},{1},{2},{3}\n'.format(
        0, test_accuracy, train_loss, elapsed_time
    ))

    for breakPoint in BREAK_POINTS:
        epochs = breakPoint['epochs']
        # loop epochs
        for epoch in range(epochs[0], epochs[1] + 1):
            # feed through data for 1 time
            for j in range(0, training_batch_count):
                startIndex = j * BATCH_SIZE
                endIndex = min(startIndex + BATCH_SIZE, TRAINING_DATA_SIZE)
                if args.aug:
                    train_step.run(feed_dict={
                        x: randomTransform(trainingData['x'][startIndex: endIndex]),
                        y_: trainingData['y'][startIndex: endIndex],
                        keep_prob: 0.5,
                        learning_rate: breakPoint['learning_rate'],
                    })
                else:
                    train_step.run(feed_dict={
                        x: trainingData['x'][startIndex: endIndex],
                        y_: trainingData['y'][startIndex: endIndex],
                        keep_prob: 0.5,
                        learning_rate: breakPoint['learning_rate'],
                    })

            test_accuracy = eval_accuracy(sess)
            train_loss = eval_loss(sess)
            elapsed_time = time.time() - start_time
            # print log after training current epoch
            print('Epoch %d, Test accuracy %g, Train loss %s, Elapsed time %.1fs' % (
                epoch, test_accuracy, train_loss, elapsed_time
            ))
            fd_log.write('{0},{1},{2},{3}\n'.format(
                epoch, test_accuracy, train_loss, elapsed_time
            ))

    fd_log.close()
