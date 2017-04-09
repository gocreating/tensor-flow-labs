import argparse
import math
import time
from random import randint
import numpy as np
import tensorflow as tf

parser = argparse.ArgumentParser()
parser.add_argument('--log', type=str, help='Filename of logs')
parser.add_argument("--aug", dest='aug', action='store_true', help='Apply data augmentation on training data')
parser.add_argument("--elu", dest='elu', action='store_true', help='Use elu instead of relu')
parser.add_argument("--bn", dest='bn', action='store_true', help='Use batch normalization')
parser.add_argument("--random-init", dest='ri', action='store_true', help='Use random initialization')
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

MEAN_R = 123.68
MEAN_G = 116.779
MEAN_B = 103.939

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
    reshapped_x = x.reshape(-1, 3, 1024)
    r = reshapped_x[:, 0].flatten()
    g = reshapped_x[:, 1].flatten()
    b = reshapped_x[:, 2].flatten()

    r = (r - MEAN_R).reshape(-1, 32, 32)
    g = (g - MEAN_G).reshape(-1, 32, 32)
    b = (b - MEAN_B).reshape(-1, 32, 32)

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

def weight_variable(shape, stddev=0.03):
    initial = tf.random_normal(shape, stddev=stddev, dtype=tf.float32)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def activation(x):
    if args.elu:
        return tf.nn.elu(x)
    else:
        return tf.nn.relu(x)

def batch_norm(x, is_training, scope):
    return tf.contrib.layers.batch_norm(
        x, is_training=is_training, updates_collections=None, decay=0.9, scope=scope
    )

def conditional_bn_and_act(x, b, is_training, scope):
    if args.bn:
        return batch_norm(activation(x), is_training, scope)
    else:
        return activation(x + b)

def eval_loss(sess):
    loss = 0.0
    for i in range(0, training_batch_count):
        startIndex = i * BATCH_SIZE
        endIndex = min(startIndex + BATCH_SIZE, TRAINING_DATA_SIZE)
        loss = loss + cross_entropy.eval(session=sess, feed_dict={
            x: trainingData['x'][startIndex: endIndex],
            y_: trainingData['y'][startIndex: endIndex],
            keep_prob: 1.0,
            is_training: False,
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
            is_training: False,
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
    params_dict = np.load('./vgg19.npy', encoding='latin1').item()

    x = tf.placeholder(tf.float32, [None, 32 * 32 * 3])
    y_ = tf.placeholder(tf.float32, [None, 10])
    keep_prob = tf.placeholder(tf.float32)
    is_training = tf.placeholder(tf.bool)
    x_image = tf.reshape(x, [-1, 32, 32, 3])

    # conv1_1
    W_conv1_1 = params_dict['conv1_1'][0] if not args.ri else weight_variable([3, 3, 3, 64])
    b_conv1_1 = params_dict['conv1_1'][1] if not args.ri else weight_variable([64])
    output = conditional_bn_and_act(conv2d(x_image, W_conv1_1), b_conv1_1, is_training, 'conv1_1')

    # conv1_2
    W_conv1_2 = params_dict['conv1_2'][0] if not args.ri else weight_variable([3, 3, 64, 64])
    b_conv1_2 = params_dict['conv1_2'][1] if not args.ri else weight_variable([64])
    output = conditional_bn_and_act(conv2d(output, W_conv1_2), b_conv1_2, is_training, 'conv1_2')

    # pool1
    output = max_pool_2x2(output)

    # conv2_1
    W_conv2_1 = params_dict['conv2_1'][0] if not args.ri else weight_variable([3, 3, 64, 128])
    b_conv2_1 = params_dict['conv2_1'][1] if not args.ri else weight_variable([128])
    output = conditional_bn_and_act(conv2d(output, W_conv2_1), b_conv2_1, is_training, 'conv2_1')

    # conv2_2
    W_conv2_2 = params_dict['conv2_2'][0] if not args.ri else weight_variable([3, 3, 128, 128])
    b_conv2_2 = params_dict['conv2_2'][1] if not args.ri else weight_variable([128])
    output = conditional_bn_and_act(conv2d(output, W_conv2_2), b_conv2_2, is_training, 'conv2_2')

    # pool2
    output = max_pool_2x2(output)

    # conv3_1
    W_conv3_1 = params_dict['conv3_1'][0] if not args.ri else weight_variable([3, 3, 128, 256])
    b_conv3_1 = params_dict['conv3_1'][1] if not args.ri else weight_variable([256])
    output = conditional_bn_and_act(conv2d(output, W_conv3_1), b_conv3_1, is_training, 'conv3_1')

    # conv3_2
    W_conv3_2 = params_dict['conv3_2'][0] if not args.ri else weight_variable([3, 3, 256, 256])
    b_conv3_2 = params_dict['conv3_2'][1] if not args.ri else weight_variable([256])
    output = conditional_bn_and_act(conv2d(output, W_conv3_2), b_conv3_2, is_training, 'conv3_2')

    # conv3_3
    W_conv3_3 = params_dict['conv3_3'][0] if not args.ri else weight_variable([3, 3, 256, 256])
    b_conv3_3 = params_dict['conv3_3'][1] if not args.ri else weight_variable([256])
    output = conditional_bn_and_act(conv2d(output, W_conv3_3), b_conv3_3, is_training, 'conv3_3')

    # conv3_4
    W_conv3_4 = params_dict['conv3_4'][0] if not args.ri else weight_variable([3, 3, 256, 256])
    b_conv3_4 = params_dict['conv3_4'][1] if not args.ri else weight_variable([256])
    output = conditional_bn_and_act(conv2d(output, W_conv3_4), b_conv3_4, is_training, 'conv3_4')

    # pool3
    output = max_pool_2x2(output)

    # conv4_1
    W_conv4_1 = params_dict['conv4_1'][0] if not args.ri else weight_variable([3, 3, 256, 512])
    b_conv4_1 = params_dict['conv4_1'][1] if not args.ri else weight_variable([512])
    output = conditional_bn_and_act(conv2d(output, W_conv4_1), b_conv4_1, is_training, 'conv4_1')

    # conv4_2
    W_conv4_2 = params_dict['conv4_2'][0] if not args.ri else weight_variable([3, 3, 512, 512])
    b_conv4_2 = params_dict['conv4_2'][1] if not args.ri else weight_variable([512])
    output = conditional_bn_and_act(conv2d(output, W_conv4_2), b_conv4_2, is_training, 'conv4_2')

    # conv4_3
    W_conv4_3 = params_dict['conv4_3'][0] if not args.ri else weight_variable([3, 3, 512, 512])
    b_conv4_3 = params_dict['conv4_3'][1] if not args.ri else weight_variable([512])
    output = conditional_bn_and_act(conv2d(output, W_conv4_3), b_conv4_3, is_training, 'conv4_3')

    # conv4_4
    W_conv4_4 = params_dict['conv4_4'][0] if not args.ri else weight_variable([3, 3, 512, 512])
    b_conv4_4 = params_dict['conv4_4'][1] if not args.ri else weight_variable([512])
    output = conditional_bn_and_act(conv2d(output, W_conv4_4), b_conv4_4, is_training, 'conv4_4')

    # pool4
    output = max_pool_2x2(output)

    # conv5_1
    W_conv5_1 = params_dict['conv5_1'][0] if not args.ri else weight_variable([3, 3, 512, 512])
    b_conv5_1 = params_dict['conv5_1'][1] if not args.ri else weight_variable([512])
    output = conditional_bn_and_act(conv2d(output, W_conv5_1), b_conv5_1, is_training, 'conv5_1')

    # conv5_2
    W_conv5_2 = params_dict['conv5_2'][0] if not args.ri else weight_variable([3, 3, 512, 512])
    b_conv5_2 = params_dict['conv5_2'][1] if not args.ri else weight_variable([512])
    output = conditional_bn_and_act(conv2d(output, W_conv5_2), b_conv5_2, is_training, 'conv5_2')

    # conv5_3
    W_conv5_3 = params_dict['conv5_3'][0] if not args.ri else weight_variable([3, 3, 512, 512])
    b_conv5_3 = params_dict['conv5_3'][1] if not args.ri else weight_variable([512])
    output = conditional_bn_and_act(conv2d(output, W_conv5_3), b_conv5_3, is_training, 'conv5_3')

    # conv5_4
    W_conv5_4 = params_dict['conv5_4'][0] if not args.ri else weight_variable([3, 3, 512, 512])
    b_conv5_4 = params_dict['conv5_4'][1] if not args.ri else weight_variable([512])
    output = conditional_bn_and_act(conv2d(output, W_conv5_4), b_conv5_4, is_training, 'conv5_4')

    # reshape
    output = tf.reshape(output, [-1, 2 * 2 * 512])

    # fc6
    W_fc6 = tf.Variable(tf.random_normal([2048, 4096], stddev=0.01, dtype=tf.float32))
    b_fc6 = tf.Variable(tf.random_normal([4096], stddev=0.01, dtype=tf.float32))
    output = activation(tf.matmul(output, W_fc6) + b_fc6)

    # dropout
    output = tf.nn.dropout(output, keep_prob)

    # fc7
    W_fc7 = params_dict['fc7'][0]
    b_fc7 = params_dict['fc7'][1]
    output = activation(tf.matmul(output, W_fc7) + b_fc7)

    # dropout
    output = tf.nn.dropout(output, keep_prob)

    # fc8
    W_fc8 = tf.Variable(tf.random_normal([4096, 10], stddev=0.01, dtype=tf.float32))
    b_fc8 = tf.Variable(tf.random_normal([10], stddev=0.01, dtype=tf.float32))
    output = activation(tf.matmul(output, W_fc8) + b_fc8)

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
        'learning_rate': 0.01,
    }, {
        'epochs': [81, 121],
        'learning_rate': 0.001,
    }, {
        'epochs': [122, 164],
        'learning_rate': 0.0001,
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
                        is_training: True,
                        learning_rate: breakPoint['learning_rate'],
                    })
                else:
                    train_step.run(feed_dict={
                        x: trainingData['x'][startIndex: endIndex],
                        y_: trainingData['y'][startIndex: endIndex],
                        keep_prob: 0.5,
                        is_training: True,
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
