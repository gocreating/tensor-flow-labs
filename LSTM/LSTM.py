import sys
import argparse
import time
import numpy as np
import tensorflow as tf

parser = argparse.ArgumentParser()
parser.add_argument('--log', type=str, help='Filename of logs')
parser.add_argument('--train-length', dest='train_length', type=int, help='training sequence length')
parser.add_argument('--test-length', dest='test_length', type=int, help='testing sequence length')
args = parser.parse_args()

BATCH_SIZE = 64
HIDDEN_SIZE = 500
EMBEDDING_SIZE = 100

def get_batch(seq_length, right_padding_length=0, batch_size=BATCH_SIZE):
    seqences = np.random.randint(
        1,
        257,
        size=(seq_length - right_padding_length) * batch_size
    )
    seqences = seqences.reshape(batch_size, seq_length - right_padding_length)
    seqences = np.pad(
        seqences,
        ((0, 0), (0, right_padding_length)),
        'constant',
        constant_values=0
    )

    return {
        'x': seqences,
        'y': seqences[:],
    }

if __name__ == '__main__':
    seq_length = max(args.train_length, args.test_length)
    pad_length = abs(args.train_length - args.test_length)

    enc_inp = [
        tf.placeholder(tf.int32, shape=(BATCH_SIZE,), name="inp%i" % t)
        for t in range(seq_length)
    ]
    labels = [
        tf.placeholder(tf.int32, shape=(BATCH_SIZE,), name="labels%i" % t)
        for t in range(seq_length)
    ]
    weights = [
        tf.ones_like(labels_t, dtype=tf.float32)
        for labels_t in labels
    ]
    dec_inp = [np.zeros(BATCH_SIZE, dtype=np.int) for t in range(seq_length)]

    cell = tf.contrib.rnn.LSTMCell(HIDDEN_SIZE)

    dec_outputs, dec_memory = tf.contrib.legacy_seq2seq.embedding_rnn_seq2seq(
        enc_inp, dec_inp, cell, 257, 257, EMBEDDING_SIZE, output_projection=None, feed_previous=False
    )

    learning_rate = tf.placeholder(tf.float32)

    loss = tf.contrib.legacy_seq2seq.sequence_loss(
        dec_outputs, labels, weights, 257
    )

    train_step = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(loss)

    correct_prediction = tf.equal(
        tf.transpose(enc_inp),
        tf.cast(tf.transpose(tf.argmax(dec_outputs, 2)), tf.int32)
    )
    each_accuracy = tf.reduce_mean(tf.cast(correct_prediction[:, 0:args.test_length], tf.float32), 1)
    accuracy = tf.reduce_mean(each_accuracy)

    # start session
    config = tf.ConfigProto(device_count={ 'GPU': 0 })
    sess = tf.InteractiveSession(config=config)

    # initize variable
    sess.run(tf.global_variables_initializer())

    BREAK_POINTS = [{
        'iterations': [1, 1000],
        'learning_rate': 1,
    }, {
        'iterations': [1001, 5000],
        'learning_rate': 0.1,
    }, {
        'iterations': [5001, 10000],
        'learning_rate': 0.01,
    }]

    filename = args.log or './log.csv'
    fd_log = open(filename, 'w')
    start_time = time.time()

    for breakPoint in BREAK_POINTS:
        iterations = breakPoint['iterations']
        # loop iterations
        for iteration in range(iterations[0], iterations[1] + 1):
            batch = get_batch(seq_length, pad_length)
            feed_dict = { learning_rate: breakPoint['learning_rate'] }
            feed_dict.update({ enc_inp[t]: batch['x'][:, t] for t in range(seq_length) })
            feed_dict.update({ labels[t]: batch['y'][:, t] for t in range(seq_length) })
            train_step.run(feed_dict=feed_dict)

            test_accuracy = accuracy.eval(session=sess, feed_dict=feed_dict)
            train_loss = loss.eval(session=sess, feed_dict=feed_dict)
            elapsed_time = time.time() - start_time

            print('Iteration %05d, Test accuracy %.6f, Train loss %.5f, Elapsed time %.1fs' % (
                iteration, test_accuracy, train_loss, elapsed_time
            ))
            fd_log.write('{0},{1},{2},{3}\n'.format(
                iteration, test_accuracy, train_loss, elapsed_time
            ))

    fd_log.close()
