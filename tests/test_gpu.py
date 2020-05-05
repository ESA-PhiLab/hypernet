import tensorflow as tf


class TestGPU:
    def test_run_on_gpu(self):
        with tf.device('/gpu:0'):
            a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3],
                            name='a')
            b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2],
                            name='b')
            c = tf.matmul(a, b)
        with tf.Session() as sess:
            print('elo')
            print(sess.run(c))
            print('elo2')

        print('elo3')
        sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
        print(sess)