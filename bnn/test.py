import tensorflow as tf

import kfac

class Model():

    def __init__(self, sess):
        self.w1 = tf.get_variable('w1',
                shape=(100, 100), dtype=tf.float32,
                initializer = tf.glorot_uniform_initializer() )

        self.w2 = tf.get_variable('w2',
                shape=(100, 10), dtype=tf.float32,
                initializer = tf.glorot_uniform_initializer() )
        
        self.params = [self.w1, self.w2]


        self.X = tf.placeholder(
            name='X', dtype=tf.float32,
            shape=[None, self.config.dim],
        )

        self.y = tf.placeholder(
            name='y', dtype=tf.float32,
            shape=[None, self.config.ny],
        )

        logits = self.forward(self.X, self.w1, self.w2)
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=logits))


    def forward(self, X, w1, w2):
        a1 = tf.matmul(X, w1)
        h1 = tf.nn.relu(a1)
        a2 = tf.matmul(h1, w2)
        return a2
            

if __name__ == '__main__':

        
    session_config = tf.ConfigProto(
        allow_soft_placement=True,
        gpu_options=tf.GPUOptions(allow_growth=True),
    )

    with tf.Graph().as_default(), tf.Session(config=session_config) as sess:

        model = Model()
        tf.global_variables_initializer().run()




