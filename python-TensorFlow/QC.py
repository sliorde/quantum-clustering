import tensorflow as tf
import numpy as np
import time


filename = "crabs.dat"
batch = np.loadtxt(open(filename,"rb"),skiprows=1,usecols = (3,4,5,6,7))

sess = tf.Session()

data = tf.placeholder(tf.float32,shape=batch.shape)

SIGMA = tf.constant(value=10.0,shape=[1],name='sigma')
TWO = tf.constant(2.0)
TWO_SIGMA_SQUARED = tf.mul(TWO, tf.square(SIGMA))
ONE_OVER_TWO_SIGMA_SQUARED = tf.inv(TWO_SIGMA_SQUARED)

x = tf.Variable(tf.zeros([batch.shape[1]]))

squared_difference = tf.reduce_sum(tf.square(tf.sub(x,data)),reduction_indices=1,name='squared-difference')
gaussian = tf.exp(tf.mul(ONE_OVER_TWO_SIGMA_SQUARED,tf.neg(squared_difference)),name='Gaussian')
wave_function = tf.reduce_sum(gaussian,keep_dims=True,name='wave-function')
laplacian = tf.reduce_sum(tf.mul(gaussian,squared_difference),keep_dims=True,name='Laplacian')
potential = tf.div(laplacian,wave_function,name='potential')

tf.scalar_summary(['potential'],potential)
tf.histogram_summary('x',x)

optimizer = tf.train.GradientDescentOptimizer(0.05)

global_step = tf.Variable(0, name='global-step', trainable=False)

optimization_step = optimizer.minimize(potential,global_step=global_step)

summary_op = tf.merge_all_summaries()
sess.run(tf.initialize_all_variables())
summary_writer = tf.train.SummaryWriter('trainDir', sess.graph)

saver = tf.train.Saver()

for ii in range(batch.shape[0]):
    start_time = time.time()
    x.assign(data[ii, :])
    _, potential_value = sess.run([optimization_step,potential],feed_dict={data:batch})
    duration = time.time() - start_time

    if ii % 1 == 0:
        print('step #' + str(ii) + ':  potential=' + str(potential_value) + '   (' + str(duration) + 'sec)')
        summary_str = sess.run(summary_op,feed_dict={data:batch})
        summary_writer.add_summary(summary_str, ii)
        summary_writer.flush()

        saver.save(sess, 'trainDir', global_step=ii)