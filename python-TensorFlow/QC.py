import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import pickle


DATA_TYPE = tf.float32

def QC(data,sigma,replicas=None,steps=10000,step_size=None,batch_size=None,how_often_to_test_stop=10,with_display=False,same_batch_for_all=False):
    if replicas is None:
        replicas = data
    number_of_data_points = data.shape[0]
    number_of_replicas = replicas.shape[0]
    dimensions = data.shape[1]

    if step_size is None:
        step_size = sigma/7

    if batch_size is None:
        batch_size = data.shape[0]

    inds_to_move_ = tf.placeholder(dtype=tf.int32,shape=(None,),name="inds")

    data_ = tf.constant(data, name='data', dtype=DATA_TYPE)
    replicas_ = tf.get_variable(name='replicas', shape=(number_of_replicas, dimensions), dtype=DATA_TYPE,
                      initializer=tf.constant_initializer(replicas, dtype=DATA_TYPE))
    replicas_to_move_ = tf.nn.embedding_lookup(replicas_, inds_to_move_, name="replicas_to_move")
    if batch_size is None:
        batch_ = data_
        squared_distances_ = tf.reduce_sum(tf.square(tf.sub(tf.expand_dims(replicas_to_move_, 1), tf.expand_dims(batch_,0))), axis=2,name='squared_distances')
    elif same_batch_for_all is True:
        inds_for_batch_ = tf.random_uniform(shape=(1,batch_size), minval=0,
                                            maxval=number_of_data_points, dtype=tf.int32, name="inds_for_batch")
        batch_ = tf.nn.embedding_lookup(data_,inds_for_batch_,name="batch")
        squared_distances_ = tf.reduce_sum(tf.square(tf.sub(tf.expand_dims(replicas_to_move_, 1), batch_)), axis=2,name='squared_distances')
    else:
        inds_for_batch_ = tf.random_uniform(shape=(tf.shape(inds_to_move_)[0], batch_size), minval=0,
                                            maxval=number_of_data_points, dtype=tf.int32, name="inds_for_batch")
        batch_ = tf.nn.embedding_lookup(data_,inds_for_batch_,name="batch")
        squared_distances_ = tf.reduce_sum(tf.square(tf.sub(tf.expand_dims(replicas_to_move_, 1), batch_)), axis=2,
                                           name='squared_distances')
    gaussians_ = tf.exp(-1*squared_distances_/(2*sigma**2), name='gaussian')
    wave_function_ = tf.reduce_sum(gaussians_, name='wave_function',axis=1)
    laplacian_ = tf.reduce_sum(tf.mul(gaussians_, squared_distances_), name='laplacian',axis=1)
    potential_ = tf.div(laplacian_, wave_function_, name='potential')
    loss_ = tf.reduce_sum(potential_)
    # optimizer_ = tf.train.MomentumOptimizer(learning_rate=step_size,momentum=0.9,use_nesterov=True)
    optimizer_ = tf.train.GradientDescentOptimizer(learning_rate=step_size,name='optimizer')
    # optimization_step_ = optimizer_.minimize(loss_,name='optimization_step')
    gradients_ = tf.gradients(loss_,replicas_,name='gradients')
    normalized_gradients_ = tf.nn.l2_normalize(gradients_[0] ,dim=1)
    optimization_step_ = optimizer_.apply_gradients([(normalized_gradients_,replicas_)])
    #     tf.reduce_sum(tf.square(gradients_),axis=1,name="gradient_norms")

    if with_display:
        plt.ion()
        plt.figure()
        ax1 = plt.axes()
        sc = plt.scatter(replicas[:,0],replicas[:,1])
        plt.axis([-2,2,-2,2])
        # plt.figure()
        # ax2 = plt.axes()


    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        step = 0
        inds = np.arange(0,replicas.shape[0])
        previous_potential = np.zeros((replicas.shape[0]))
        previous_potential.fill(np.inf)
        while step<steps:
            print(step)

            _,potential_value = sess.run([optimization_step_,potential_], feed_dict={inds_to_move_: inds})
            if ((step%how_often_to_test_stop)==0):
                prev_inds = inds
                inds = inds[previous_potential[inds] > potential_value]
                previous_potential[prev_inds] = potential_value
                if len(inds) == 0:
                    break

            if with_display:
                x = sess.run(replicas_)
                sc.set_offsets(x[:,:2])
                # plt.sca(ax2)
                # if np.any(prev_inds==0):
                #     plt.scatter(step,np.mean(potential_values_for_moving_average[0,:]))
                #     plt.scatter(step, potential_value[prev_inds==0],c='r')
                plt.pause(0.0001)

            step += 1
        if with_display:
            plt.ioff()
            plt.show()

        x = sess.run(replicas_)
        return x




if __name__ == "__main__":
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    data,y = mnist.train.next_batch(55000)

    # np.random.seed(111)
    # data = np.random.randn(1000,2)

    x = QC(data,2000,step_size=100,batch_size=1,same_batch_for_all=True,with_display=False,how_often_to_test_stop=1000)
    with open("out.pickle", 'wb') as f:
        pickle.dump(x, f)


