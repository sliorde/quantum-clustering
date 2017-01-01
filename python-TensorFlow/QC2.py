import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt


DATA_TYPE = tf.float32

def QC(data,sigma,replicas=None,steps=10000,step_size=None,batch_size=None,how_often_to_test_stop=10,with_display=False):
    if replicas is None:
        replicas = data
    number_of_data_points = data.shape[0]
    number_of_replicas = replicas.shape[0]
    dimensions = data.shape[1]

    if step_size is None:
        step_size = sigma/7

    data_ = tf.constant(data, name='data', dtype=DATA_TYPE)
    replicas_ = [None for i in range(number_of_replicas)]
    inds_for_batch_ = [None for i in range(number_of_replicas)]
    batch_ = [None for i in range(number_of_replicas)]
    squared_distances_ = [None for i in range(number_of_replicas)]
    gaussians_ = [None for i in range(number_of_replicas)]
    wave_function_ = [None for i in range(number_of_replicas)]
    laplacian_ = [None for i in range(number_of_replicas)]
    potential_ = [None for i in range(number_of_replicas)]
    step_size_ = [None for i in range(number_of_replicas)]
    optimizer_ = [None for i in range(number_of_replicas)]
    gradient_ = [None for i in range(number_of_replicas)]
    normalized_gradient_= [None for i in range(number_of_replicas)]
    optimization_step_ = [None for i in range(number_of_replicas)]
    for i in range(number_of_replicas):
        print('building ' + str(i))
        replicas_[i] = tf.get_variable(name='replicas'+str(i), shape=(1, dimensions), dtype=DATA_TYPE,
                      initializer=tf.constant_initializer(replicas[i], dtype=DATA_TYPE))
        if batch_size is None:
            batch_[i] = data_
        else:
            inds_for_batch_[i] = tf.random_uniform(shape=(batch_size,),minval=0,maxval=number_of_data_points,dtype=tf.int32,name="inds_for_batch"+str(i))
            batch_[i] = tf.nn.embedding_lookup(data_,inds_for_batch_[i],name="batch"+str(i))
        squared_distances_[i] = tf.reduce_sum(tf.square(tf.sub(replicas_[i],batch_[i])), axis=1,name='squared_distances'+str(i))
        gaussians_[i] = tf.exp(-1 * squared_distances_[i] / (2 * sigma ** 2), name='gaussian'+str(i))
        wave_function_[i] = tf.reduce_sum(gaussians_[i], name='wave_function'+str(i))
        laplacian_[i] = tf.reduce_sum(tf.mul(gaussians_[i], squared_distances_[i]), name='laplacian'+str(i))
        potential_[i] = tf.div(laplacian_[i], wave_function_[i], name='potential'+str(i))
        step_size_[i] = tf.placeholder_with_default(step_size,shape=(),name='step_size'+str(i))
        # optimizer_[i] = tf.train.MomentumOptimizer(learning_rate=step_size_[i],momentum=0.9,use_nesterov=True)
        optimizer_[i] = tf.train.GradientDescentOptimizer(learning_rate=step_size_[i], name='optimizer'+str(i))
        gradient_[i] = tf.gradients(potential_[i], replicas_[i], name='gradients'+str(i))
        normalized_gradient_[i] = tf.nn.l2_normalize(gradient_[i][0], dim=1,name='normalized_gradients'+str(i))
        optimization_step_[i] = optimizer_[i].apply_gradients([(normalized_gradient_[i], replicas_[i])],name='optimization_step'+str(i))
        # optimization_step_[i] = optimizer_[i].minimize(potential_[i])

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
        inds = np.arange(0,number_of_replicas,dtype=np.int)
        previous_potential = np.zeros((number_of_replicas))
        previous_potential.fill(np.inf)
        while step<steps:
            print(step)
            out = sess.run([optimization_step_[i] for i in inds]+[potential_[i] for i in inds])
            potential_value = out[len(inds):]
            if ((step%how_often_to_test_stop)==0):
                prev_inds = inds
                inds = inds[previous_potential[inds] > potential_value]
                previous_potential[prev_inds] = potential_value
                if len(inds) == 0:
                    break

            step += 1

            if with_display:
                x = sess.run(replicas_)
                x = np.concatenate(x,axis=0)
                sc.set_offsets(x[:,:2])
                # plt.sca(ax2)
                # if np.any(prev_inds==0):
                #     plt.scatter(step,np.mean(potential_values_for_moving_average[0,:]))
                #     plt.scatter(step, potential_value[prev_inds==0],c='r')
                plt.pause(0.0001)
    if with_display:
        plt.ioff()
        plt.show()




if __name__ == "__main__":
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    data,y = mnist.train.next_batch(55000)

    # np.random.seed(111)
    # data = np.random.randn(150,2)

    QC(data,2000,with_display=False,batch_size=100)

