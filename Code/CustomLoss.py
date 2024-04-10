import tensorflow as tf

def manifold_canberra_loss(yTrue, yPred, reshapeSize , kSize = 2, stride = 2):

    yTrueReshaped = tf.reshape(yTrue, shape = (-1, reshapeSize , 1))
    yPredReshaped = tf.reshape(yPred, shape = (-1, reshapeSize , 1))

    true_downscaled = tf.nn.avg_pool1d(yTrueReshaped, ksize = kSize, strides = stride, padding='SAME')
    pred_downscaled = tf.nn.avg_pool1d(yPredReshaped, ksize = kSize, strides = stride, padding='SAME')


    return tf.reduce_mean(tf.reduce_sum(tf.abs(true_downscaled - pred_downscaled) / (tf.abs(true_downscaled) + tf.abs(pred_downscaled)), axis=-1))


def manifold_hamming_loss(yTrue, yPred , reshapeSize , kSize = 2, stride = 2):

    yTrueReshaped = tf.reshape(yTrue, shape = (-1, reshapeSize , 1))
    yPredReshaped = tf.reshape(yPred, shape = (-1, reshapeSize , 1))

    true_downscaled = tf.nn.avg_pool1d(yTrueReshaped, ksize = kSize, strides = stride, padding='SAME')
    pred_downscaled = tf.nn.avg_pool1d(yPredReshaped, ksize = kSize, strides = stride, padding='SAME')


    return tf.mean( true_downscaled * (1 - pred_downscaled ) + ( 1 - true_downscaled) * pred_downscaled)


def modified_canberra_loss(yTrue , yPred) :
    return tf.reduce_mean(tf.reduce_sum((yTrue + yPred) / (yTrue + yPred + 2)))

def canberra_loss(yTrue, yPred):
    tf.cast(yTrue , tf.float32)
    tf.cast(yPred , tf.float32)

    return tf.reduce_mean(tf.reduce_sum(tf.abs(yTrue - yPred) / (tf.abs(yTrue) + tf.abs(yPred)), axis=-1))   

def hamming_loss(yTrue, yPred):
    tf.cast(yTrue , tf.float32)
    tf.cast(yPred , tf.float32)
    return tf.reduce_mean( yTrue * (1 - yPred ) + ( 1 - yTrue) * yPred)

def hamming_mse(yTrue, yPred, reshapeSize , kSize = 2, stride = 2):

    return tf.reduce_mean(tf.reduce_sum(tf.cast(tf.not_equal(yTrue, yPred), tf.float32), axis=-1))



def manifold_continous_loss(yTrue, yPred , reshapeSize , kSize = 2 , stride = 2):

    # Reshape input tensors to have rank 3 

    yTrueReshaped = tf.reshape(yTrue, shape=(-1, reshapeSize , 1))
    yPredReshaped = tf.reshape(yPred, shape=(-1, reshapeSize , 1))

    # Downsample the true and predicted data by applying the average pooling operation

    true_downscaled = tf.nn.avg_pool1d(yTrueReshaped, ksize = kSize, strides = stride, padding='SAME')
    pred_downscaled = tf.nn.avg_pool1d(yPredReshaped, ksize = kSize, strides = stride, padding='SAME')

    # Compute the MSE loss between the downscaled true and predicted data

    loss = tf.reduce_mean((true_downscaled - pred_downscaled) ** 2)

    return loss

def jacobian_mse(yTrue, yPred, lambdaReg = 1e-4):
    

    # Total Loss = Reconstruction Loss + Manifold_Regularization_Loss
    
    
    # Reconstruction Loss (MSE)

    reconstruction_loss = tf.reduce_mean(tf.square(yTrue - yPred))

    # Manifold Regularization Term

    with tf.GradientTape() as tape:

        # Convert yTrue to a tensor with the same shape as yPred

        yTrue_tensor = tf.convert_to_tensor(yTrue, dtype=tf.float32)

        latent_representation = yPred  

    jacobian = tape.jacobian(latent_representation, yTrue_tensor)

    # Calculate the Frobenius norm of the Jacobian
    
    manifold_regularization_loss = lambdaReg * tf.reduce_mean(tf.square(jacobian))

    # Total Loss
    total_loss = reconstruction_loss + manifold_regularization_loss

    return total_loss
