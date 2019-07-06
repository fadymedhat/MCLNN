# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division

import numpy as np
import tensorflow as tf

from keras import initializers
from keras import activations, regularizers, constraints
from keras import backend as K
from keras.engine import InputSpec, Layer

class MaskedConditional(Layer):
    '''Masked Conditional Neural Network layer.

    # Arguments
        output_dim: int > 0.
        init: name of initialization function for the weights of the layer
            (see Keras [initializations](../initializations.md)),
            or alternatively, Theano function to use for weights
            initialization. This parameter is only relevant
            if you don't pass a `weights` argument.
        activation: name of activation function to use
            (see Keras [activations](../activations.md)),
            or alternatively, elementwise Theano function.
            If you don't specify anything, no activation is applied
            (ie. "linear" activation: a(x) = x).
        order: frames in a single temporal direction of an MCLNN
        bandwidth: consecutive 1's to enable features in a single feature vector.
        overlap: overlapping distance between two neighbouring hidden nodes.
        input_dim: dimensionality of the input (integer).
            This argument (or alternatively, the keyword argument `input_shape`)
            is required when using this layer as the first layer in a model.
    # Input shape
        3D tensor with shape:  (nb_samples, input_row,input_dim)`.
    # Output shape
        3D tensor with shape:  (nb_samples, output_row,output_dim)`.
    '''

    def __init__(self, output_dim, init='glorot_uniform', activation='linear',
                 weights=None, order=None, bandwidth=None, overlap=None, layer_is_masked=True,
                 W_regularizer=None, b_regularizer=None, activity_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 input_dim=None, **kwargs):

        self.init = initializers.get(init) # m
        self.activation = activations.get(activation)
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.order = order

        print('INPUT DIM  ', self.input_dim, 'OUTPUT DIM  ', self.output_dim)

        self.bandwidth = bandwidth
        self.overlap = overlap
        self.layer_is_masked=layer_is_masked

        # --K_START -- Refer to keras documentation for the below.
        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.initial_weights = weights
        # --K_END -- Refer to keras documentation for the above.

        self.input_spec = [InputSpec(ndim=3)]

        if self.input_dim:
            kwargs['input_shape'] = (self.input_dim[0],self.input_dim[1],)
        super(MaskedConditional, self).__init__(**kwargs)


    def construct_mask(self, feature_count, hidden_count, bandwidth, overlap, layer_is_masked):

        bw = bandwidth
        ov = overlap
        l = feature_count
        e = hidden_count

        a = np.arange(1, bw + 1)
        g = np.arange(1, int(np.ceil((l * e) / (l + bw - ov))) + 1)

        if layer_is_masked is False:
            binary_mask = np.ones([l, e])
        else:
            mask = np.zeros([l, e])
            flat_matrix = mask.flatten('F')

            for i in range(len(a)):
                for j in range(len(g)):
                    lx = a[i] + (g[j] - 1) * (l + bw - ov)
                    if lx <= l * e:
                        flat_matrix[lx - 1] = 1

            binary_mask = np.transpose(flat_matrix.reshape(e, l))

        return binary_mask.astype(np.float32)

    def build(self, input_shape):
        assert len(input_shape) == 3
        input_dim = input_shape[2]
        self.input_spec = [InputSpec(dtype=K.floatx(),
                                     shape=(None, input_shape[1], input_dim))]

        print('self.input_spec : ', self.input_spec )

        # Create a trainable weight variable for this layer.
        self.W = self.add_weight(name='kernel',
                                      shape=(self.order * 2 + 1, input_dim, self.output_dim),
                                      initializer='uniform',
                                      trainable=True)

        self.b = self.add_weight(name='bias',
                                    shape=(self.output_dim,),
                                    initializer='uniform',
                                    trainable=True)

        self.weightmask = self.construct_mask(feature_count=input_dim,
                                              hidden_count=self.output_dim,
                                              bandwidth=self.bandwidth,
                                              overlap=self.overlap,
                                              layer_is_masked=self.layer_is_masked)

        # --K_START -- Refer to keras documentation for the below.
        self.regularizers = []
        if self.W_regularizer:
            self.W_regularizer.set_param(self.W)
            self.regularizers.append(self.W_regularizer)

        if self.b_regularizer:
            self.b_regularizer.set_param(self.b)
            self.regularizers.append(self.b_regularizer)

        if self.activity_regularizer:
            self.activity_regularizer.set_layer(self)
            self.regularizers.append(self.activity_regularizer)

        self.constraints = {}
        if self.W_constraint:
            self.constraints[self.W] = self.W_constraint
        if self.b_constraint:
            self.constraints[self.b] = self.b_constraint

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        # --K_START -- Refer to keras documentation for the above.

        super(MaskedConditional, self).build([self.order * 2 + 1,input_shape])  # Be sure to call this at the end


    def func(self, a, sequences):

        with tf.name_scope('mask'):
            mask = tf.convert_to_tensor(self.weightmask, dtype='float32' )
        with tf.name_scope('concatenated_segments'):
            samples = self.concatenated_segments

        with tf.name_scope('indices'):
            indices = sequences[1]

        with tf.name_scope('weights'):
            weights = sequences[0]

        with tf.name_scope('frames_sliced'):
            frames_sliced = tf.gather(samples, indices)

        with tf.name_scope('masked_weights'):
            masked_weights = weights * mask

        with tf.name_scope('frames_dot_weights'):
            frames_dot_weights = tf.tensordot(frames_sliced, masked_weights, [[1], [0]])

        with tf.name_scope('accumulate'):
            accumulate = tf.add(frames_dot_weights, a)
        return accumulate

    def call(self, mini_batch, mask=None):

        segment_count = tf.shape(mini_batch)[0]
        segment_length = mini_batch.shape[1]
        feature_count = mini_batch.shape[2]

        with tf.name_scope('Concatenated_Segements'):
            self.concatenated_segments = tf.reshape(mini_batch, [segment_count * segment_length, feature_count])

        with tf.name_scope('index_preparation'):
            with tf.name_scope('frames_index_in_minibatch'):
                # number of frames after concatenating the minibatch samples
                frame_count_per_minibatch = tf.shape(self.concatenated_segments)[0]
                # index vector for all the frames
                frames_index_per_minibatch = tf.range((frame_count_per_minibatch))

            with tf.name_scope('remove_twice_order_indices'):
                # reshaping the index vector to minibatch * segment_length
                frames_index_per_segment_matrix = tf.reshape(frames_index_per_minibatch, [segment_count,
                                                                                  segment_length])

                # remove the columns corresponding to the order from the index matrix
                # this ensures that n frames will remain when processing the frame at
                # position [q - (n+1)], where 1 is the window's middle frame
                frames_index_per_segment_trimmed_matrix = frames_index_per_segment_matrix[:, : -self.order * 2]

            with tf.name_scope('trimmed_indices_tiling'):
                # reshaping the index matrix after trimming back to a vector
                frames_index_per_minibatch_trimmed_flattened = tf.reshape(frames_index_per_segment_trimmed_matrix, [-1])

                # repeating the flat index a number of times equal to the 2 x order + 1
                frames_index_per_minibatch_trimmed_flattened = tf.expand_dims(frames_index_per_minibatch_trimmed_flattened, 0)
                frames_index_per_minibatch_trim_flat_tile = tf.tile(frames_index_per_minibatch_trimmed_flattened,
                                                                    (self.order * 2 + 1, 1))

            with tf.name_scope('window_index'):

                trimmed_vector_length = tf.shape(frames_index_per_minibatch_trimmed_flattened)[1]
                order_increments = tf.range((self.order * 2 + 1))

                order_increments = tf.expand_dims(order_increments, 0)
                order_increments_tile = tf.tile(order_increments, (trimmed_vector_length, 1))

                order_increments_tile_transpose = tf.transpose(order_increments_tile)

                # the window_index will have a 2n+1 rows, each row has indices of the frames of the segments
                window_index = order_increments_tile_transpose + frames_index_per_minibatch_trim_flat_tile


        result = tf.scan(
            fn=self.func,
            elems=(self.W, window_index),
            initializer=tf.zeros([ tf.shape(window_index)[1], self.W.shape[2]],dtype='float32'),
            parallel_iterations=1,
            back_prop=True,
            swap_memory=False,
            infer_shape=True,
            reverse=False,
            name='mclnn_scan'
        )

        result = result[-1]
        result = tf.add(result,self.b)
        activation_input = tf.reshape(result, [segment_count, segment_length - self.order * 2, result.shape[1]])
        return self.activation(activation_input)

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 3
        return (input_shape[0],input_shape[1] - self.order * 2, self.output_dim)





