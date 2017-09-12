from __future__ import absolute_import
import numpy as np

from keras import backend as K
from keras import activations
from keras import initializers
from keras import regularizers
from keras import constraints
from keras.engine import Layer
from keras.engine import InputSpec
from keras.legacy import interfaces
from keras.layers import Recurrent
from keras.layers.recurrent import _time_distributed_dense


class SRU(Recurrent):
    """Simple Recurrent Unit - https://arxiv.org/pdf/1709.02755.pdf.

    # Arguments
        units: Positive integer, dimensionality of the output space.
        activation: Activation function to use
            (see [activations](../activations.md)).
            If you pass None, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        recurrent_activation: Activation function to use
            for the recurrent step
            (see [activations](../activations.md)).
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix,
            used for the linear transformation of the inputs.
            (see [initializers](../initializers.md)).
        recurrent_initializer: Initializer for the `recurrent_kernel`
            weights matrix,
            used for the linear transformation of the recurrent state.
            (see [initializers](../initializers.md)).
        bias_initializer: Initializer for the bias vector
            (see [initializers](../initializers.md)).
        unit_forget_bias: Boolean.
            If True, add 1 to the bias of the forget gate at initialization.
            Setting it to true will also force `bias_initializer="zeros"`.
            This is recommended in [Jozefowicz et al.](http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf)
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        recurrent_regularizer: Regularizer function applied to
            the `recurrent_kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        bias_regularizer: Regularizer function applied to the bias vector
            (see [regularizer](../regularizers.md)).
        activity_regularizer: Regularizer function applied to
            the output of the layer (its "activation").
            (see [regularizer](../regularizers.md)).
        kernel_constraint: Constraint function applied to
            the `kernel` weights matrix
            (see [constraints](../constraints.md)).
        recurrent_constraint: Constraint function applied to
            the `recurrent_kernel` weights matrix
            (see [constraints](../constraints.md)).
        bias_constraint: Constraint function applied to the bias vector
            (see [constraints](../constraints.md)).
        dropout: Float between 0 and 1.
            Fraction of the units to drop for
            the linear transformation of the inputs.
        recurrent_dropout: Float between 0 and 1.
            Fraction of the units to drop for
            the linear transformation of the recurrent state.

    # References
        - [Long short-term memory](http://www.bioinf.jku.at/publications/older/2604.pdf) (original 1997 paper)
        - [Learning to forget: Continual prediction with LSTM](http://www.mitpressjournals.org/doi/pdf/10.1162/089976600300015015)
        - [Supervised sequence labeling with recurrent neural networks](http://www.cs.toronto.edu/~graves/preprint.pdf)
        - [A Theoretically Grounded Application of Dropout in Recurrent Neural Networks](http://arxiv.org/abs/1512.05287)
    """
    @interfaces.legacy_recurrent_support
    def __init__(self, units,
                 activation='tanh',
                 recurrent_activation='sigmoid',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 unit_forget_bias=True,
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 bias_constraint=None,
                 dropout=0.,
                 recurrent_dropout=0.,
                 **kwargs):
        super(SRU, self).__init__(**kwargs)
        self.units = units
        self.activation = activations.get(activation)
        self.recurrent_activation = activations.get(recurrent_activation)
        self.use_bias = use_bias

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.unit_forget_bias = unit_forget_bias

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.recurrent_constraint = constraints.get(recurrent_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        self.dropout = min(1., max(0., dropout))
        self.recurrent_dropout = min(1., max(0., recurrent_dropout))
        self.state_spec = [InputSpec(shape=(None, self.units)),
                           InputSpec(shape=(None, self.units))]

    def build(self, input_shape):
        if isinstance(input_shape, list):
            input_shape = input_shape[0]

        batch_size = input_shape[0] if self.stateful else None
        self.input_dim = input_shape[2]
        self.input_spec[0] = InputSpec(shape=(batch_size, None, self.input_dim))  # (timesteps, batchsize, inputdim)

        self.states = [None, None]
        if self.stateful:
            self.reset_states()

        # There may be cases where input dim does not match output units.
        # In such a case, the code in pytorch adds another set of weights
        # to bring the intermediate shape to the correct dimentions.
        # Here, I call it the `u` kernel, though it doesnt have any specific
        # implementation yet.
        self.kernel_dim = 3 if self.input_dim == self.units else 4
        self.kernel = self.add_weight(shape=(self.input_dim, self.units * self.kernel_dim),
                                      name='kernel',
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)

        if self.use_bias:
            if self.unit_forget_bias:
                def bias_initializer(shape, *args, **kwargs):
                    return K.concatenate([
                        self.bias_initializer((self.units,), *args, **kwargs),
                        initializers.Ones()((self.units,), *args, **kwargs),
                        self.bias_initializer((self.units,), *args, **kwargs),
                    ])
            else:
                bias_initializer = self.bias_initializer
            self.bias = self.add_weight(shape=(self.units * self.kernel_dim - 1,),
                                        name='bias',
                                        initializer=bias_initializer,
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None

        self.kernel_w = self.kernel[:, :self.units]
        self.kernel_f = self.kernel[:, self.units: self.units * 2]
        self.kernel_r = self.kernel[:, self.units * 2: self.units * 3]

        if self.kernel_dim == 4:
            self.kernel_u = self.kernel[:, self.units * 3: self.units * 4]
        else:
            self.kernel_u = None

        if self.use_bias:
            # self.bias_w = self.bias[:self.units]
            self.bias_f = self.bias[self.units: self.units * 2]
            self.bias_r = self.bias[self.units * 2: self.units * 3]
            if self.kernel_dim == 4:
                self.bias_u = self.bias[self.units * 3: self.units * 4]
        else:
            # self.bias_w = None
            self.bias_f = None
            self.bias_r = None
            self.bias_u = None
        self.built = True

    def preprocess_input(self, inputs, training=None):
        if self.implementation == 0:
            input_shape = K.int_shape(inputs)
            input_dim = input_shape[2]
            timesteps = input_shape[1]

            x_w = _time_distributed_dense(inputs, self.kernel_w, None,
                                          self.dropout, input_dim, self.units,
                                          timesteps, training=training)
            x_f = _time_distributed_dense(inputs, self.kernel_f, self.bias_f,
                                          self.dropout, input_dim, self.units,
                                          timesteps, training=training)
            x_r = _time_distributed_dense(inputs, self.kernel_r, self.bias_r,
                                          self.dropout, input_dim, self.units,
                                          timesteps, training=training)

            if self.kernel_dim == 4:
                x_u = _time_distributed_dense(inputs, self.kernel_u, self.bias_u,
                                              self.dropout, input_dim, self.units,
                                              timesteps, training=training)

                return K.concatenate([x_w, x_f, x_r, x_u], axis=2)
            else:
                return K.concatenate([x_w, x_f, x_r], axis=2)
        else:
            return inputs

    def get_constants(self, inputs, training=None):
        constants = []
        if self.implementation != 0 and 0 < self.dropout < 1:
            input_shape = K.int_shape(inputs)  # (timesteps, batchsize, inputdim)
            input_dim = input_shape[-1]
            ones = K.ones_like(K.reshape(inputs[:, 0, 0], (-1, 1)))
            ones = K.tile(ones, (1, int(input_dim)))

            def dropped_inputs():
                return K.dropout(ones, self.dropout)

            dp_mask = [K.in_train_phase(dropped_inputs,
                                        ones,
                                        training=training) for _ in range(3)]
            constants.append(dp_mask)
        else:
            constants.append([K.cast_to_floatx(1.) for _ in range(3)])

        if 0 < self.recurrent_dropout < 1:
            ones = K.ones_like(K.reshape(inputs[:, 0, 0], (-1, 1)))
            ones = K.tile(ones, (1, self.units))

            def dropped_inputs():
                return K.dropout(ones, self.recurrent_dropout)
            rec_dp_mask = [K.in_train_phase(dropped_inputs,
                                            ones,
                                            training=training) for _ in range(3)]
            constants.append(rec_dp_mask)
        else:
            constants.append([K.cast_to_floatx(1.) for _ in range(3)])

        constants.append(inputs)  # append the inputs so that we can utilize them in x_t
        self.time_step = 0

        return constants

    def step(self, inputs, states):
        h_tm1 = states[0]  # not used
        c_tm1 = states[1]
        dp_mask = states[2]
        rec_dp_mask = states[3]
        x_inputs = states[4]

        # To see correct batch shapes, set batch_input_shape to some value,
        # otherwise the None can be confusing to interpret.
        # print("X inputs shape : ", K.int_shape(x_inputs))
        # print('h_tm1 shape: ', K.int_shape(h_tm1))
        # print('c_tm1 shape: ', K.int_shape(c_tm1))

        if self.implementation == 2:
            z = K.dot(inputs * dp_mask[0], self.kernel)
            z = z * rec_dp_mask[0]
            if self.use_bias:
                z = K.bias_add(z, self.bias)

            z0 = z[:, :self.units]
            z1 = z[:, self.units: 2 * self.units]
            z2 = z[:, 2 * self.units: 3 * self.units]

            f = self.recurrent_activation(z1)
            r = self.recurrent_activation(z2)

            # print("W shape : ", K.int_shape(z0))
            # print("F shape : ", K.int_shape(f))
            # print("R shape : ", K.int_shape(r))
            c = f * c_tm1 + (1 - f) * z0
            h = r * self.activation(c) + (1 - r) * x_inputs[:, self.time_step, :]  # x_inputs should not have 0 index
        else:
            if self.implementation == 0:
                x_w = inputs[:, :self.units]
                x_f = inputs[:, self.units: 2 * self.units]
                x_r = inputs[:, 2 * self.units: 3 * self.units]
            elif self.implementation == 1:
                x_w = K.dot(inputs * dp_mask[0], self.kernel_w)
                x_f = K.dot(inputs * dp_mask[1], self.kernel_f) + self.bias_f
                x_r = K.dot(inputs * dp_mask[2], self.kernel_r) + self.bias_r
            else:
                raise ValueError('Unknown `implementation` mode.')

            w = x_w * rec_dp_mask[0]
            f = self.recurrent_activation(x_f)
            r = self.recurrent_activation(x_r)

            # print("W shape : ", K.int_shape(w))
            # print("F shape : ", K.int_shape(f))
            # print("R shape : ", K.int_shape(r))
            c = f * c_tm1 + (1 - f) * w
            h = r * self.activation(c) + (1 - r) * x_inputs[:, self.time_step, :]  # x_inputs should not have 0 index

        self.time_step += 1
        print('timestep : ', self.time_step)
        if 0 < self.dropout + self.recurrent_dropout:
            h._uses_learning_phase = True
        return h, [h, c]

    def get_config(self):
        config = {'units': self.units,
                  'activation': activations.serialize(self.activation),
                  'recurrent_activation': activations.serialize(self.recurrent_activation),
                  'use_bias': self.use_bias,
                  'kernel_initializer': initializers.serialize(self.kernel_initializer),
                  'recurrent_initializer': initializers.serialize(self.recurrent_initializer),
                  'bias_initializer': initializers.serialize(self.bias_initializer),
                  'unit_forget_bias': self.unit_forget_bias,
                  'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
                  'recurrent_regularizer': regularizers.serialize(self.recurrent_regularizer),
                  'bias_regularizer': regularizers.serialize(self.bias_regularizer),
                  'activity_regularizer': regularizers.serialize(self.activity_regularizer),
                  'kernel_constraint': constraints.serialize(self.kernel_constraint),
                  'recurrent_constraint': constraints.serialize(self.recurrent_constraint),
                  'bias_constraint': constraints.serialize(self.bias_constraint),
                  'dropout': self.dropout,
                  'recurrent_dropout': self.recurrent_dropout}
        base_config = super(SRU, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
