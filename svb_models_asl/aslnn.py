"""
Inference forward models for ASL data
"""
import random
import os
import os.path

try:
    import tensorflow.compat.v1 as tf
except ImportError:
    import tensorflow as tf

import numpy as np
from fabber import Fabber

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

from svb import __version__
from svb.model import Model, ModelOption, ValueList
from .aslrest import AslRestModel
from svb import DataModel
from svb.parameter import get_parameter
import svb.dist as dist
import svb.prior as prior

class SinglePLDModel():
    """
    NN model for evaluating ASL signal at single PLD
    """
    def __init__(self, load_dir=None):
        self.variable_weights = []
        self.variable_biases = []
        if load_dir is not None:
            print("Loading model from %s" % load_dir)
            self.trained_weights = []
            self.trained_biases = []
            idx = 0
            while 1:
                weights_file = os.path.join(load_dir, "weights%i.npy" % idx)
                biases_file = os.path.join(load_dir, "biases%i.npy" % idx)
                if not os.path.exists(weights_file) and not os.path.exists(biases_file):
                    print("Failed to open weights/biases for layer %i" % idx)
                    break
                elif not os.path.exists(weights_file) or not os.path.exists(biases_file):
                    raise RuntimeError("For time point %i, could not find both weights and biases")
                else:
                    self.trained_weights.append(np.load(weights_file))
                    self.trained_biases.append(np.load(biases_file))
                idx += 1
            print("Loaded %i layers" % len(self.trained_weights))
        else:
            self.trained_weights = None
            self.trained_biases = None
            
    def _create_nn(self, x, trainable=True):
        """
        Create the network

        :param x: Input array (ftiss/delttiss)
        :param trainable: If True, generate trainable network with variable weights biases.
                          If False, generated fixed network using previously trained weights/biases
        """
        layers = []
        layers.append(self._add_layer(0, x, 2, 10, activation_function=tf.nn.tanh, trainable=trainable))
        layers.append(self._add_layer(1, layers[-1], 10, 10, activation_function=tf.nn.tanh, trainable=trainable))
        layers.append(self._add_layer(2, layers[-1], 10, 1, activation_function=None, trainable=trainable))
        return layers

    def _add_layer(self, idx, inputs, in_size, out_size, activation_function=None, trainable=True):
        if trainable:
            weights = tf.Variable(tf.random_normal([in_size, out_size]))
            biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
            self.variable_weights.append(weights)
            self.variable_biases.append(biases)
        elif self.trained_weights is None:
            raise RuntimeError("Tried to create non-trainable network for evaluation before trainable network has been trained")
        else:
            weights = tf.constant(self.trained_weights[idx])
            biases = tf.constant(self.trained_biases[idx])
            
        Wx_plus_b = tf.matmul(inputs, weights) + biases
        if activation_function is None:  
            outputs = Wx_plus_b
        else:  
            outputs = activation_function(Wx_plus_b)
        return outputs  

    def train(self, x_train, y_train, steps, learning_rate):
        """
        Train the model with ASL data

        :param x_train: Training X values (ftiss, delttiss)
        :param y_train: Training Y values (ASL signal / delta-M)
        """
        print(x_train.shape, y_train.shape)
        graph = tf.Graph()
        with graph.as_default():
            x_input = tf.placeholder(tf.float32, [None, 2])
            y_input = tf.placeholder(tf.float32, [None, 1])
            layers = self._create_nn(x_input, trainable=True)
            prediction = layers[-1]
            loss = tf.reduce_mean(tf.reduce_sum(tf.square(y_input - prediction), reduction_indices=[1]))
            optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

            sess = tf.Session()
            sess.run(tf.global_variables_initializer())
            n_batches = 100
            for step in range(steps):
                mean_loss = 0
                for batch in range(n_batches):
                    x = x_train[batch:-1:n_batches, :]
                    y = y_train[batch:-1:n_batches, :]

                    batch_loss, optimizer_ = sess.run([loss, optimizer], feed_dict={x_input: x, y_input: y})
                    mean_loss += batch_loss
                mean_loss = mean_loss / n_batches
                if step % 100 == 0:
                    print(step, mean_loss)

            self.trained_weights = []
            self.trained_biases = []
            for weights, biases in zip(self.variable_weights, self.variable_biases):
                self.trained_weights.append(sess.run(weights))
                self.trained_biases.append(sess.run(biases))

    def evaluate(self, x):
        """
        Evaluate the trained model

        :param x: X values (ftiss, delttiss)
        :return: tensor operation containing Y values (ASL signal / delta-M)
        """
        layers = self._create_nn(x, trainable=False)
        return layers[-1]

    def ievaluate(self, x):
        """
        Evaluate the trained model interactively 
        (i.e. return the actual answer as a Numpy array not
        as a TensorFlow operation)

        :param x: X values (ftiss, delttiss)
        :return: Numpy array containing Y values (ASL signal / delta-M)
        """
        with tf.Graph().as_default():
            prediction = self.evaluate(x)
            sess = tf.Session()
            sess.run(tf.global_variables_initializer())
            return sess.run(prediction)

    def save(self, save_dir):
        """
        Save training weights/biases

        :param dir: Save directory
        """
        if self.trained_weights is None:
            raise RuntimeError("Can't save model before it has been trained!")
        
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        for idx, wb in enumerate(zip(self.trained_weights, self.trained_biases)):
            np.save(os.path.join(save_dir, "weights%i.npy" % idx), wb[0])
            np.save(os.path.join(save_dir, "biases%i.npy" % idx), wb[1])
        
class AslNNModel(Model):
    """
    ASL resting state model using NN for evaluation, trained using Fabber
    """
    OPTIONS = [
        ModelOption("tau", "Bolus duration", units="s", clargs=("--tau", "--bolus"), type=float, default=1.8),
        ModelOption("casl", "Data is CASL/pCASL", type=bool, default=False),
        ModelOption("att", "Bolus arrival time", units="s", type=float, default=1.3),
        ModelOption("attsd", "Bolus arrival time prior std.dev.", units="s", type=float, default=None),
        ModelOption("t1", "Tissue T1 value", units="s", type=float, default=1.3),
        ModelOption("t1b", "Blood T1 value", units="s", type=float, default=1.65),
        ModelOption("tis", "Inversion times", units="s", type=ValueList(float)),
        ModelOption("plds", "Post-labelling delays (for CASL instead of TIs)", units="s", type=ValueList(float)),
        ModelOption("repeats", "Number of repeats - single value or one per TI/PLD", units="s", type=ValueList(int), default=1),
        ModelOption("slicedt", "Increase in TI/PLD per slice", units="s", type=float, default=0),
        ModelOption("pc", "Blood/tissue partition coefficient", type=float, default=0.9),
        ModelOption("fcalib", "Perfusion value to use in estimation of effective T1", type=float, default=0.01),
        ModelOption("train_lr", "Training learning rate", type=float, default=0.001),
        ModelOption("train_steps", "Training steps", type=int, default=30000),
        ModelOption("train_examples", "Number of training examples", type=int, default=500),
        ModelOption("train_save", "Directory to save trained model weights to"),
        ModelOption("train_load", "Directory to load trained model weights from"),
    ]

    def __init__(self, data_model, **options):
        Model.__init__(self, data_model, **options)
        if self.plds is not None:
            self.tis = [self.tau + pld for pld in self.plds]
        if self.attsd is None:
            self.attsd = 1.0 if len(self.tis) > 1 else 0.1
        if isinstance(self.repeats, int):
            self.repeats = [self.repeats]
        if len(self.repeats) == 1:
            # FIXME variable repeats
            self.repeats = self.repeats[0]

        self.params = [
            get_parameter("ftiss", dist="FoldedNormal", 
                          mean=0.0, prior_var=1e6, post_var=1.0, 
                          post_init=self._init_flow,
                          **options),
            get_parameter("delttiss", dist="FoldedNormal", 
                          mean=self.att, var=self.attsd**2,
                          **options)
        ]

        self.ti_models = []
        if self.train_load:
            for idx in range(len(self.tis)):
                self.ti_models.append(SinglePLDModel(os.path.join(self.train_load, "t%i" % idx)))
        else:
            # Train model using Fabber
            x_train, x_test, y_train, y_test = self._get_training_data_svb(n=self.train_examples)
            for idx, ti in enumerate(self.tis):
                model = SinglePLDModel()
                self.ti_models.append(model)
                y_train_ti = np.expand_dims(y_train[:, idx], 1)
                y_test_ti = np.expand_dims(y_test[:, idx], 1)
                model.train(x_train, y_train_ti, self.train_steps, self.train_lr)
                y_pred = model.ievaluate(x_test)
                accuracy = r2_score(y_pred, y_test_ti)
                self.log.info('Trained model for TI: %.3f using %i steps and %.5f learning rate - accuracy %.3f' % (ti, self.train_steps, self.train_lr, accuracy))
                if self.train_save:
                    model.save(os.path.join(self.train_save, "t%i" % idx))

    def _get_training_data(self, n, **fabber_options):
        """
        Generate training data by evaluating Fabber model
        """
        self.log.info("Generating %i instances of training data" % n)
        fab = Fabber()
        options = {
            "model" : "buxton",
            'lambda': 0.9,
            'tau' : self.tau,
            'ti' : self.tis,
            't1b': self.t1b,
            "prior-noise-stddev" : 1,
            'casl': True,
            'repeats': 1,
            't1': self.t1,
        }
        options.update(**fabber_options)
        print(options)

        params = {}
        n_ti = len(self.tis)
        fit_data = np.zeros((n, 2 + n_ti), dtype=np.float32)
        last_percent = -1
        for idx in range(n):
            ftiss = random.uniform(0.0, 20)
            delttiss = random.uniform(1.2, 1.4)
            params["ftiss"] = ftiss
            params['delttiss'] = delttiss
            output = fab.model_evaluate(options, params, nvols=len(self.tis))
            fit_data[idx,0] = ftiss
            fit_data[idx,1] = delttiss
            fit_data[idx,2:(3 + n_ti)] = np.array((output))
            percent = int(idx * 100 / n)
            if percent % 10 == 0 and percent != last_percent:
                self.log.info("%3d%%" % percent)
                last_percent = percent

        self.log.info("DONE")

        # Split training data into training and test data sets
        x_train, x_test, y_train, y_test = train_test_split(fit_data[:,0:2], fit_data[:,2:8], test_size=0.3)
        self.log.info("Separated %i instances of training data and %i instances of test data" % (x_train.shape[0], x_test.shape[0]))
        return x_train, x_test, y_train, y_test

    def _get_training_data_svb(self, n, **fabber_options):
        """
        Generate training data by evaluating SVB model
        """
        self.log.info("Generating %i instances of training data" % n)
        options = {
            "model" : "buxton",
            'lambda': 0.9,
            'tau' : self.tau,
            'ti' : self.tis,
            't1b': self.t1b,
            "prior-noise-stddev" : 1,
            'casl': True,
            'repeats': 1,
            't1': self.t1,
        }
        options.update(**fabber_options)
        print(options)

        sig = np.zeros((1, len(self.tis)), dtype=np.float32)
        data_model = DataModel(sig)
        model = AslRestModel(data_model, tis=self.tis, **options)
        ftiss = np.random.uniform(0, 20, size=(n,))
        delttiss = np.random.uniform(1.2, 1.4, size=(n,))
        tpts = np.zeros((n, len(self.tis)), dtype=np.float32)
        tpts[..., :] = self.tis
        params = np.zeros((2, n, 1), dtype=np.float32)
        params[0, :, 0] = ftiss
        params[1, :, 0] = delttiss
        modelsig = model.ievaluate(params, tpts)
        print("Generated %i instances of test data" % n)

        self.log.info("DONE")

        # Split training data into training and test data sets
        x = np.zeros((n, 2), dtype=np.float32)
        x[:, 0] = ftiss
        x[:, 1] = delttiss
        x_train, x_test, y_train, y_test = train_test_split(x, modelsig, test_size=0.3)
        self.log.info("Separated %i instances of training data and %i instances of test data" % (x_train.shape[0], x_test.shape[0]))
        return x_train, x_test, y_train, y_test

    def evaluate(self, params, tpts):
        """
        Basic PASL/pCASL kinetic model

        :param t: Sequence of time values of length N
        :param params Sequence of parameter values arrays, one for each parameter.
                      Each array is MxS tensor where M is the number of voxels and S
                      the number of samples. This may be supplied as a PxMxN tensor 
                      where P is the number of parameters.

        :return: MxSxN tensor containing model output at the specified time values
                 and for each sample using the specified parameter values
        """
        # Extract parameter tensors
        #
        # m = number of 'voxels'
        # n = samples (correspond to timepoints in t)
        t = self.log_tf(tpts, name="tpts", shape=True)
        ftiss = self.log_tf(params[0], name="ftiss", shape=True) # shape [m, n]
        delt = self.log_tf(params[1], name="delt", shape=True) # shape [m, n]
        reshaped = self.log_tf(tf.stack([ftiss, delt], axis=2), shape=True)

        opt_param_idx = 2
        t1 = self.t1

        # NN evaluation goes here!
        # Note that for now t had better match the TIs we trained with!
        ti_signals = []
        for ti_idx, ti_model in enumerate(self.ti_models):
            ti_signal = ti_model.evaluate(reshaped)
            for rpt_idx in range(self.repeats):
                ti_signals.append(ti_signal)
        signal = tf.squeeze(tf.stack(ti_signals, axis=2))
        return self.log_tf(signal, name="asl_signal", shape=True)
        #return ftiss

    def tpts(self):
        if self.data_model.n_tpts != len(self.tis) * self.repeats:
            raise ValueError("ASL model configured with %i time points, but data has %i" % (len(self.tis)*self.repeats, self.data_model.n_tpts))

        # FIXME assuming grouped by TIs/PLDs
        if self.slicedt > 0:
            # Generate voxelwise timings array using the slicedt value
            t = np.zeros(list(self.data_model.shape) + [self.data_model.n_tpts])
            for z in range(self.data_model.shape[2]):
                t[:, :, z, :] = np.array(sum([[ti + z*self.slicedt] * self.repeats for ti in self.tis], []))
        else:
            # Timings are the same for all voxels
            t = np.array(sum([[ti] * self.repeats for ti in self.tis], []))
        return t.reshape(-1, self.data_model.n_tpts)

    def __str__(self):
        return "ASL neural network model: %s" % __version__

    def _init_flow(self, _param, _t, data):
        """
        Initial value for the flow parameter
        """
        return tf.reduce_mean(data, axis=1), None

    def _init_fblood(self, _param, _t, data):
        """
        Initial value for the fblood parameter
        """
        return tf.reduce_mean(data, axis=1), None
