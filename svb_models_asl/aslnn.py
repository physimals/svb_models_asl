"""
Inference forward model for ASL data using neural network
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
        
class AslNNModel(Model):
    """
    ASL resting state model using NN for evaluation
    
    The network is trained using simulated data from the analytic ASLREST
    model
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
        ModelOption("train_ti_max", "Maximum TI to train for", type=float, default=20.0),
        ModelOption("train_delttiss_max", "Maximum value of ATT to train for", type=float, default=3.0),
        ModelOption("train_lr", "Training learning rate", type=float, default=0.001),
        ModelOption("train_steps", "Training steps", type=int, default=30000),
        ModelOption("train_batch_size", "Training batch size", type=int, default=100),
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
            get_parameter("ftiss", dist="LogNormal", 
                          mean=1.5, prior_var=1e6, post_var=1.5, 
                          post_init=self._init_flow,
                          **options),
            get_parameter("delttiss", dist="FoldedNormal", 
                          mean=self.att, var=self.attsd**2,
                          **options)
        ]

        self.variable_weights = []
        self.variable_biases = []
        self.trained_weights = None
        self.trained_biases = None
        if self.train_save:
            self._init_nn()

    def __str__(self):
        return "ASL neural network model: %s" % __version__

    def evaluate(self, params, tpts):
        """
        Basic PASL/pCASL kinetic model

        :param t: Time values of shape 1x1xN or Mx1xN
        :param params Sequence of parameter values arrays, one for each parameter.
                      Each array is MxSx1 tensor where M is the number of voxels and S
                      the number of samples. This may be supplied as a PxMxSx1 tensor 
                      where P is the number of parameters.

        :return: MxSxN tensor containing model output at the specified time values
                 and for each sample using the specified parameter values
        """
        if self.trained_weights is None:
            self._init_nn()

        # Extract parameter tensors
        t = self.log_tf(tpts, name="tpts", shape=True, force=False)
        ftiss = self.log_tf(params[0], name="ftiss", shape=True, force=False) # [M, S]
        delt = self.log_tf(params[1], name="delt", shape=True, force=False) # [M, S]

        # Tile time points and delt to MxSxN    
        tshape = tf.shape(t)
        dshape = tf.shape(delt)
        t = self.log_tf(tf.reshape(t, (tshape[0], 1, tshape[-1])), shape=True, force=False, name="t1")
        t = self.log_tf(tf.tile(t, (dshape[0]/tshape[0], dshape[1], 1)), shape=True, force=False, name="ttield")
        delt = tf.reshape(delt, (dshape[0], dshape[1], 1))
        delt = self.log_tf(tf.tile(delt, (1, 1, tshape[-1])), shape=True, force=False, name="delttiled")

        # Evaluate the NN using the TI and delttiss parameters then
        # multiply by f to get the scaled output
        signal = self._evaluate_nn(tf.stack([t, delt], axis=-1))
        signal = self.log_tf(tf.squeeze(signal), name="sig", shape=True, force=False)
        return self.log_tf(tf.multiply(ftiss, signal), name="scaled", shape=True, force=False)
        
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

    def _init_nn(self):
        self.log.info("Initializing neural-network based ASL model")
        if self.train_load:
            # Load previously trained model
            self._load_nn(self.train_load)
        else:
            # Train model using simulated data and report performance
            # on simulated test data set
            x_train, x_test, y_train, y_test = self._get_training_data_svb(n=self.train_examples)
            self._train_nn(x_train, y_train, self.train_steps, self.train_lr, batch_size=self.train_batch_size)
            
            y_pred = self._ievaluate_nn(x_test)
            accuracy = r2_score(y_pred, y_test)
            self.log.info(' - Trained model using %i steps and %.5f learning rate - accuracy %.3f' % (self.train_steps, self.train_lr, accuracy))
            if self.train_save:
                self._save_nn(self.train_save)

    def _get_training_data_svb(self, n, **kwargs):
        """
        Generate training data by evaluating SVB model
        """
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
        options.update(**kwargs)

        # Generate training data using ftiss=1 (since this is just
        # a scaling factor and is imposed independently of the NN)
        sig = np.zeros((1, len(self.tis)), dtype=np.float32)
        data_model = DataModel(sig)
        model = AslRestModel(data_model, tis=self.tis, **options)
        tpts = np.random.uniform(1.0, 5.0, size=(n,))
        delttiss = np.random.uniform(0.1, self.train_delttiss_max, size=(n,))
        params = np.zeros((2, n, 1), dtype=np.float32)
        params[0, :, 0] = 1.0
        params[1, :, 0] = delttiss
        modelsig = model.ievaluate(params, tpts[..., np.newaxis])
        self.log.info(" - Generated %i instances of training data" % n)

        # Split training data into training and test data sets
        x = np.zeros((n, 2), dtype=np.float32)
        x[:, 0] = tpts
        x[:, 1] = delttiss
        y = np.squeeze(modelsig, axis=-1)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
        self.log.info(" - Separated %i instances of training data and %i instances of test data" % (x_train.shape[0], x_test.shape[0]))
        return x_train, x_test, y_train, y_test

    def _load_nn(self, load_dir):
        self.log.info(" - Loading previously trained model from %s" % load_dir)
        self.trained_weights = []
        self.trained_biases = []
        idx = 0
        while 1:
            weights_file = os.path.join(load_dir, "weights%i.npy" % idx)
            biases_file = os.path.join(load_dir, "biases%i.npy" % idx)
            if not os.path.exists(weights_file) and not os.path.exists(biases_file):
                break
            elif not os.path.exists(weights_file) or not os.path.exists(biases_file):
                raise RuntimeError("For time point %i, could not find both weights and biases")
            else:
                self.trained_weights.append(np.load(weights_file))
                self.trained_biases.append(np.load(biases_file))
            idx += 1
        self.log.info(" - Loaded %i layers" % len(self.trained_weights))

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

    def _train_nn(self, x_train, y_train, steps, learning_rate, batch_size=100):
        """
        Train the model with ASL data

        :param x_train: Training X values (pld, delttiss) [2, N]
        :param y_train: Training Y values (ASL signal / delta-M) [N]
        """
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
            import math
            n_batches = math.ceil(x_train.shape[0] / batch_size)
            self.log.info(" - Batch size %i, number of batches %i" % (batch_size, n_batches))
            for step in range(steps):
                mean_loss = 0
                for batch in range(n_batches):
                    x = x_train[batch:-1:n_batches, :]
                    y = y_train[batch:-1:n_batches, np.newaxis]

                    batch_loss, optimizer_ = sess.run([loss, optimizer], feed_dict={x_input: x, y_input: y})
                    mean_loss += batch_loss
                mean_loss = mean_loss / n_batches
                if step % 100 == 0:
                    self.log.info(" - Step %i, cost %f" % (step, mean_loss))

            self.trained_weights = []
            self.trained_biases = []
            for weights, biases in zip(self.variable_weights, self.variable_biases):
                self.trained_weights.append(sess.run(weights))
                self.trained_biases.append(sess.run(biases))

    def _evaluate_nn(self, x):
        """
        Evaluate the trained model

        :param x: X values (pld, delttiss)
        :return: tensor operation containing Y values (ASL signal / delta-M)
        """
        layers = self._create_nn(x, trainable=False)
        return layers[-1]

    def _ievaluate_nn(self, x):
        """
        Evaluate the trained model interactively 
        (i.e. return the actual answer as a Numpy array not
        as a TensorFlow operation)

        :param x: X values (pld, delttiss)
        :return: Numpy array containing Y values (ASL signal / delta-M)
        """
        with tf.Graph().as_default():
            prediction = self._evaluate_nn(x)
            sess = tf.Session()
            sess.run(tf.global_variables_initializer())
            return sess.run(prediction)

    def _save_nn(self, save_dir):
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
