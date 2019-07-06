from layers import *
from metrics import *

flags = tf.app.flags
FLAGS = flags.FLAGS


class Model(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging

        self.vars = {}
        self.placeholders = {}

        self.layers = []
        self.GCNlayers = []
        self.activations = []

        self.inputs = None
        self.outputs = None

        self.loss = 0
        self.accuracy = 0
        self.optimizer = None
        self.opt_op = None
        self.embed = None

    def _build(self):
        raise NotImplementedError

    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            self._build()

        # Build sequential layer model
        self.activations.append(self.inputs)

        for i in range(len(self.layers)):
            layer = self.layers[i]
            hidden, pre_GCN = layer(self.activations[-1])
            self.GCNlayers.append(pre_GCN)
            if i >= FLAGS.coarsen_level and i < FLAGS.coarsen_level * 2:
                hidden = hidden + self.GCNlayers[FLAGS.coarsen_level * 2-i - 1] 
            self.activations.append(hidden)
        self.outputs = self.activations[-1]
        self.embed = self.activations[-2]

        # Store model variables for easy access
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

        # Build metrics
        self._loss()
        self._accuracy()

        self.opt_op = self.optimizer.minimize(self.loss)

    def predict(self):
        pass

    def _loss(self):
        raise NotImplementedError

    def _accuracy(self):
        raise NotImplementedError

    def save(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = saver.save(sess, "tmp/%s.ckpt" % self.name)
        print("Model saved in file: %s" % save_path)

    def load(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = "tmp/%s.ckpt" % self.name
        saver.restore(sess, save_path)
        print("Model restored from file: %s" % save_path)


class MLP(Model):
    def __init__(self, placeholders, input_dim, **kwargs):
        super(MLP, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = input_dim
        # self.input_dim = self.inputs.get_shape().as_list()[1]  # To be supported in future Tensorflow versions
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.placeholders = placeholders

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

        self.build()

    def _loss(self):
        # Weight decay loss
        for var in self.layers[0].vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        # Cross entropy error
        self.loss += masked_softmax_cross_entropy(self.outputs, self.placeholders['labels'],
                                                  self.placeholders['labels_mask'])

    def _accuracy(self):
        self.accuracy = masked_accuracy(self.outputs, self.placeholders['labels'],
                                        self.placeholders['labels_mask'])

    def _build(self):
        self.layers.append(Dense(input_dim=self.input_dim,
                                 output_dim=FLAGS.hidden1,
                                 placeholders=self.placeholders,
                                 act=tf.nn.relu,
                                 dropout=True,
                                 sparse_inputs=True,
                                 logging=self.logging))

        self.layers.append(Dense(input_dim=FLAGS.hidden1,
                                 output_dim=self.output_dim,
                                 placeholders=self.placeholders,
                                 act=lambda x: x,
                                 dropout=True,
                                 logging=self.logging))

    def predict(self):
        return tf.nn.softmax(self.outputs)


class GCN(Model):
    def __init__(self, placeholders, input_dim, **kwargs):
        super(GCN, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = input_dim
        # self.input_dim = self.inputs.get_shape().as_list()[1]  # To be supported in future Tensorflow versions
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.placeholders = placeholders

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

        self.build()

    def _loss(self):
        # Weight decay loss
        # for var in self.layers[0].vars.values():
        #     self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)
        for i in range(len(self.layers)):
            for var in self.layers[i].vars.values():
                self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)        

        # Cross entropy error
        self.loss += masked_softmax_cross_entropy(self.outputs, self.placeholders['labels'],
                                                  self.placeholders['labels_mask'])

    def _accuracy(self):
        self.accuracy = masked_accuracy(self.outputs, self.placeholders['labels'],
                                        self.placeholders['labels_mask'])

    def _build(self):

        self.layers.append(GraphConvolution(input_dim=self.input_dim,
                                            output_dim=FLAGS.hidden1,
                                            placeholders=self.placeholders,
                                            act=tf.nn.relu,
                                            dropout=True,
                                            sparse_inputs=True,
                                            logging=self.logging))

        self.layers.append(GraphConvolution(input_dim=FLAGS.hidden1,
                                            output_dim=self.output_dim,
                                            placeholders=self.placeholders,
                                            act=lambda x: x,
                                            dropout=True,
                                            logging=self.logging))

    def predict(self):
        return tf.nn.softmax(self.outputs)


class HGCN(Model):
    def __init__(self, placeholders, input_dim, transfer_list, adj_list, node_wgt_list, **kwargs):
        super(HGCN, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = input_dim
        # self.input_dim = self.inputs.get_shape().as_list()[1]  # To be supported in future Tensorflow versions
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.placeholders = placeholders

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
        self.transfer_list = transfer_list
        self.adj_list = adj_list
        self.node_wgt_list = node_wgt_list

        self.W_node_wgt = tf.Variable(tf.random_uniform([FLAGS.max_node_wgt,FLAGS.node_wgt_embed_dim],
                     minval=-math.sqrt(6/(3*FLAGS.node_wgt_embed_dim+3*self.input_dim)),
                     maxval=math.sqrt(6/(3*FLAGS.node_wgt_embed_dim+3*self.input_dim))),
                     name="W_node_wgt")
        self.build()

    def _loss(self):
        # Weight decay loss
        #print(len(self.layers))
        for i in range(len(self.layers)):
            for var in self.layers[i].vars.values():
                self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var) 

        # Cross entropy error
        self.loss += masked_softmax_cross_entropy(self.outputs, self.placeholders['labels'],
                                                  self.placeholders['labels_mask'])

    def _accuracy(self):
        self.accuracy = masked_accuracy(self.outputs, self.placeholders['labels'],
                                        self.placeholders['labels_mask'])

    def _build(self):

        FCN_hidden_list = [FLAGS.hidden] * 100
        node_emb = tf.nn.embedding_lookup(self.W_node_wgt, self.node_wgt_list[0])
        self.layers.append(GraphConvolution(input_dim=self.input_dim,
                                            output_dim=FCN_hidden_list[0],
                                            placeholders=self.placeholders,
                                            support = self.adj_list[0]* FLAGS.channel_num ,
                                            transfer = self.transfer_list[0],
                                            node_emb = node_emb,
                                            mod = 'input',
                                            layer_index = 0,
                                            act=tf.nn.relu,
                                            dropout=True,
                                            sparse_inputs=True,                  
                                            logging=self.logging))  #G0


        for i in range(FLAGS.coarsen_level - 1):
            node_emb = tf.nn.embedding_lookup(self.W_node_wgt, self.node_wgt_list[i + 1])
            self.layers.append(GraphConvolution(input_dim=FCN_hidden_list[i],
                                            output_dim=FCN_hidden_list[i + 1],
                                            placeholders=self.placeholders,
                                            support = self.adj_list[i + 1] * FLAGS.channel_num,
                                            transfer = self.transfer_list[i + 1],  
                                            node_emb = node_emb,
                                            mod = 'coarsen',
                                            layer_index = i + 1,                                         
                                            act=tf.nn.relu,
                                            dropout=True,
                                            logging=self.logging) ) #Gi

        for i in range(FLAGS.coarsen_level, FLAGS.coarsen_level * 2):
            node_emb = tf.nn.embedding_lookup(self.W_node_wgt, self.node_wgt_list[2*FLAGS.coarsen_level - i])
            self.layers.append(GraphConvolution(input_dim=FCN_hidden_list[i - 1],
                                            output_dim=FCN_hidden_list[i],
                                            placeholders=self.placeholders, 
                                            support = self.adj_list[2*FLAGS.coarsen_level - i] *FLAGS.channel_num,
                                            transfer = self.transfer_list[2*FLAGS.coarsen_level -1 -i],  
                                            node_emb = node_emb,
                                            mod = 'refine',
                                            layer_index = i,                                         
                                            act=tf.nn.relu,
                                            dropout=True,
                                            logging=self.logging) )#G?-1



        self.layers.append(GraphConvolution(input_dim=FCN_hidden_list[FLAGS.coarsen_level * 2 - 1],
                                            output_dim=self.output_dim,
                                            placeholders=self.placeholders,
                                            support = self.adj_list[0] * FLAGS.channel_num,
                                            transfer = self.transfer_list[0],  
                                            node_emb = 0,
                                            mod = 'output',
                                            layer_index = FLAGS.coarsen_level * 2,                                         
                                            act=lambda x: x,
                                            dropout=True,
                                            logging=self.logging))  



    def predict(self):
        return tf.nn.softmax(self.outputs)
