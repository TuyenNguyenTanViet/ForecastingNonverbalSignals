import tensorflow as tf
from lib import rnn_cell_extensions 

class GAN_models(object):
    def __init__(self,
                source_seq_size, 
                context_len, 
                target_seq_size, 
                rnn_size, 
                num_layers):
        self.source_seq_size = source_seq_size
        self.context_len = context_len
        self.target_seq_size = target_seq_size
        self.rnn_size = rnn_size
        self.num_layers = num_layers


    def EncoderP(self, con_in):
        with tf.variable_scope("EncoderP"): 
            con_in = tf.unstack(con_in, self.source_seq_size[0], 1)
            lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.rnn_size, forget_bias=1.0)
            outputs_p1, states = tf.contrib.rnn.static_rnn(lstm_cell, con_in, dtype=tf.float32)
            outputs = tf.layers.dense(outputs_p1[-1], self.context_len, name='EncoderP1/fc')
            return  outputs

    def GeneratorB(self, enc_in, dec_in):
        with tf.variable_scope("GeneratorB"): 
            cell_robot_enc = tf.contrib.rnn.BasicLSTMCell(self.rnn_size, forget_bias=1.0, name="cell_robot_enc")
            cell_robot_dec = tf.contrib.rnn.BasicLSTMCell(self.rnn_size, forget_bias=1.0, name="cell_robot_dec")
            if self.num_layers > 1:
                cell_robot_enc = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.BasicLSTMCell(self.rnn_size, forget_bias=1.0) for _ in range(self.num_layers)])
                cell_robot_dec = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.BasicLSTMCell(self.rnn_size, forget_bias=1.0) for _ in range(self.num_layers)])

            cell_robot_dec = rnn_cell_extensions.LinearSpaceDecoderWrapper(cell_robot_dec, self.target_seq_size[1],
                                                                    w_name="w_out2", b_name="b_out2", scope="GeneratorB/loop/weights")         
            cell_robot_dec = rnn_cell_extensions.ResidualWrapper(cell_robot_dec)   

            outputs_robot = []
            def lf(prev, i): 
                return prev

            _, enc_state = tf.contrib.rnn.static_rnn(cell_robot_enc, enc_in, dtype=tf.float32) 
            outputs_robot, self.states = tf.contrib.legacy_seq2seq.rnn_decoder(dec_in, enc_state, cell_robot_dec, loop_function=lf)  
            outputs_robot = [tf.slice(outputs_robot[i], [0, 0], [-1, self.target_seq_size[1]]) for i in range(len(outputs_robot))]
            
            return outputs_robot

    def GeneratorF(self, enc_in, dec_in):
        with tf.variable_scope("GeneratorF"): 
            cell_robot_enc = tf.contrib.rnn.BasicLSTMCell(self.rnn_size, forget_bias=1.0, name="cell_robot_enc")
            cell_robot_dec = tf.contrib.rnn.BasicLSTMCell(self.rnn_size, forget_bias=1.0, name="cell_robot_dec")
            if self.num_layers > 1:
                cell_robot_enc = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.BasicLSTMCell(self.rnn_size, forget_bias=1.0) for _ in range(self.num_layers)])
                cell_robot_dec = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.BasicLSTMCell(self.rnn_size, forget_bias=1.0) for _ in range(self.num_layers)])

            cell_robot_dec = rnn_cell_extensions.LinearSpaceDecoderWrapper(cell_robot_dec, self.target_seq_size[2],
                                                                    w_name="w_out2", b_name="b_out2", scope="GeneratorF/loop/weights")         
            cell_robot_dec = rnn_cell_extensions.ResidualWrapper(cell_robot_dec)   

            outputs_robot = []
            def lf(prev, i): 
                return prev

            _, enc_state = tf.contrib.rnn.static_rnn(cell_robot_enc, enc_in, dtype=tf.float32)  # Encoder 
            outputs_robot, self.states = tf.contrib.legacy_seq2seq.rnn_decoder(dec_in, enc_state, cell_robot_dec, loop_function=lf)  # Decoder
            outputs_robot = [tf.slice(outputs_robot[i], [0, 0], [-1, self.target_seq_size[2]]) for i in range(len(outputs_robot))]
            
            return outputs_robot

    def GeneratorH(self, enc_in, dec_in):
        with tf.variable_scope("GeneratorH"):  # GeneratorB: body features
            # tf.contrib.rnn.GRUCell or bi-directional
            cell_robot_enc = tf.contrib.rnn.BasicLSTMCell(self.rnn_size, forget_bias=1.0, name="cell_robot_enc")
            cell_robot_dec = tf.contrib.rnn.BasicLSTMCell(self.rnn_size, forget_bias=1.0, name="cell_robot_dec")
            if self.num_layers > 1:
                cell_robot_enc = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.BasicLSTMCell(self.rnn_size, forget_bias=1.0) for _ in range(self.num_layers)])
                cell_robot_dec = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.BasicLSTMCell(self.rnn_size, forget_bias=1.0) for _ in range(self.num_layers)])

            # # === Add space decoder ===
            cell_robot_dec = rnn_cell_extensions.LinearSpaceDecoderWrapper(cell_robot_dec, self.target_seq_size[3],
                                                                    w_name="w_out2", b_name="b_out2", scope="GeneratorH/loop/weights")         
            cell_robot_dec = rnn_cell_extensions.ResidualWrapper(cell_robot_dec)   

            outputs_robot = []
            def lf(prev, i): 
                return prev

            _, enc_state = tf.contrib.rnn.static_rnn(cell_robot_enc, enc_in, dtype=tf.float32)  
            outputs_robot, self.states = tf.contrib.legacy_seq2seq.rnn_decoder(dec_in, enc_state, cell_robot_dec, loop_function=lf)  
            outputs_robot = [tf.slice(outputs_robot[i], [0, 0], [-1, self.target_seq_size[3]]) for i in range(len(outputs_robot))]
            
            return outputs_robot

    def Discriminator(self, input_, reuse_flag=False):
        with tf.variable_scope("Discriminator", reuse=reuse_flag):
            input_ = tf.unstack(input_, self.target_seq_size[0], 1)
            lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.rnn_size, forget_bias=1.0)
            _O, _ = tf.contrib.rnn.static_rnn(lstm_cell, input_, dtype=tf.float32)
            _results = tf.layers.dense(_O[-1], 1, name='fc')

        return _results

    def D_loss(self, d_real_action, d_fake_action):
        d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_real_action, labels=tf.ones_like(d_real_action)*0.98))
        d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_fake_action, labels=tf.zeros_like(d_fake_action)))
        d_loss_total = d_loss_real + d_loss_fake

        return d_loss_total


    def G_rec_loss(self, real_action, fake_action, g_reconstruction_loss_weight):
        return tf.reduce_mean(tf.square(tf.subtract(real_action, fake_action)))*g_reconstruction_loss_weight

    def G_adv_loss(self, d_fake):
        adv_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_fake, labels=tf.ones_like(d_fake)))
        return adv_loss
