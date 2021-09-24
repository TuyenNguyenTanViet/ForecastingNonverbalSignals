'''
Authors:      Nguyen Tan Viet Tuyen, Oya Celiktutan
Email:       tan_viet_tuyen.nguyen@kcl.ac.uk
Affiliation: SAIR LAB, King's College London
Project:     LISI -- Learning to Imitate Nonverbal Communication Dynamics for Human-Robot Social Interaction

Python version: 3.6
'''


import os
import numpy as np
import tensorflow as tf
import pickle5 as pickle
from sklearn.utils import shuffle
from lib.config import *
from model import GAN_models

# Learning
tf.app.flags.DEFINE_float("learning_rate", .0005, "Learning rate.")
tf.app.flags.DEFINE_integer("batch_size", 1024, "Batch size to use during training.")
tf.app.flags.DEFINE_integer("epoch", int(1e3), "epoch to train for.")
tf.app.flags.DEFINE_integer("warm_up", 50, "Initial training without adversarial loss") 
tf.app.flags.DEFINE_integer("save_every", 50, "How often to save the model.")
tf.app.flags.DEFINE_integer("start_saving", 500, "Start saving the model") 
tf.app.flags.DEFINE_integer("gF_loss_weight", 10, "Reconstruction weights of Gface")
tf.app.flags.DEFINE_integer("gB_loss_weight", 10, "Reconstruction weights of Gbody")
tf.app.flags.DEFINE_integer("gH_loss_weight", 10, "Reconstruction weights of Ghand")
tf.app.flags.DEFINE_integer("size", 1024, "Size of each model layer.")
tf.app.flags.DEFINE_integer("num_layers", 1, "Number of layers in the model.")
tf.app.flags.DEFINE_string("restore_dir", "", "Restore directory.") 
tf.app.flags.DEFINE_integer("restore_indx", 0, "Restore directory.") 
tf.app.flags.DEFINE_string("dataset", "./dataset/UDIVA_2d.pickle", "Training set.") 
tf.app.flags.DEFINE_string("model_dir", "./model", "Training directory.") 

FLAGS = tf.app.flags.FLAGS
summaries_dir = os.path.normpath(os.path.join(FLAGS.model_dir, "log")) 

class GANtrainer():
    def __init__(self, 
                batch_size,
                learning_rate,
                summaries_dir,
                dtype=tf.float32):

        self.batch_size = batch_size
        self.learning_rate = tf.Variable(float(learning_rate), trainable=False, dtype=dtype)
        self.summaries_dir = summaries_dir

        self.context_inputs = tf.placeholder(dtype, shape=[None, source_len, n_features_h1], name="context_inputs")
        self.real_p = tf.placeholder(dtype, shape=[None, source_len+target_len, n_features_h1], name="context_inputs")
        self.enc_in_B =  tf.placeholder(dtype, shape=[None, source_len-1, n_features_h2_B], name="enc_in_B")
        self.dec_in_B =  tf.placeholder(dtype, shape=[None, target_len, n_features_h2_B], name="dec_in_B")
        self.dec_out_B = tf.placeholder(dtype, shape=[None, target_len, n_features_h2_B], name="dec_out_B")
        self.enc_in_F =  tf.placeholder(dtype, shape=[None, source_len-1, n_features_h2_F], name="enc_in_F")
        self.dec_in_F =  tf.placeholder(dtype, shape=[None, target_len, n_features_h2_F], name="dec_in_F")
        self.dec_out_F = tf.placeholder(dtype, shape=[None, target_len, n_features_h2_F], name="dec_out_F")
        self.enc_in_H =  tf.placeholder(dtype, shape=[None, source_len-1, n_features_h2_H], name="enc_in_F")
        self.dec_in_H =  tf.placeholder(dtype, shape=[None, target_len, n_features_h2_H], name="dec_in_F")
        self.dec_out_H = tf.placeholder(dtype, shape=[None, target_len, n_features_h2_H], name="dec_out_F")

    def train(self):
        model = GAN_models(
            (source_len, n_features_h1), 
            context_len, 
            (target_len, n_features_h2_B, n_features_h2_F, n_features_h2_H), 
            FLAGS.size, 
            FLAGS.num_layers)

        context_inputs = self.context_inputs
        real_p_future   = self.real_p[:,-target_len:,:]

        outputs_human = model.EncoderP(context_inputs)

        context_tiled = tf.tile(tf.expand_dims(tf.cast(outputs_human, dtype=tf.float32), axis=1),
                                    multiples=[1, target_len, 1])
        dec_in_B = tf.concat([self.dec_in_B, context_tiled], 2)
        dec_in_F = tf.concat([self.dec_in_F, context_tiled], 2)
        dec_in_H = tf.concat([self.dec_in_H, context_tiled], 2)

        enc_in_B = tf.transpose(self.enc_in_B, [1, 0, 2])
        dec_in_B = tf.transpose(dec_in_B, [1, 0, 2])
        dec_out_B = tf.transpose(self.dec_out_B, [1, 0, 2])

        enc_in_B = tf.reshape(enc_in_B, [-1, n_features_h2_B])
        dec_in_B = tf.reshape(dec_in_B, [-1, n_features_h2_B + context_len])
        dec_out_B = tf.reshape(dec_out_B, [-1, n_features_h2_B])

        enc_in_B = tf.split(enc_in_B, source_len-1, axis=0)
        dec_in_B = tf.split(dec_in_B, target_len, axis=0)
        dec_out_B = tf.split(dec_out_B, target_len, axis=0)

        G_B = model.GeneratorB(enc_in_B, dec_in_B)

        enc_in_F = tf.transpose(self.enc_in_F, [1, 0, 2])
        dec_in_F = tf.transpose(dec_in_F, [1, 0, 2])
        dec_out_F = tf.transpose(self.dec_out_F, [1, 0, 2])

        enc_in_F = tf.reshape(enc_in_F, [-1, n_features_h2_F])
        dec_in_F = tf.reshape(dec_in_F, [-1, n_features_h2_F + context_len])
        dec_out_F = tf.reshape(dec_out_F, [-1, n_features_h2_F])

        enc_in_F = tf.split(enc_in_F, source_len-1, axis=0)
        dec_in_F = tf.split(dec_in_F, target_len, axis=0)
        dec_out_F = tf.split(dec_out_F, target_len, axis=0)

        G_F = model.GeneratorF(enc_in_F, dec_in_F)

        enc_in_H = tf.transpose(self.enc_in_H, [1, 0, 2])
        dec_in_H = tf.transpose(dec_in_H, [1, 0, 2])
        dec_out_H = tf.transpose(self.dec_out_H, [1, 0, 2])

        enc_in_H = tf.reshape(enc_in_H, [-1, n_features_h2_H])
        dec_in_H = tf.reshape(dec_in_H, [-1, n_features_h2_H + context_len])
        dec_out_H = tf.reshape(dec_out_H, [-1, n_features_h2_H])

        enc_in_H = tf.split(enc_in_H, source_len-1, axis=0)
        dec_in_H = tf.split(dec_in_H, target_len, axis=0)
        dec_out_H = tf.split(dec_out_H, target_len, axis=0)

        G_H = model.GeneratorH(enc_in_H, dec_in_H)
        
        generated_p_fututure = tf.transpose(tf.concat([G_B, G_F, G_H], axis=2), [1, 0, 2])

        d_real = model.Discriminator(real_p_future)
        d_fake = model.Discriminator(generated_p_fututure, reuse_flag=True)

        gen_loss_B = model.G_rec_loss(real_action=dec_out_B,
                                 fake_action=G_B, g_reconstruction_loss_weight= FLAGS.gB_loss_weight)

        gen_loss_F = model.G_rec_loss(real_action=dec_out_F,
                                 fake_action=G_F, g_reconstruction_loss_weight= FLAGS.gF_loss_weight)

        gen_loss_H = model.G_rec_loss(real_action=dec_out_H,
                                 fake_action=G_H, g_reconstruction_loss_weight= FLAGS.gH_loss_weight)

        dis_loss = model.D_loss(d_real, d_fake)
        gen_rec_loss = gen_loss_B+gen_loss_F+gen_loss_H
        gen_adv_loss = model.G_adv_loss(d_fake)
        gen_total_loss = gen_rec_loss + gen_adv_loss

        T_vars = tf.trainable_variables()
        G_vars = [var for var in T_vars if (var.name.startswith('GeneratorB') or var.name.startswith('GeneratorF') or var.name.startswith('GeneratorH') or var.name.startswith('EncoderP'))]
        D_vars = [var for var in T_vars if (var.name.startswith('Discriminator'))]

        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            g_trainer_wo_adv = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.999).minimize(gen_rec_loss, var_list=G_vars)
            g_trainer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.999).minimize(gen_total_loss, var_list=G_vars)
            d_trainer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.999).minimize(dis_loss, var_list=D_vars)

        tf_loss_G = tf.placeholder(tf.float32, shape=None, name='loss_G_sum')
        tf_loss_GB = tf.placeholder(tf.float32, shape=None, name='loss_GB_sum')
        tf_loss_GF = tf.placeholder(tf.float32, shape=None, name='loss_GF_sum')
        tf_loss_GH = tf.placeholder(tf.float32, shape=None, name='loss_GH_sum')
        tf_loss_D = tf.placeholder(tf.float32, shape=None, name='loss_D_sum')
        tf_loss_G_sum = tf.summary.scalar('loss_G_adv', tf_loss_G)
        loss_GB_sum = tf.summary.scalar('loss_GB', tf_loss_GB)
        loss_GF_sum = tf.summary.scalar('loss_GF', tf_loss_GF)
        loss_GH_sum = tf.summary.scalar('loss_GH', tf_loss_GH)
        tf_loss_D_sum = tf.summary.scalar('loss_D', tf_loss_D)
        learning_rate_summary = tf.summary.scalar('learning_rate/learning_rate', self.learning_rate)
        performance_sum = tf.summary.merge([tf_loss_G_sum, loss_GB_sum, loss_GF_sum, loss_GH_sum, tf_loss_D_sum, learning_rate_summary])
        writer = tf.summary.FileWriter(os.path.normpath(os.path.join(self.summaries_dir, 'train')), graph=tf.get_default_graph())
        gan_saver = tf.train.Saver(var_list=tf.trainable_variables(), max_to_keep=None)

        with create_session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            if (FLAGS.restore_indx):
                checkpoint_EG = FLAGS.restore_dir + "/checkpoint-" + str(FLAGS.restore_indx) + "/checkpoint-" + str(FLAGS.restore_indx)
                gan_saver.restore(sess, checkpoint_EG)
                print("Restored model at: ", checkpoint_EG)

            t_context_inp, t_encoder_inp_B, t_decoder_inp_B, t_decoder_out_B, t_encoder_inp_F, t_decoder_inp_F, t_decoder_out_F, t_encoder_inp_H, t_decoder_inp_H, t_decoder_out_H , t_real_p= data_generator(data_type='train_FC_all')

            print("train_context_inp", np.shape(t_context_inp))
            for epoch in range(FLAGS.restore_indx, FLAGS.epoch+1):
                G_losses = []
                GB_losses = []
                GF_losses = []
                GH_losses = []
                D_losses = []
                if epoch < FLAGS.warm_up:
                    for iter in range(np.shape(t_context_inp)[0]//FLAGS.batch_size):
                        context_inp = t_context_inp[iter*FLAGS.batch_size:(iter+1)*FLAGS.batch_size]
                        real_p_inp  = t_real_p[iter*FLAGS.batch_size:(iter+1)*FLAGS.batch_size]
                        encoder_inp_B = t_encoder_inp_B[iter*FLAGS.batch_size:(iter+1)*FLAGS.batch_size]
                        decoder_inp_B = t_decoder_inp_B[iter*FLAGS.batch_size:(iter+1)*FLAGS.batch_size]
                        decoder_out_B = t_decoder_out_B[iter*FLAGS.batch_size:(iter+1)*FLAGS.batch_size]
                        encoder_inp_F = t_encoder_inp_F[iter*FLAGS.batch_size:(iter+1)*FLAGS.batch_size]
                        decoder_inp_F = t_decoder_inp_F[iter*FLAGS.batch_size:(iter+1)*FLAGS.batch_size]
                        decoder_out_F = t_decoder_out_F[iter*FLAGS.batch_size:(iter+1)*FLAGS.batch_size]
                        encoder_inp_H = t_encoder_inp_H[iter*FLAGS.batch_size:(iter+1)*FLAGS.batch_size]
                        decoder_inp_H = t_decoder_inp_H[iter*FLAGS.batch_size:(iter+1)*FLAGS.batch_size]
                        decoder_out_H = t_decoder_out_H[iter*FLAGS.batch_size:(iter+1)*FLAGS.batch_size]
                        loss_D, _ = sess.run([dis_loss, d_trainer], feed_dict={self.context_inputs: context_inp,
                                                                                self.real_p: real_p_inp,
                                                                                self.enc_in_B: encoder_inp_B,
                                                                                self.dec_in_B: decoder_inp_B,
                                                                                self.dec_out_B: decoder_out_B,
                                                                                self.enc_in_F: encoder_inp_F,
                                                                                self.dec_in_F: decoder_inp_F,
                                                                                self.dec_out_F: decoder_out_F,
                                                                                self.enc_in_H: encoder_inp_H,
                                                                                self.dec_in_H: decoder_inp_H,
                                                                                self.dec_out_H: decoder_out_H})
                        D_losses.append(loss_D)
                        loss_B, loss_F, loss_H, _ = sess.run([gen_loss_B, gen_loss_F, gen_loss_H, g_trainer_wo_adv], feed_dict={self.context_inputs: context_inp,
                                                                                                                self.real_p: real_p_inp,
                                                                                                                self.enc_in_B: encoder_inp_B,
                                                                                                                self.dec_in_B: decoder_inp_B,
                                                                                                                self.dec_out_B: decoder_out_B,
                                                                                                                self.enc_in_F: encoder_inp_F,
                                                                                                                self.dec_in_F: decoder_inp_F,
                                                                                                                self.dec_out_F: decoder_out_F,
                                                                                                                self.enc_in_H: encoder_inp_H,
                                                                                                                self.dec_in_H: decoder_inp_H,
                                                                                                                self.dec_out_H: decoder_out_H})
                        
                        G_losses.append(0)
                        GB_losses.append(loss_B)
                        GF_losses.append(loss_F)
                        GH_losses.append(loss_H)

                    print("epoch", epoch, "G_losses", np.mean(G_losses), "GB_losses", np.mean(GB_losses), "GF_losses", np.mean(GF_losses), "GH_losses", np.mean(GH_losses), "D_losses", np.mean(D_losses))
                else:
                    for iter in range(np.shape(t_context_inp)[0]//FLAGS.batch_size):
                        context_inp = t_context_inp[iter*FLAGS.batch_size:(iter+1)*FLAGS.batch_size]
                        real_p_inp  = t_real_p[iter*FLAGS.batch_size:(iter+1)*FLAGS.batch_size]
                        encoder_inp_B = t_encoder_inp_B[iter*FLAGS.batch_size:(iter+1)*FLAGS.batch_size]
                        decoder_inp_B = t_decoder_inp_B[iter*FLAGS.batch_size:(iter+1)*FLAGS.batch_size]
                        decoder_out_B = t_decoder_out_B[iter*FLAGS.batch_size:(iter+1)*FLAGS.batch_size]
                        encoder_inp_F = t_encoder_inp_F[iter*FLAGS.batch_size:(iter+1)*FLAGS.batch_size]
                        decoder_inp_F = t_decoder_inp_F[iter*FLAGS.batch_size:(iter+1)*FLAGS.batch_size]
                        decoder_out_F = t_decoder_out_F[iter*FLAGS.batch_size:(iter+1)*FLAGS.batch_size]
                        encoder_inp_H = t_encoder_inp_H[iter*FLAGS.batch_size:(iter+1)*FLAGS.batch_size]
                        decoder_inp_H = t_decoder_inp_H[iter*FLAGS.batch_size:(iter+1)*FLAGS.batch_size]
                        decoder_out_H = t_decoder_out_H[iter*FLAGS.batch_size:(iter+1)*FLAGS.batch_size]

                        loss_D, _ = sess.run([dis_loss, d_trainer], feed_dict={self.context_inputs: context_inp,
                                                                    self.real_p: real_p_inp,
                                                                    self.enc_in_B: encoder_inp_B,
                                                                    self.dec_in_B: decoder_inp_B,
                                                                    self.dec_out_B: decoder_out_B,
                                                                    self.enc_in_F: encoder_inp_F,
                                                                    self.dec_in_F: decoder_inp_F,
                                                                    self.dec_out_F: decoder_out_F,
                                                                    self.enc_in_H: encoder_inp_H,
                                                                    self.dec_in_H: decoder_inp_H,
                                                                    self.dec_out_H: decoder_out_H})
                        D_losses.append(loss_D)
                        loss_G, loss_B, loss_F, loss_H, _ = sess.run([gen_adv_loss, gen_loss_B, gen_loss_F, gen_loss_H, g_trainer], feed_dict={self.context_inputs: context_inp,
                                                                                                                        self.real_p: real_p_inp,
                                                                                                                        self.enc_in_B: encoder_inp_B,
                                                                                                                        self.dec_in_B: decoder_inp_B,
                                                                                                                        self.dec_out_B: decoder_out_B,
                                                                                                                        self.enc_in_F: encoder_inp_F,
                                                                                                                        self.dec_in_F: decoder_inp_F,
                                                                                                                        self.dec_out_F: decoder_out_F,
                                                                                                                        self.enc_in_H: encoder_inp_H,
                                                                                                                        self.dec_in_H: decoder_inp_H,
                                                                                                                        self.dec_out_H: decoder_out_H})
                        G_losses.append(loss_G)
                        GB_losses.append(loss_B)
                        GF_losses.append(loss_F)
                        GH_losses.append(loss_H)

                    print("epoch", epoch, "G_losses", np.mean(G_losses), "GB_losses", np.mean(GB_losses), "GF_losses", np.mean(GF_losses), "GH_losses", np.mean(GH_losses), "D_losses", np.mean(D_losses))

                summaries = sess.run(performance_sum, feed_dict={tf_loss_G: np.mean(G_losses),
                                                                tf_loss_GB: np.mean(GB_losses),
                                                                tf_loss_GF: np.mean(GF_losses),
                                                                tf_loss_GH: np.mean(GH_losses),
                                                                tf_loss_D: np.mean(D_losses)})
                writer.add_summary(summaries, epoch)

                if epoch>= FLAGS.start_saving and epoch % FLAGS.save_every == 0:
                    print("Saving the model...")
                    gan_saver.save(sess, os.path.normpath(
                        os.path.join(FLAGS.model_dir, "checkpoint-"+str(epoch), 'checkpoint')),
                        global_step=epoch)


def create_session():
    return tf.Session(config=tf.ConfigProto(device_count={"CPU":1, "GPU":0}, 
                                            inter_op_parallelism_threads=0,
                                            allow_soft_placement=True,
                                            gpu_options={'allow_growth': True, 'visible_device_list': "0"},
                                            intra_op_parallelism_threads=0))
def data_generator(data_type):
    if data_type not in ['train_FC_all']:
        raise (ValueError, "'{0}' is not an appropriate data type.".format(data_type))

    if data_type == 'train_FC_all':
        with open(FLAGS.dataset, 'rb') as f:
            data = pickle.load(f)

        # all_name_motions = data['train_sess_name']
        p_observed_1 = np.array(data['train_motion_FC1'], dtype="float32")  
        p_forecast_1 = np.array(data['train_motion_FC2'], dtype="float32") 
        p_observed_2 = np.array(data['train_motion_FC2'], dtype="float32") # reverse
        p_forecast_2 = np.array(data['train_motion_FC1'], dtype="float32") # reverse
        
        p_observed = np.vstack((p_observed_1, p_observed_2))
        p_forecast = np.vstack((p_forecast_1, p_forecast_2))
        p_observed, p_forecast = shuffle(p_observed, p_forecast, random_state=0)
        num_train = np.shape(p_observed)[0]

        context_inp = p_observed[0:num_train,0:source_len,0:n_features_h1]
        real_p      = p_forecast[0:num_train,0:source_len+target_len,0:n_features_h1]

        encoder_inp_B = p_forecast[0:num_train, 0:(source_len-1), 0:n_features_h2_B]
        decoder_inp_B = p_forecast[0:num_train, (source_len-1):-1,0:n_features_h2_B]
        decoder_out_B = p_forecast[0:num_train, -target_len:,     0:n_features_h2_B]

        encoder_inp_F = p_forecast[0:num_train, 0:(source_len-1), n_features_h2_B:n_features_h2_B+n_features_h2_F]
        decoder_inp_F = p_forecast[0:num_train, (source_len-1):-1,n_features_h2_B:n_features_h2_B+n_features_h2_F]
        decoder_out_F = p_forecast[0:num_train, -target_len:,     n_features_h2_B:n_features_h2_B+n_features_h2_F]

        encoder_inp_H = p_forecast[0:num_train, 0:(source_len-1), n_features_h2_B+n_features_h2_F:]
        decoder_inp_H = p_forecast[0:num_train, (source_len-1):-1,n_features_h2_B+n_features_h2_F:]
        decoder_out_H = p_forecast[0:num_train, -target_len:,     n_features_h2_B+n_features_h2_F:]


        return context_inp, encoder_inp_B, decoder_inp_B, decoder_out_B, encoder_inp_F, decoder_inp_F, decoder_out_F, encoder_inp_H, decoder_inp_H, decoder_out_H, real_p
   
if __name__ == "__main__":
    trainer = GANtrainer( FLAGS.batch_size,
                          FLAGS.learning_rate,
                          summaries_dir,
                          dtype=tf.float32)
    trainer.train()