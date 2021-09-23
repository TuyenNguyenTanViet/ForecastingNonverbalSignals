import os
import tensorflow as tf
import numpy as np
import argparse
import h5py
import json
import pandas as pd
from model import GAN_models
from lib.config import *
from lib.annotations_parser import is_valid, is_to_predict, parse_face_at, parse_lhand_at, parse_rhand_at, parse_body_at

n_features = (10+28+20+20)*2 # body+face+rhand+lhand
n_past = 100
n_features_B = 10*2 
n_features_F =  28*2 
n_features_RH = 20*2
n_features_LH =  20*2 

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=int, required=False, default=1024)
    parser.add_argument('--num_layers', type=int, required=False, default=1)
    parser.add_argument('--model_dir', type=str, required=False, default="./model")
    parser.add_argument('--model_index', type=int, required=False, default=1000)
    parser.add_argument('--annotations_dir', type=str, required=False, default="/home/tuyen/Documents/UDI/v_allGAN/testing/talk_annotations_test_masked/")
    parser.add_argument('--segments_path', type=str, required=False, default="/home/tuyen/Documents/UDI/v_allGAN/testing/test_segments_topredict.csv")
    parser.add_argument('--json_folder', type=str, required=False, default="./json")

    return parser.parse_args()

def create_session():
    return tf.Session(config=tf.ConfigProto(device_count={"CPU":1, "GPU":1}, 
                                            inter_op_parallelism_threads=0,
                                            allow_soft_placement=True,
                                            gpu_options={'allow_growth': True, 'visible_device_list': "0"},
                                            intra_op_parallelism_threads=0))

def normalize_motion_std(Y_train):
    std_val = np.std(Y_train)
    mean_pose = np.mean(Y_train)
    eps = 1e-8
    Y_train_normalized = np.divide(Y_train - mean_pose, std_val + eps)

    return Y_train_normalized, std_val, mean_pose

def denormalize_motion_std(sequence, std_val, mean_pose):
    eps = 1e-8
    reconstructed = np.multiply(sequence, std_val + eps) + mean_pose

    return reconstructed

def generate_seq(annotations, session, p_start):
    annotations_path_FC1 = os.path.join(annotations, session, "FC1_T", "annotations_cleaned_masked.hdf5")
    annotations_path_FC2 = os.path.join(annotations, session, "FC2_T", "annotations_cleaned_masked.hdf5")
    f_FC1 = h5py.File(annotations_path_FC1, "r") 
    f_FC2 = h5py.File(annotations_path_FC2, "r")
    assert os.path.exists(annotations_path_FC1) or annotations_path_FC1.split(".")[-1].lower() != "hdf5", "HDF5 FC1_T file could not be opened."
    assert os.path.exists(annotations_path_FC2) or annotations_path_FC2.split(".")[-1].lower() != "hdf5", "HDF5 FC2_T file could not be opened."
    assert len(f_FC1) == len(f_FC2), "len(f_FC1) != len(f_FC2)" 
    human_features_FC1 = get_data(annotations_path_FC1, p_start - n_past, p_start-1)
    human_features_FC2 = get_data(annotations_path_FC2, p_start - n_past, p_start-1)
    sess_motion_FC1, sess_max_FC1, sess_mean_FC1 = normalize_motion_std(human_features_FC1) # normalize_motion
    sess_motion_FC2, sess_max_FC2, sess_mean_FC2 = normalize_motion_std(human_features_FC2) # normalize_motion

    return sess_motion_FC1, sess_motion_FC2, sess_max_FC1, sess_max_FC2, sess_mean_FC1, sess_mean_FC2

def get_data(annotations_path, start_frame, end_frame):
    human_features = np.empty([0, n_features])
    for i in range(start_frame, end_frame+1):
        # indicating if frame is valid
        valid = is_valid(annotations_path, i)
        # indicating if body, face, and hands are not included in => need to be predicted
        to_predict = is_to_predict(annotations_path, i)
        if valid and not to_predict:
            # landmarks_body (10, 3)
            landmarks_body, _ = parse_body_at(annotations_path, i)[1:]
            # landmarks_face (28, 3)
            landmarks_face, _ = parse_face_at(annotations_path, i)[1:]
            # landmarks_rhand (20,3)
            landmarks_rhand, _ = parse_rhand_at(annotations_path, i)[1:]
            # landmarks_lhand (20,3)
            landmarks_lhand, _ = parse_lhand_at(annotations_path, i)[1:]
            # Still, there are some errors in the validation data => double check
            if (landmarks_rhand is not None) and (landmarks_lhand is not None):
                landmarks_body = landmarks_body[:,0:2] # getting only X, Y
                landmarks_face = landmarks_face[:,0:2] # getting only X, Y
                landmarks_rhand = landmarks_rhand[:,0:2] # getting only X, Y
                landmarks_lhand = landmarks_lhand[:,0:2] # getting only X, Y
                landmarks_body = landmarks_body.flatten()
                landmarks_face = landmarks_face.flatten()
                landmarks_rhand = landmarks_rhand.flatten()
                landmarks_lhand = landmarks_lhand.flatten()
                current_frame = np.hstack([landmarks_body, landmarks_face, landmarks_rhand, landmarks_lhand])
                human_features = np.vstack((human_features, current_frame))
            else:
                landmarks_body = landmarks_body[:,0:2] # getting only X, Y
                landmarks_face = landmarks_face[:,0:2] # getting only X, Y
                landmarks_rhand = np.repeat(landmarks_body[9:10, 0:2], 20, axis=0) # right-wrist
                landmarks_lhand = np.repeat(landmarks_body[8:9, 0:2], 20, axis=0) # left-wrist
                landmarks_body = landmarks_body.flatten()
                landmarks_face = landmarks_face.flatten()
                landmarks_rhand = landmarks_rhand.flatten()
                landmarks_lhand = landmarks_lhand.flatten()
                current_frame = np.hstack([landmarks_body, landmarks_face, landmarks_rhand, landmarks_lhand])
                human_features = np.vstack((human_features, current_frame))
    return human_features

def data_provider(sess_motion_FC1, sess_motion_FC2, sess_task, sess_std_FC1, sess_std_FC2, sess_mean_FC1, sess_mean_FC2):

    if sess_task == 'FC1_T': # generate FC1 => consider FC2 as contextual input
        p_observed = sess_motion_FC2
        p_forecast = sess_motion_FC1
        p_forecast_std = sess_std_FC1
        p_forecast_mean = sess_mean_FC1

    elif sess_task == 'FC2_T': # generate FC2 => consider FC1 as contextual input
        p_observed = sess_motion_FC1
        p_forecast = sess_motion_FC2
        p_forecast_std = sess_std_FC2
        p_forecast_mean = sess_mean_FC2

    p_observed = np.array(p_observed, dtype="float32")
    p_forecast = np.array(p_forecast, dtype="float32")

    context_inp   = p_observed[:,0:n_features_h1]

    encoder_inp_B = p_forecast[0:(source_len-1), 0:n_features_h2_B]
    decoder_inp_B = np.tile(np.expand_dims(p_forecast[-1,0:n_features_h2_B], axis=0), (target_len,1))
    

    
    encoder_inp_F = p_forecast[0:(source_len-1), n_features_h2_B:n_features_h2_B+n_features_h2_F]
    decoder_inp_F = np.tile(np.expand_dims(p_forecast[-1,n_features_h2_B:n_features_h2_B+n_features_h2_F], axis=0), (target_len,1))

    encoder_inp_H = p_forecast[0:(source_len-1), n_features_h2_B+n_features_h2_F:]
    decoder_inp_H = np.tile(np.expand_dims(p_forecast[-1,n_features_h2_B+n_features_h2_F:], axis=0), (target_len,1))

    
    return p_forecast_std, p_forecast_mean, context_inp, encoder_inp_B, decoder_inp_B, encoder_inp_F, decoder_inp_F, encoder_inp_H, decoder_inp_H

def generate_json(size, num_layers, model_path, model_index, annotations_dir, segments_path, json_folder):
    ###Place holder ###
    dtype=tf.float32
    tf_context_inputs = tf.placeholder(dtype, shape=[None, source_len, n_features_h1], name="context_inputs")
    tf_enc_in_B =  tf.placeholder(dtype, shape=[None, source_len-1, n_features_h2_B], name="enc_in_B")
    tf_dec_in_B =  tf.placeholder(dtype, shape=[None, target_len, n_features_h2_B], name="dec_in_B")
    tf_dec_out_B = tf.placeholder(dtype, shape=[None, target_len, n_features_h2_B], name="dec_out_B")
    tf_enc_in_F =  tf.placeholder(dtype, shape=[None, source_len-1, n_features_h2_F], name="enc_in_F")
    tf_dec_in_F =  tf.placeholder(dtype, shape=[None, target_len, n_features_h2_F], name="dec_in_F")
    tf_dec_out_F = tf.placeholder(dtype, shape=[None, target_len, n_features_h2_F], name="dec_out_F")
    tf_enc_in_H =  tf.placeholder(dtype, shape=[None, source_len-1, n_features_h2_H], name="enc_in_F")
    tf_dec_in_H =  tf.placeholder(dtype, shape=[None, target_len, n_features_h2_H], name="dec_in_F")
    tf_dec_out_H = tf.placeholder(dtype, shape=[None, target_len, n_features_h2_H], name="dec_out_F")
    ###Place holder ###

    ### Models ###
    model = GAN_models(
        (source_len, n_features_h1), 
        context_len, 
        (target_len, n_features_h2_B, n_features_h2_F, n_features_h2_H), 
        size, # hidden recurrent layer size = 1024
        num_layers)

    context_inputs = tf_context_inputs
    outputs_human = model.EncoderP(context_inputs)

    # (None, 30, 10) - (None, source_seq_size[0], context_len)
    context_tiled = tf.tile(tf.expand_dims(tf.cast(outputs_human, dtype=tf.float32), axis=1),
                                multiples=[1, target_len, 1])
    ## 
    dec_in_B = tf.concat([tf_dec_in_B, context_tiled], 2)
    dec_in_F = tf.concat([tf_dec_in_F, context_tiled], 2)
    dec_in_H = tf.concat([tf_dec_in_H, context_tiled], 2)

    enc_in_B = tf.transpose(tf_enc_in_B, [1, 0, 2])
    dec_in_B = tf.transpose(dec_in_B, [1, 0, 2])
    dec_out_B = tf.transpose(tf_dec_out_B, [1, 0, 2])

    enc_in_B = tf.reshape(enc_in_B, [-1, n_features_h2_B])
    dec_in_B = tf.reshape(dec_in_B, [-1, n_features_h2_B + context_len])
    dec_out_B = tf.reshape(dec_out_B, [-1, n_features_h2_B])

    enc_in_B = tf.split(enc_in_B, source_len-1, axis=0)
    dec_in_B = tf.split(dec_in_B, target_len, axis=0)
    dec_out_B = tf.split(dec_out_B, target_len, axis=0)

    G_B = model.GeneratorB(enc_in_B, dec_in_B)

    enc_in_F = tf.transpose(tf_enc_in_F, [1, 0, 2])
    dec_in_F = tf.transpose(dec_in_F, [1, 0, 2])
    dec_out_F = tf.transpose(tf_dec_out_F, [1, 0, 2])

    enc_in_F = tf.reshape(enc_in_F, [-1, n_features_h2_F])
    dec_in_F = tf.reshape(dec_in_F, [-1, n_features_h2_F + context_len])
    dec_out_F = tf.reshape(dec_out_F, [-1, n_features_h2_F])

    enc_in_F = tf.split(enc_in_F, source_len-1, axis=0)
    dec_in_F = tf.split(dec_in_F, target_len, axis=0)
    dec_out_F = tf.split(dec_out_F, target_len, axis=0)

    G_F = model.GeneratorF(enc_in_F, dec_in_F)

    enc_in_H = tf.transpose(tf_enc_in_H, [1, 0, 2])
    dec_in_H = tf.transpose(dec_in_H, [1, 0, 2])
    dec_out_H = tf.transpose(tf_dec_out_H, [1, 0, 2])

    enc_in_H = tf.reshape(enc_in_H, [-1, n_features_h2_H])
    dec_in_H = tf.reshape(dec_in_H, [-1, n_features_h2_H + context_len])
    dec_out_H = tf.reshape(dec_out_H, [-1, n_features_h2_H])

    enc_in_H = tf.split(enc_in_H, source_len-1, axis=0)
    dec_in_H = tf.split(dec_in_H, target_len, axis=0)
    dec_out_H = tf.split(dec_out_H, target_len, axis=0)

    G_H = model.GeneratorH(enc_in_H, dec_in_H)

    with create_session() as sess:
        checkpoint = model_path + "/checkpoint-" + str(model_index)+"/checkpoint-" + str(model_index)
        gan_saver1 = tf.train.Saver(var_list=tf.trainable_variables())
        gan_saver1.restore(sess, checkpoint)
        print("Successfully restored model: ", checkpoint)

        """ 
        Loading the segment file 
        Getting the corresponding previous FC1, FC2 motion frames
        Feeding to Generator, produce the predicted frame
        """
        predictions_df = pd.read_csv(segments_path)
        data_idx = 0
        submission_dict = {}
        for session, session_df in predictions_df.groupby("session"):
            session = f"{int(session):06d}"
            session_dict = {}
            for task, task_df in session_df.groupby("task"):
                task_dict = {}
                for i in range(len(task_df.index)):
                    frame_init = task_df.iloc[i]["init"]
                    frame_final = task_df.iloc[i]["end"]
                    sess_motion_FC1, sess_motion_FC2, sess_std_FC1, sess_std_FC2, sess_mean_FC1, sess_mean_FC2 = generate_seq(annotations_dir, session, frame_init)
                    p_forecast_std, p_forecast_mean, g_context_inp, g_encoder_inp_B, g_decoder_inp_B, g_encoder_inp_F, g_decoder_inp_F, g_encoder_inp_H, g_decoder_inp_H  = data_provider(sess_motion_FC1, sess_motion_FC2, task, sess_std_FC1, sess_std_FC2, sess_mean_FC1, sess_mean_FC2)
                    context_inp = np.expand_dims(g_context_inp, axis=0)
                    encoder_inp_B = np.expand_dims(g_encoder_inp_B, axis=0)
                    decoder_inp_B = np.expand_dims(g_decoder_inp_B, axis=0) 

                    encoder_inp_F = np.expand_dims(g_encoder_inp_F, axis=0)
                    decoder_inp_F = np.expand_dims(g_decoder_inp_F, axis=0) 

                    encoder_inp_H = np.expand_dims(g_encoder_inp_H, axis=0)
                    decoder_inp_H = np.expand_dims(g_decoder_inp_H, axis=0)

                    generated_B = sess.run(G_B, feed_dict={tf_context_inputs: context_inp, 
                                                                tf_enc_in_B: encoder_inp_B, 
                                                                tf_dec_in_B: decoder_inp_B})
                    generated_F = sess.run(G_F, feed_dict={tf_context_inputs: context_inp, 
                                                                tf_enc_in_F: encoder_inp_F, 
                                                                tf_dec_in_F: decoder_inp_F})
                    generated_H = sess.run(G_H, feed_dict={tf_context_inputs: context_inp, 
                                                                tf_enc_in_H: encoder_inp_H, 
                                                                tf_dec_in_H: decoder_inp_H})

                    generated_B = np.squeeze(np.transpose(generated_B, [1, 0, 2]))
                    generated_F = np.squeeze(np.transpose(generated_F, [1, 0, 2]))
                    generated_H = np.squeeze(np.transpose(generated_H, [1, 0, 2]))
                    generated_motion = np.concatenate([generated_B, generated_F, generated_H], axis=1)
                    # [n_future, n_fetures] - [50,156]
                    generated_motion = denormalize_motion_std(generated_motion, std_val=p_forecast_std, mean_pose=p_forecast_mean)
                    f_idx = 0
                    for frame in range(frame_init, frame_final + 1):
                        task_dict[f"{frame:05d}"] = {
                            "face":       (generated_motion[f_idx,n_features_B:n_features_B+n_features_F].reshape([n_features_F//2, 2])).tolist(),                            # the SECOND 28*2 elements
                            "body":       (generated_motion[f_idx,0:n_features_B].reshape([n_features_B//2, 2])).tolist(),                                                    # the FIRST  10*2 elements
                            "left_hand":  (generated_motion[f_idx,n_features_B+n_features_F+n_features_RH:].reshape([n_features_LH//2, 2])).tolist(),                         # the FOURTH 20*3 elements 
                            "right_hand": (generated_motion[f_idx,n_features_B+n_features_F:n_features_B+n_features_F+n_features_RH].reshape([n_features_RH//2, 2])).tolist()      # the THIRD  20*3 elements
                        }
                        f_idx += 1
                    print("data_idx", data_idx, "session", session, "task", task, "frame_init", frame_init, "frame_final", frame_final, "generated_motion shape of:", np.shape(generated_motion))
                    data_idx += 1
                session_dict[task] = task_dict
            submission_dict[session] = session_dict
        
        if not os.path.exists(json_folder):
            os.makedirs(json_folder)
        output_file = os.path.join(json_folder, PRED_FILENAME)
        with open(output_file, 'w') as json_file:
            json.dump(submission_dict, json_file, ensure_ascii=False)
        print(f"Submission file saved to '{output_file}'")

if __name__ == "__main__":
    args = get_arguments()
    annotations_valid = generate_json(args.size, args.num_layers, args.model_dir, args.model_index, \
        args.annotations_dir, args.segments_path, args.json_folder)
