'''
Authors:      Nguyen Tan Viet Tuyen, Oya Celiktutan
Email:       tan_viet_tuyen.nguyen@kcl.ac.uk
Affiliation: SAIR LAB, King's College London
Project:     LISI -- Learning to Imitate Nonverbal Communication Dynamics for Human-Robot Social Interaction

Python version: 3.6
'''

import glob
import numpy as np
import os
import h5py
import argparse
import pickle5 as pickle
from sklearn.utils import shuffle
from lib.config import *
from lib.annotations_parser import is_valid, is_to_predict, parse_face_at, parse_lhand_at, parse_rhand_at, parse_body_at

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, required=False, default="./dataset")
    parser.add_argument('--annotations_dir', type=str, required=True, default="/path_to/talk_annotations_train")
    return parser.parse_args()

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

def get_all_data(annotations_path, start_frame, end_frame):
    human_features = np.empty([0, n_features])
    patience = 5
    trial = 0
    for i in range(start_frame, end_frame):
        valid = is_valid(annotations_path, i)
        to_predict = is_to_predict(annotations_path, i)
        if valid and not to_predict:
            landmarks_body, _ = parse_body_at(annotations_path, i)[1:]
            landmarks_face, _ = parse_face_at(annotations_path, i)[1:]
            landmarks_rhand, _ = parse_rhand_at(annotations_path, i)[1:]
            landmarks_lhand, _ = parse_lhand_at(annotations_path, i)[1:]
            if (landmarks_body is not None) and (landmarks_face is not None) and (landmarks_rhand is not None) and (landmarks_lhand is not None):
                landmarks_body = landmarks_body[:,0:2] 
                landmarks_face = landmarks_face[:,0:2] 
                landmarks_rhand = landmarks_rhand[:,0:2] 
                landmarks_lhand = landmarks_lhand[:,0:2] 
                landmarks_body = landmarks_body.flatten()
                landmarks_face = landmarks_face.flatten()
                landmarks_rhand = landmarks_rhand.flatten()
                landmarks_lhand = landmarks_lhand.flatten()
                current_frame = np.hstack([landmarks_body, landmarks_face, landmarks_rhand, landmarks_lhand])
                human_features = np.vstack((human_features, current_frame))
            else:
                trial+=1
                if (i > start_frame) and (trial <= patience):
                    human_features = np.vstack((human_features, current_frame))
                else:
                    return None
        else:
            return None
    return human_features

def generate_seq(annotations, session):
    sess_motion_FC1 = []
    sess_motion_FC2 = []
    sess_name = []
    annotations_path_FC1 = os.path.join(annotations, session, "FC1_T", ANNOTATIONS_FILE)
    annotations_path_FC2 = os.path.join(annotations, session, "FC2_T", ANNOTATIONS_FILE)
    # annotations_path_FC1 = os.path.join(annotations, session, "FC1_T", "annotations_cleaned.hdf5")
    # annotations_path_FC2 = os.path.join(annotations, session, "FC2_T", "annotations_cleaned.hdf5")
    f_FC1 = h5py.File(annotations_path_FC1, "r") 
    f_FC2 = h5py.File(annotations_path_FC2, "r")
    assert os.path.exists(annotations_path_FC1) or annotations_path_FC1.split(".")[-1].lower() != "hdf5", "HDF5 FC1_T file could not be opened."
    assert os.path.exists(annotations_path_FC2) or annotations_path_FC2.split(".")[-1].lower() != "hdf5", "HDF5 FC2_T file could not be opened."
    assert len(f_FC1) == len(f_FC2), "len(f_FC1) != len(f_FC2)" 
    for start in range(0, len(f_FC1) - n_past - n_future - window_size, window_size):
        human_features_FC1 = get_all_data(annotations_path_FC1, start, start+n_past+n_future)
        human_features_FC2 = get_all_data(annotations_path_FC2, start, start+n_past+n_future)
        if (human_features_FC1 is not None) and (human_features_FC2 is not None):
            sess_motion_FC1.append(np.array(human_features_FC1))
            sess_motion_FC2.append(np.array(human_features_FC2))
            sess_name.append(session)

    sess_motion_FC1, sess_std_FC1, sess_mean_FC1 = normalize_motion_std(sess_motion_FC1) 
    sess_motion_FC2, sess_std_FC2, sess_mean_FC2 = normalize_motion_std(sess_motion_FC2) 

    return sess_motion_FC1, sess_motion_FC2, sess_name, sess_std_FC1, sess_std_FC2, sess_mean_FC1, sess_mean_FC2

def main(annotations_dir, dataset_dir):
    train_motion_FC1 = np.empty([0, n_past+n_future, n_features])
    train_motion_FC2 = np.empty([0, n_past+n_future, n_features])
    train_sess_name = []
    train_sess_name_ = []
    train_sess_std_FC1 = []
    train_sess_std_FC2 = []
    train_sess_mean_FC1 = []
    train_sess_mean_FC2 = []

    for folder_name in glob.glob(os.path.join(annotations_dir, '*')):
        session = folder_name[-6:]
        
        motion_FC1, motion_FC2, sess_name, sess_max_FC1, sess_max_FC2, sess_mean_FC1, sess_mean_FC2 = generate_seq(annotations_dir, session)
        print("session: ", session, "motion_FC1", np.shape(motion_FC1), "motion_FC2", np.shape(motion_FC2), 
                "sess_name", np.shape(sess_name), "sess_max_FC1", np.shape(sess_max_FC1), "sess_max_FC2", np.shape(sess_max_FC2), 
                "sess_mean_FC1", np.shape(sess_mean_FC1), "sess_mean_FC2", np.shape(sess_mean_FC2))
        if len(motion_FC1)>0 and len(motion_FC2)>0:
            train_motion_FC1 = np.vstack((train_motion_FC1, motion_FC1))
            train_motion_FC2 = np.vstack((train_motion_FC2, motion_FC2))
            train_sess_name = np.append(train_sess_name, sess_name, axis=0)
            ##
            train_sess_name_ = np.append(train_sess_name_, session)
            train_sess_std_FC1 = np.append(train_sess_std_FC1, sess_max_FC1)
            train_sess_std_FC2 = np.append(train_sess_std_FC2, sess_max_FC2)
            train_sess_mean_FC1 = np.append(train_sess_mean_FC1, sess_mean_FC1)
            train_sess_mean_FC2 = np.append(train_sess_mean_FC2, sess_mean_FC2)

    print("train_motion_FC1", np.shape(train_motion_FC1), "train_motion_FC2", np.shape(train_motion_FC2), "train_sess_name", np.shape(train_sess_name)) 
    print("train_sess_name_", np.shape(train_sess_name_),
          "train_sess_std_FC1", np.shape(train_sess_std_FC1), "train_sess_std_FC2", np.shape(train_sess_std_FC2), 
          "train_sess_mean_FC1", np.shape(train_sess_mean_FC1), "train_sess_mean_FC2", np.shape(train_sess_mean_FC2)) 
    print('Shuffling dataset...')
    train_motion_FC1, train_motion_FC2, train_sess_name = shuffle(train_motion_FC1, train_motion_FC2, train_sess_name, random_state=0)
    data = {
    'train_sess_name': train_sess_name,
    'train_motion_FC1': train_motion_FC1,
    'train_motion_FC2': train_motion_FC2,
    }
    print('Pickling dataset...')
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)
    UDIVA_2d = os.path.join(dataset_dir, "UDIVA_2d.pickle")
    UDIVA_values = os.path.join(dataset_dir, "UDIVA_values.pickle")

    with open(UDIVA_2d, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
        print('done')

    data_1 = {
    'train_sess_name_': train_sess_name_,
    'train_sess_std_FC1': train_sess_std_FC1, 
    'train_sess_std_FC2': train_sess_std_FC2, 
    'train_sess_mean_FC1': train_sess_mean_FC1,
    'train_sess_mean_FC2': train_sess_mean_FC2,
    }
    print('Pickling dataset...')
    with open(UDIVA_values, 'wb') as f:
        pickle.dump(data_1, f, pickle.HIGHEST_PROTOCOL)
        print('done')
if __name__ == '__main__':
    args = get_arguments()
    main(args.annotations_dir, args.dataset_dir)