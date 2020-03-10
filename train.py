import argparse
import json
import logging
import os
import sys
import pickle
import joblib

import pandas as pd
import matplotlib.pyplot as plt
import math
import numpy as np
import logging

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Dropout

from model import AutoEncRec, masked_mse
from sklearn.model_selection import train_test_split

from scipy.sparse.csr import csr_matrix
from scipy.sparse import save_npz, load_npz

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

def train(args):
    """ Train Model

    """      
    print(args)
    
    train, test, df_movie_idx = prepare_and_split_dataset(args)

    print("Load Model")
    # Params
    model_params = dict(
        input_dim=train.shape[1], 
        n_dims=[int(i) for i in args.n_dims.split(',')]
    )

    # Model
    model = AutoEncRec(**model_params)
    model.compile(optimizer=args.optimizer, loss='mse') #masked_mse(0.0)
    model.build((1, train.shape[1]))
    print(model.summary())

    print("training...")
    print(train.shape)
    hist = model.fit(train, train, 
                    validation_split=0.1, 
                    batch_size = args.batch_size, 
                    epochs = args.epochs)

    print(df_movie_idx.head())

    # # Model Info 
    model_info = {'model_init': model_params}

    # # Save Model Information
    save_model(args.model_dir, model, model_info, df_movie_idx)

    # # Load Model
    model = model_fn(args.model_dir)
    model.compile(optimizer=args.optimizer, loss='mse') #masked_mse(0.0)

    # # Evaluation Model
    loss      = model.evaluate(test,  test, verbose=2)
    metrics   = {'loss': loss}
    print("Metrics", metrics)

    # # Save Train Output
    save_output(args.output_dir, vars(args), metrics)
    
def prepare_and_split_dataset(args):
    """ Load and Prepare dataset train, test and valid to Loader

    Parameters:
      args            -- Args

    Return:
      df_movie_idx    -- idx of Movie Dataframe
      df_train        -- Train DataFrame
      df_test         -- Test DataFrame
    """    
    logger.info("Prepare Dataset")

    df = pd.read_csv(args.train_data_dir)

    print("Dataset ", df.shape)
    print(df.head())

    # Extract Idx of Account and Movie
    df_user_idx = df[['userId']].drop_duplicates().reset_index(drop=True)\
                    .reset_index().rename(columns={'index': 'userId_idx'})

    df_movie_idx = df[['movieId']].drop_duplicates().reset_index(drop=True)\
                        .reset_index().rename(columns={'index': 'movieId_idx'})


    # Merge Dataset
    df_train = df.merge(
                    df_user_idx, on='userId', how='inner')\
                .merge(
                    df_movie_idx, on='movieId', how='inner')

    print("Dataset Transform", df.shape)

    print(df_train.head())

    print("Create sparce matrix")
    # Create sparce matrix
    spc_data = csr_matrix((df_train['rating'].values, (df_train.userId_idx.values, df_train.movieId_idx.values)), 
                    shape=(len(df_user_idx), len(df_movie_idx)))


    print("Group dataset per Account")
    # Split Train and Valid Dataset
    df_train, df_test = train_test_split(spc_data.toarray(), test_size=0.2, random_state=args.seed)

    return df_train, df_test, df_movie_idx

def save_model(model_dir: str, model:Model, model_info: dict(), df_movie_idx: pd.DataFrame):
    """ Save Tensorflow model and Metadata to `model_dir` directory.

    Parameters:
      model_dir  -- Path to loads saved model binary file(s)
      model      -- Tensorflow nn.Module
      model_info -- Metadata Model
      df_movie_idx -- Dataframe with idx movie
    """    
    logger.info("Saving the model.")

    # Save movie idx
    movie2dict = dict(zip(df_movie_idx.movieId, df_movie_idx.movieId_idx))
    joblib.dump(movie2dict, os.path.join(model_dir, 'movies_idx.pkl'))

    # Model Information
    with open(os.path.join(model_dir, "model_info.json"), "w") as model_info_file:
        json.dump(model_info, model_info_file, indent=4)

    # Save Model
    path = os.path.join(model_dir, 'model.ckpt')
    model.save_weights(path)

def save_output(output_dir: str, args: dict(), metrics: dict()):
    """ Save Tensorflow model and Metadata to `model_dir` directory.

    Parameters:
      output_dir -- Path to save files
      args       -- Dict Args
      metrics    -- Evaluate metrics
    """    

    # Arguments
    with open(os.path.join(output_dir, "args.json"), "w") as args_file:
        json.dump(args, args_file, indent=4)

    # Metrics
    with open(os.path.join(output_dir, "metrics.json"), "w") as metrics_file:
        json.dump(metrics, metrics_file, indent=4)

def model_fn(model_dir: str):
    """ Load the Tensorflow model from the `model_dir` directory.

    Parameters:
      model_dir -- Path to loads saved model binary file(s)

    Return:
      model     -- Tensorflow nn.Module
    """
    print("Loading model.")

    # Load Tensorflow Model
    # ----------------------------------
    #
    with open(os.path.join(model_dir, "model_info.json"), "r") as model_info_file:
        model_info = json.load(model_info_file)

        model = AutoEncRec(**model_info['model_init'])
    
    model.load_weights(os.path.join(model_dir, 'model.ckpt')).expect_partial()

    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Model Params
    parser.add_argument('--n-dims', type=str, default="64, 32, 64",help='')


    # Train Params
    parser.add_argument('--batch-size', type=int, default=10, metavar='N',
                        help='input batch size for training (default: 500)')
    
    parser.add_argument('--epochs', type=int, default=20, metavar='N',
                        help='number of epochs to train (default: 2)')

    parser.add_argument('--optimizer', type=str, default="adam", help='')

    parser.add_argument('--optimizer-params', type=json.loads, default='{}', help='')

    parser.add_argument('--seed', type=int, default=42, metavar='S', help='random seed (default: 1)')
    
    # Container environment
    parser.add_argument('--hosts', type=list, default=json.loads(os.environ['SM_HOSTS']))
    
    parser.add_argument('--current-host', type=str, default=os.environ['SM_CURRENT_HOST'])
    
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])

    parser.add_argument('--output-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    
    parser.add_argument('--train-data-dir', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    
    parser.add_argument('--test-data-dir', type=str, default=os.environ['SM_CHANNEL_VALIDATION'])
    
    parser.add_argument('--num-gpus', type=int, default=os.environ['SM_NUM_GPUS'])

    train(parser.parse_args())