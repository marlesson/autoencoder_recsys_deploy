# https://aws.amazon.com/pt/blogs/machine-learning/preprocess-input-data-before-making-predictions-using-amazon-sagemaker-inference-pipelines-and-scikit-learn/
import logging

import argparse
import json
import logging
import os
import sys
import pickle
import joblib
from datetime import datetime
import time

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


# Constants
#
JSON_CONTENT_TYPE        = 'application/json'

def model_fn(model_dir: str):
    """ Load the Tensorflow model from the `model_dir` directory.

    Parameters:
      model_dir -- Path to loads saved model binary file(s)

    Return:
      model     -- Tensorflow Model
    """
    print("Loading model.")

    # Load Tensorflow Model
    # ----------------------------------
    #
    with open(os.path.join(model_dir, "model_info.json"), "r") as model_info_file:
        model_info = json.load(model_info_file)

        model = AutoEncRec(**model_info['model_init'])
        model.item_idx = joblib.load(os.path.join(model_dir, 'movies_idx.pkl'))

    model.load_weights(os.path.join(model_dir, 'model.ckpt')).expect_partial()

    return model

def input_fn(request_body: str, content_type: str):
    """ Parses input data from different interface(HTTP request or Python function calling)

    Parameters:
      request_body -- An Json in str input 
        {
          "user_uuid": "1",
          "watched_movies": [23, 23, 21, 10]
        }      
      content_type -- Content Type of request body

    Return:
      data         -- Deserialize request_body we can perform prediction on
    """  
    print('Deserializing the input data.')

    if content_type == JSON_CONTENT_TYPE:
        data = json.loads(request_body)
        return data

    raise Exception('Requested unsupported ContentType in content_type: {}'.format(content_type))

def predict_fn(input_object: str, model: Model):
    """ Takes parsed input and make predictions with the model loaded by model_fn.
        Perform prediction on the deserialized object, with the loaded model

    Parameters:
      input_object -- Deserialize request_body we can perform prediction on
      model        -- Tensorflow Model

    Return:
      predictions  -- Dict Object with Ordering Predicted by RecSys Model
    """  
    logger.info("Calling model")
    start_time        = time.time()

    topN              = 10

    # User UUID
    user_uuid         = input_object['user_uuid']

    # Get watched movies
    watched_movies_idx = [model.item_idx[i] for i in input_object['watched_movies']]
    
    # Inverse IDX to ID
    inv_item_idx = dict((v, k) for k, v in model.item_idx.items())    

    # Transfomation Data to Sparse Data
    data_input = csr_matrix((np.ones(len(watched_movies_idx)), 
                         (np.zeros(len(watched_movies_idx)), watched_movies_idx)),
                        shape=(1, model.input_dim)).toarray()
    data_pred  = model.predict(data_input)[0]

    print("--- Inference time: %s seconds ---" % (time.time() - start_time))
    
    # Sorted Recommender List
     
    idx_pred    = list(set(list(range(model.input_dim))) - set(watched_movies_idx))
    
    sorted_pred = dict(
                    sorted(
                        zip(
                            list(idx_pred), 
                            list(data_pred[idx_pred].astype(float))
                        ), 
                    key=lambda x: x[1],
                reverse=True))
       
    # Result Format
    result = {
      "status": "Ok",
      "evaluation": {
        "user_uuid":          input_object['user_uuid'],
        "watched_movies":     input_object['watched_movies'],
        "recommended_movie_ids": [inv_item_idx[i] for i in list(sorted_pred.keys())[:topN]],
        "scores":             list(sorted_pred.values())[:topN],
        "datetime":           datetime.utcnow().isoformat(sep='T', timespec='milliseconds'),
        "modelVersion":       model.version,
      }
    }

    return result

def output_fn(prediction: hash, accept: str):        
    """ encodes prediction made by predict_fn and return to corresponding interface 
        (HTTP response or python function return)

    Parameters:
      prediction   -- Dict Object with Ordering Predicted by RecSys Model
      accept       -- Content Type of request body

    Return:
      result_body  -- Json Object with Ordering Predicted by RecSys Model
    """    
    logger.info('Serializing the generated output.')
    if accept == JSON_CONTENT_TYPE: return json.dumps(prediction), accept
    raise Exception('Requested unsupported ContentType in Accept: {}'.format(accept))

# def predictor(args):
#   request_body = '{"user_uuid": "1", "watched_movies": [1, 3114, 87222, 84944, 260, 1196, 1210, 2628, 79006, 2116, 7153, 5952]}'

#   model  = model_fn(args.model_dir)
#   data   = input_fn(request_body, JSON_CONTENT_TYPE)
#   pred   = predict_fn(data, model)
#   output = output_fn(pred, JSON_CONTENT_TYPE)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Container environment
    parser.add_argument('--hosts', type=list, default=json.loads(os.environ['SM_HOSTS']))
    
    parser.add_argument('--current-host', type=str, default=os.environ['SM_CURRENT_HOST'])
    
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])

    parser.add_argument('--output-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    
    parser.add_argument('--train-data-dir', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    
    parser.add_argument('--test-data-dir', type=str, default=os.environ['SM_CHANNEL_VALIDATION'])
    
    parser.add_argument('--num-gpus', type=int, default=os.environ['SM_NUM_GPUS'])

    predictor(parser.parse_args())