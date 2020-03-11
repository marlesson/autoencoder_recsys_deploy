# https://cloud.google.com/ai-platform/training/docs/packaging-trainer
import logging

import json
import os
import sys
import joblib
from datetime import datetime
import time

import pandas as pd
import numpy as np

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

class AutoEncPredictor(object):

  """An example Predictor for an AI Platform custom prediction routine."""

  def __init__(self, model):
    """Stores artifacts for prediction. Only initialized via `from_path`.
    """
    self._model = model

  def predict(self, instances, **kwargs):
    """Performs custom prediction.

    Preprocesses inputs, then performs prediction using the trained Keras
    model.    
    """

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

    input_object      = instances[0]

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

    return [result]



@classmethod
  def from_path(cls, model_dir):
      """Creates an instance of MyPredictor using the given path.

      This loads artifacts that have been copied from your model directory in
      Cloud Storage. MyPredictor uses them during prediction.

      Args:
          model_dir: The local directory that contains the trained Keras
              model and the pickled preprocessor instance. These are copied
              from the Cloud Storage model directory you provide when you
              deploy a version resource.

      Returns:
          An instance of `AutoEncPredictor`.
      """

      # Load Tensorflow Model
      # ----------------------------------
      #
      with open(os.path.join(model_dir, "model_info.json"), "r") as model_info_file:
          model_info = json.load(model_info_file)

          model = AutoEncRec(**model_info['model_init'])
          model.item_idx = joblib.load(os.path.join(model_dir, 'movies_idx.pkl'))

      model.load_weights(os.path.join(model_dir, 'model.ckpt')).expect_partial()

      return cls(model)