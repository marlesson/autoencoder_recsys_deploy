import json
import os
import sys
import joblib
from datetime import datetime
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Dropout
from google.cloud import storage

from scipy.sparse.csr import csr_matrix
from scipy.sparse import save_npz, load_npz

# Constants
#
JSON_CONTENT_TYPE        = 'application/json'
BUCKET_DIST_PATH         = 'ia-plataform-model'

# We keep model as global variable so we don't have to reload it in case of warm invocations
model = None

class AutoEncRec(Model):
    '''Vanilla Autoencer'''
    
    def __init__(self, input_dim, n_dims = [64, 32, 64], dropout_rate = 0.2):
        super(AutoEncRec,self).__init__()
        self.version   = "Vanilla Autoencorder 1.0"
        
        self.input_dim = input_dim

        self.enc_1 = Dense(n_dims[0], input_shape = (input_dim, ), activation='selu')
        self.enc_2 = Dense(n_dims[1], activation='selu')
        self.dec_1 = Dense(n_dims[2], activation='selu')
        self.dec_2 = Dense(input_dim, activation='linear')     
        self.dropout = Dropout(dropout_rate)

        self.item_idx = None

    def encoder(self, x):
        net = self.enc_1(x)
        net = self.enc_2(net)
        return net
    
    def decoder(self, x):
        net = self.dec_1(x)
        net = self.dec_2(net)
        return net

    def call(self, inputs):
        net = self.decoder(self.dropout(self.encoder(inputs)))
        return net

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

def predict_fn(input_object: str, model: Model):
    """ Takes parsed input and make predictions with the model loaded by model_fn.
        Perform prediction on the deserialized object, with the loaded model

    Parameters:
      input_object -- Deserialize request_body we can perform prediction on
      model        -- Tensorflow Model

    Return:
      predictions  -- Dict Object with Ordering Predicted by RecSys Model
    """  
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

def download_metadata():
  """ Download Metadata Models BUCKET to /tmp
  """   
    
  def download_blob(bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(source_blob_name)

    blob.download_to_filename(destination_file_name)

    print('Blob {} downloaded to {}.'.format(source_blob_name,destination_file_name))

  download_blob(BUCKET_DIST_PATH, 'dist/model.ckpt.index', '/tmp/model.ckpt.index')
  download_blob(BUCKET_DIST_PATH, 'dist/model.ckpt.data-00000-of-00001', '/tmp/model.ckpt.data-00000-of-00001')
  download_blob(BUCKET_DIST_PATH, 'dist/model_info.json', '/tmp/model_info.json')
  download_blob(BUCKET_DIST_PATH, 'dist/movies_idx.pkl', '/tmp/movies_idx.pkl')

def recommender(request):
  """ Predictor Funcion
      gcloud functions deploy recommender --runtime python37 --trigger-http --memory 1024 --region=us-east1
  """     
  global model

  # Model load which only happens during cold starts
  if model is None:
    download_metadata()
    model = model_fn('/tmp/')

  data = request.get_json()
  pred = predict_fn(data, model)

  return json.dumps(pred), 200, {'Content-Type': JSON_CONTENT_TYPE}