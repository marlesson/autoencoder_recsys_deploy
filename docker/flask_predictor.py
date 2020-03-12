
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Dropout
from datetime import datetime

from model import AutoEncRec, masked_mse
from predictor import model_fn, input_fn, predict_fn, output_fn
import numpy as np
import flask
import io
import time
from scipy.sparse.csr import csr_matrix
from scipy.sparse import save_npz, load_npz


app   = flask.Flask(__name__)
model = model_fn('dist')

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

@app.route("/recommender", methods=["POST"])
def recommender():
  """ Predictor Funcion
  """   
  pred = {}
  if flask.request.method == "POST":  
    # Post
    json = flask.request.json
    pred = predict_fn(json, model)

  return flask.jsonify(pred)

if __name__ == "__main__":
    app.run(host='0.0.0.0')