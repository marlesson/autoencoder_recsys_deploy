## Train Local

SM_NUM_GPUS="1" SM_CHANNEL_VALIDATION="" SM_CHANNEL_TRAINING="data/movielens100k/ratings.csv" SM_OUTPUT_DATA_DIR="dist" SM_MODEL_DIR="dist" SM_CURRENT_HOST="0.0.0.0" SM_HOSTS="{}" python train.py  --epochs 20

## Docker

https://cloud.google.com/container-registry/docs/advanced-authentication

### Teste Local 
docker build -t recsys-autoenc:latest .
docker run -a stderr  -p 5000:5000 recsys-autoenc:latest

### Push GCP
docker build -f Dockerfile -t gcr.io/cartola-202712/recsys-autoenc:latest .
docker push gcr.io/cartola-202712/recsys-autoenc:latest

### Deploy
gcloud compute instances create-with-container recsys-autoenc-1 \
      --container-image gcr.io/cartola-202712/recsys-autoenc:latest \
      --zone=us-east1-d \
      --tags http-server

gcloud compute firewall-rules create allow-http \
    --allow tcp:5000 --target-tags http-server



## GCP Functions

gcloud functions deploy recommender --runtime python37 --trigger-http --memory 1024 --region=us-east1


## Floyd

floyd init recsys-autoenc