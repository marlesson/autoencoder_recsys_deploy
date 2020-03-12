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

## GCP IA-Platform

TRAINER_PACKAGE_PATH="/trainer"
now=$(date +"%Y%m%d_%H%M%S")
JOB_NAME="autoenc_$now"
MAIN_TRAINER_MODULE="trainer.train"
JOB_DIR="gs://ia-plataform-model/job/output/path"
PACKAGE_STAGING_PATH="gs://ia-plataform-model/staging/path"
REGION="us-east1"
RUNTIME_VERSION="1.14"

gcloud ai-platform jobs submit training $JOB_NAME \
        --scale-tier basic \
        --package-path $(pwd)$TRAINER_PACKAGE_PATH \
        --module-name $MAIN_TRAINER_MODULE \
        --job-dir $JOB_DIR \
        --region $REGION \
        -- \
        --epochs=10 \
        --train-data-dir "gs://ia-plataform-model/dataset/ratings.csv" \
        --output-dir "dist" \
        --sm-model-dir "dist"


## GCP Functions

gcloud functions deploy recommender --runtime python37 --trigger-http --memory 1024 --region=us-east1


## Floyd

floyd init recsys-autoenc

floyd run --cpu --env tensorflow-2.1 'SM_NUM_GPUS="1" SM_CHANNEL_VALIDATION="" SM_CHANNEL_TRAINING="data/movielens100k/ratings.csv" SM_OUTPUT_DATA_DIR="dist" SM_MODEL_DIR="dist" SM_CURRENT_HOST="0.0.0.0" SM_HOSTS="{}" python train.py  --epochs 20'
