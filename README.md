## Docker

https://cloud.google.com/container-registry/docs/advanced-authentication

docker build -t recsys-autoenc:latest .
docker run -a stderr  -p 5000:5000 recsys-autoenc:latest

docker build -f Dockerfile -t gcr.io/cartola-202712/recsys-autoenc:latest .
docker push gcr.io/cartola-202712/recsys-autoenc:latest

gcloud compute instances create-with-container recsys-autoenc \
      --container-image gcr.io/cartola-202712/recsys-autoenc:latest \
      --zone=us-east1 \
      --tags http-server

gcloud compute firewall-rules create allow-http \
    --allow tcp:5000 --target-tags http-server

## GCP Functions

gcloud functions deploy recommender --runtime python37 --trigger-http --memory 1024 --region=us-east1