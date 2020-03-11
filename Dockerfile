FROM python:3.6
WORKDIR /app

COPY requirements.txt /app
RUN pip install -r ./requirements.txt

COPY flask_predictor.py /app
COPY output /app/model

EXPOSE 5000

CMD ["python", "flask_predictor.py"]~