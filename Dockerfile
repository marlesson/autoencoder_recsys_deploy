FROM python:3.6
WORKDIR /app

COPY requirements.txt /app
RUN pip install -r ./requirements.txt

COPY model.py /app
COPY predictor.py /app
COPY flask_predictor.py /app
COPY dist /app/dist

EXPOSE 5000

CMD ["python", "flask_predictor.py"]~