FROM python:3.12

RUN pip install -U pip 

WORKDIR /streamlit

COPY [ "requirements.txt", "./" ]

RUN pip install -r requirements.txt

RUN apt-get update && apt-get install -y build-essential

COPY [ "reddit_model_pipeline.joblib", "app.py", "./"]

EXPOSE 8000

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]