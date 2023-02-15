FROM python:latest

RUN mkdir /app
WORKDIR /app
ADD . /app
RUN pip install -r requirements.txt

EXPOSE 8501
CMD ["streamlit", "run", "/app/app.py"]
