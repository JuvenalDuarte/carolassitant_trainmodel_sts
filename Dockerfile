FROM python:3.8

RUN mkdir /app
WORKDIR /app
ADD requirements.txt /app/

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

ADD . /app

RUN rm -rf tmp

CMD ["python", "run.py"]
