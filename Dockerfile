FROM nvidia/cuda:10.2-base

RUN mkdir /app
WORKDIR /app
ADD requirements.txt /app/
RUN pip install -r requirements.txt

ADD . /app

RUN rm -rf tmp

CMD ["python", "run.py"]
