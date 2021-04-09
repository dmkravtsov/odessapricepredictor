FROM python:3.8.5-buster
RUN mkdir /app
COPY . /app
WORKDIR  /app
RUN pip install --upgrade pip
RUN pip3 install -r requirements.txt  
EXPOSE 8000
