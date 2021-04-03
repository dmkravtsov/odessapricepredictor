FROM python:3.8.5-buster
ENV APP_ROOT /src
RUN mkdir ${APP_ROOT}
WORKDIR ${APP_ROOT}}
ADD . ${APP_ROOT}
RUN pip install --upgrade pip
RUN pip3 install -r ${APP_ROOT}/requirements.txt  
EXPOSE 5000