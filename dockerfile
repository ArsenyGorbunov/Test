FROM digi0ps/python-opencv

RUN CPUCORES=$(getconf _NPROCESSORS_ONLN)
 
#Essential packages
RUN apt update && apt install -y build-essential cmake pkg-config \
    libx11-dev libatlas-base-dev libgtk-3-dev libboost-python-dev \
    && pip install numpy

COPY . /app
WORKDIR /app

RUN pip install -r requirements.txt

CMD python ./yolo_tracking.py