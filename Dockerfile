FROM python:3.6

WORKDIR /lyons_rianne_hw4

COPY . /lyons_rianne_hw4

RUN pip install numpy

RUN pip install cython

RUN pip install dynet

CMD python /lyons_rianne_hw4/asgn4.py
