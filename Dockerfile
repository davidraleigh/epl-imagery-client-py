FROM python:3.5 as builder

MAINTAINER David Raleigh <david@echoparklabs.io>

RUN apt update

RUN pip3 install --upgrade pip && \
    pip3 install grpcio-tools

WORKDIR /opt/src/epl-imagery-api

COPY ./ ./

RUN python3 -mgrpc_tools.protoc -I=./proto/ --python_out=./epl/client/imagery --grpc_python_out=./epl/client/imagery ./proto/epl_imagery.proto



FROM python:3.5-slim

MAINTAINER David Raleigh <david@echoparklabs.io>

RUN apt update

RUN pip3 install --upgrade pip && \
    pip3 install grpcio && \
    pip3 install numpy

# TODO only for testing install

WORKDIR /opt/src/epl-imagery-api

COPY --from=builder /opt/src/epl-imagery-api /opt/src/epl-imagery-api

RUN python setup.py install

ARG GRPC_SERVICE_PORT=50051
ARG GRPC_SERVICE_HOST=localhost

ENV GRPC_SERVICE_PORT ${GRPC_SERVICE_PORT}
ENV GRPC_SERVICE_HOST ${GRPC_SERVICE_HOST}
