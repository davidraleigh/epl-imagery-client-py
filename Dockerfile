FROM python:3.5 as builder

MAINTAINER David Raleigh <david@echoparklabs.io>

RUN apt update

RUN pip3 install --upgrade pip && \
    pip3 install grpcio-tools

WORKDIR /opt/src/epl-imagery-api

COPY ./ ./

# firgured out the package defintin by looking at comments in this issue https://github.com/google/protobuf/issues/2283
RUN python3 -mgrpc_tools.protoc -I=./proto/ --python_out=./ --grpc_python_out=./ ./proto/epl/grpc/imagery/epl_imagery.proto



FROM python:3.5-slim

MAINTAINER David Raleigh <david@echoparklabs.io>

RUN DEBIAN_FRONTEND=noninteractive apt update && \
    pip3 install --upgrade pip && \
    pip3 install grpcio && \
    pip3 install numpy

# TODO only for testing install
RUN pip3 install pytest lxml requests shapely
# TODO only for testing install

WORKDIR /opt/src/epl-imagery-api

COPY --from=builder /opt/src/epl-imagery-api /opt/src/epl-imagery-api

RUN python3 setup.py install

ARG GRPC_SERVICE_HOST=localhost
ARG GRPC_SERVICE_PORT=50051

ENV IMAGERY_SERVICE ${GRPC_SERVICE_HOST}:${GRPC_SERVICE_PORT}
ENV GRPC_SERVICE_HOST ${GRPC_SERVICE_HOST}
ENV GRPC_SERVICE_PORT ${GRPC_SERVICE_PORT}
