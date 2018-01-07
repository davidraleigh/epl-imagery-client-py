# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
import grpc

import epl.service.imagery.epl_imagery_api_pb2 as epl__imagery__api__pb2


class ImageryOperatorsStub(object):
  """
  gRPC Interfaces for working with geometry operators
  """

  def __init__(self, channel):
    """Constructor.

    Args:
      channel: A grpc.Channel.
    """
    self.MetadataSearch = channel.unary_unary(
        '/geometry.ImageryOperators/MetadataSearch',
        request_serializer=epl__imagery__api__pb2.MetadataRequest.SerializeToString,
        response_deserializer=epl__imagery__api__pb2.MetadataResult.FromString,
        )


class ImageryOperatorsServicer(object):
  """
  gRPC Interfaces for working with geometry operators
  """

  def MetadataSearch(self, request, context):
    """Execute a single geometry operation
    """
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')


def add_ImageryOperatorsServicer_to_server(servicer, server):
  rpc_method_handlers = {
      'MetadataSearch': grpc.unary_unary_rpc_method_handler(
          servicer.MetadataSearch,
          request_deserializer=epl__imagery__api__pb2.MetadataRequest.FromString,
          response_serializer=epl__imagery__api__pb2.MetadataResult.SerializeToString,
      ),
  }
  generic_handler = grpc.method_handlers_generic_handler(
      'geometry.ImageryOperators', rpc_method_handlers)
  server.add_generic_rpc_handlers((generic_handler,))
