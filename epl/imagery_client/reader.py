import pyproj

import os
import grpc

import numpy as np


from datetime import datetime
from datetime import date
from typing import List
from enum import Enum, IntEnum


import epl.service.imagery.epl_imagery_api_pb2 as epl_imagery_api_pb2
import epl.service.imagery.epl_imagery_api_pb2_grpc as epl_imagery_api_pb2_grpc
from google.protobuf import timestamp_pb2

# EPL_IMAGERY_API_KEY = os.environ['EPL_IMAGERY_API_KEY']
# EPL_IMAGERY_API_SECRET = os.environ['EPL_IMAGERY_API_SECRET']


class SpacecraftID(IntEnum):
    UNKNOWN_SPACECRAFT = 0
    LANDSAT_1_MSS = 1
    LANDSAT_2_MSS = 2
    LANDSAT_3_MSS = 4
    LANDSAT_123_MSS = 7
    LANDSAT_4_MSS = 8
    LANDSAT_5_MSS = 16
    LANDSAT_45_MSS = 24
    LANDSAT_4 = 32
    LANDSAT_5 = 64
    LANDSAT_45 = 96
    LANDSAT_7 = 128
    LANDSAT_8 = 256
    ALL = 512


class _DateType(Enum):
    START_DATE = 0
    END_DATE = 1


class Metadata:
    temp = None


class MetadataService:
    @staticmethod
    def __prep_date(date_input, date_type: _DateType) -> timestamp_pb2.Timestamp:
        if not date_input:
            return None

        if type(date_input) is date:
            # TODO, remove this logic. Too much secret help to the user who shouldn't be using api this way.
            # only in place to pass first set of tests.
            if date_type == _DateType.START_DATE:
                date_input = datetime.combine(date_input, datetime.min.time())
            elif date_type == _DateType.END_DATE:
                date_input = datetime.combine(date_input, datetime.max.time())

        if type(date_input) is not datetime:
            return None

        timestamp_message = timestamp_pb2.Timestamp()
        # not necessary at this time to support decimal seconds for imagery
        timestamp_message.FromJsonString(date_input.strftime("%Y-%m-%dT%H:%M:%SZ"))
        return timestamp_message


    def search_aws(mount_base_path,
                   wrs_path,
                   wrs_row,
                   collection_date: datetime,
                   processing_level: str = "L1TP"):
        return None

    def search(
            self,
            satellite_id: SpacecraftID=None,
            bounding_box=None,
            start_date: datetime=None,
            end_date: datetime=None,
            sort_by=None,
            limit=10,
            sql_filters=None) -> List[Metadata]:
        channel = grpc.insecure_channel('localhost:50051')
        stub = epl_imagery_api_pb2_grpc.ImageryOperatorsStub(channel)

        request = epl_imagery_api_pb2.MetadataRequest(satellite_id=satellite_id,
                                                      bounding_box=bounding_box,
                                                      sort_by=sort_by,
                                                      limit=limit,
                                                      sql_filters=sql_filters)

        if start_date:
            request.start_date.CopyFrom(MetadataService.__prep_date(start_date, _DateType.START_DATE))
        if end_date:
            request.end_date.CopyFrom(MetadataService.__prep_date(end_date, _DateType.END_DATE))

        result = stub.MetadataSearch(request)

        return result


class Landsat:
    def fetch_imagery_array(self,
                            band_definitions,
                            scale_params=None,
                            cutline_wkb: bytes = None,
                            extent: tuple = None,
                            extent_cs: pyproj.Proj = None,
                            output_type: epl_imagery_api_pb2.DataType = None,
                            xRes=60,
                            yRes=60) -> np.ndarray:
        return None

    def get_dataset(self,
                    band_definitions,
                    output_type: epl_imagery_api_pb2.DataType,
                    scale_params=None,
                    extent: tuple = None,
                    cutline_wkb: bytes = None,
                    xRes=60,
                    yRes=60):
        return None


class Storage:
    temp = None





class BandMap:
    temp = None

class Band:
    temp = None

class WRSGeometries:
    temp = None

class RasterBandMetadata:
    temp = None

class RasterMetadata:
    temp = None

class DataType:
    temp = None

class FunctionDetails:
    temp = None