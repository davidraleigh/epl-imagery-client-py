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
            sql_filters=None):
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
    __metadata = None

    def __init__(self, metadata):
        if isinstance(metadata, list):
            self.__metadata = metadata
        else:
            self.__metadata = [metadata]

    def fetch_imagery_array(self,
                            band_definitions,
                            scale_params=None,
                            cutline_wkb: bytes = None,
                            extent: tuple = None,
                            extent_cs: pyproj.Proj = None,
                            output_type: epl_imagery_api_pb2.DataType = None,
                            xRes=60,
                            yRes=60) -> np.ndarray:
        MB = 1024 * 1024
        GRPC_CHANNEL_OPTIONS = [('grpc.max_message_length', 64 * MB), ('grpc.max_receive_message_length', 64 * MB)]
        channel = grpc.insecure_channel('localhost:50051', options=GRPC_CHANNEL_OPTIONS)
        stub = epl_imagery_api_pb2_grpc.ImageryOperatorsStub(channel)
        request = epl_imagery_api_pb2.ImageryRequest(xRes=xRes,
                                                     yRes=yRes)
        grpc_band_definitions = []
        for index, band_def in enumerate(band_definitions):
            if isinstance(band_def, IntEnum):
                grpc_band_def = epl_imagery_api_pb2.BandDefinition(band_type=band_def)
            elif isinstance(band_def, int):
                grpc_band_def = epl_imagery_api_pb2.BandDefinition(band_number=band_def)
            elif isinstance(band_def, FunctionDetails):
                print("this is becoming problematic")
            if scale_params and len(scale_params) > index:
                grpc_band_def.scale_params.extend(scale_params[index])
            grpc_band_definitions.append(grpc_band_def)

        request.band_definitions.extend(grpc_band_definitions)
        request.metadata.extend(self.__metadata)
        if cutline_wkb:
            request.cutline_wkb.extend(cutline_wkb)
        result = stub.ImagerySearchNArray(request)
        nd_array = np.ndarray(buffer=np.array(result.data_uint32), shape=result.shape, dtype=np.uint8, order='F')
        return nd_array

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


class Band(IntEnum):
    # Crazy Values so that the Band.<ENUM>.value isn't used for anything
    UNKNOWN_BAND = 0
    ULTRA_BLUE = 1001
    BLUE = 1002
    GREEN = 1003
    RED = 1004
    NIR = 1005
    SWIR1 = 1006
    THERMAL = 1007
    SWIR2 = 1008
    PANCHROMATIC = 1009
    CIRRUS = 1010
    TIRS1 = 1011
    TIRS2 = 1012
    INFRARED2 = 1013
    INFRARED1 = 1014
    ALPHA = 1015


class BandMap:
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