import os
import grpc
import tempfile
import py_compile
import re

import numpy as np

from datetime import datetime
from datetime import date
from typing import List, Tuple
from enum import Enum, IntEnum


import epl.grpc.imagery.epl_imagery_pb2 as epl_imagery_pb2
import epl.grpc.imagery.epl_imagery_pb2_grpc as epl_imagery_pb2_grpc
from google.protobuf import timestamp_pb2



# EPL_IMAGERY_API_KEY = os.environ['EPL_IMAGERY_API_KEY']
# EPL_IMAGERY_API_SECRET = os.environ['EPL_IMAGERY_API_SECRET']
MB = 1024 * 1024
GRPC_CHANNEL_OPTIONS = [('grpc.max_message_length', 64 * MB), ('grpc.max_receive_message_length', 64 * MB)]
GRPC_SERVICE_PORT = os.getenv('GRPC_SERVICE_PORT', 50051)
GRPC_SERVICE_HOST = os.getenv('GRPC_SERVICE_HOST', 'localhost')
IMAGERY_SERVICE = "{0}:{1}".format(GRPC_SERVICE_HOST, GRPC_SERVICE_PORT)
print(IMAGERY_SERVICE)
ip_reg = re.compile(r"[\d]{1,3}\.[\d]{1,3}\.[\d]{1,3}\.[\d]{1,3}")


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


class DataType(Enum):
    # Byte, UInt16, Int16, UInt32, Int32, Float32, Float64, CInt16, CInt32, CFloat32 or CFloat64
    """enum DataType {
    BYTE = 0;
    INT16 = 1;
    UINT16 = 2;
    INT32 = 3;
    UINT32 = 4;
    FLOAT32 = 5;
    FLOAT64 = 6;
    CFLOAT32 = 7;
    CFLOAT64 = 8;
}
    """

    UNKNOWN_GDAL = ("Unknown",  1,           -1,         0, np.void)

    BYTE         = ("Byte",     0,           255,        1, np.uint8)

    UINT16       = ("UInt16",   0,           65535,      2, np.uint16)
    INT16        = ("Int16",    -32768,      32767,      3, np.int16)

    UINT32       = ("UInt32",   0,           4294967295, 4, np.uint32)
    INT32        = ("Int32",    -2147483648, 2147483647, 5, np.int32)

    FLOAT32      = ("Float32",  -3.4E+38,    3.4E+38,    6, np.float)
    FLOAT64      = ("Float64",  -1.7E+308,   1.7E+308,   7, np.float64)

    # CINT16
    # CINT32
    #TODO I think these ranges are reversed CFloat32 and CFloat64
    CFLOAT32     = ("CFloat32", -1.7E+308,   1.7E+308,   10, np.complex64)
    CFLOAT64     = ("CFloat64", -3.4E+38,    3.4E+38,    11, np.complex64)

    def __init__(self, name, range_min, range_max, grpc_num, numpy_type):
        self.__name = name
        self.range_min = range_min
        self.range_max = range_max
        self.__grpc_num = grpc_num
        self.__numpy_type = numpy_type

    def __or__(self, other):
        return self.__grpc_num | other.__grpc_num

    def __and__(self, other):
        return self.__grpc_num & other.__grpc_num

    @property
    def name(self):
        return self.__name

    @property
    def numpy_type(self):
        return self.__numpy_type


class FunctionDetails:
    """
    Make a pixel function
    """
    name = None
    band_definitions = None
    data_type = None
    code = None
    arguments = None
    transfer_type = None

    def __init__(self,
                 name: str,
                 band_definitions: list,
                 data_type: DataType,
                 code: str=None,
                 arguments: dict=None,
                 transfer_type: DataType=None):
        self.name = name
        self.band_definitions = band_definitions
        self.data_type = data_type

        if code:
            # TODO, still ugly that I have to use a temporary file: Also, stupid that I can't catch GDAL errors
            function_file = tempfile.NamedTemporaryFile(prefix=self.name, suffix=".py", delete=True)
            function_file.write(code.encode())
            function_file.flush()

            py_compile.compile(function_file.name, doraise=True)
            # delete file after compiling
            function_file.close()
            self.code = code

        # TODO arguments should maybe have some kind of setter
        if arguments:
            self.arguments = {k: str(v) for k, v in arguments.items()}
        self.transfer_type = transfer_type


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
        # not necessary at this time to support decimal seconds for epl_grpc
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
            cloud_cover: Tuple[float]=None,
            sql_filters=None):

        if GRPC_SERVICE_HOST == "localhost" or ip_reg.match(GRPC_SERVICE_HOST):
            channel = grpc.insecure_channel(IMAGERY_SERVICE, options=GRPC_CHANNEL_OPTIONS)
        else:
            channel_credentials = grpc.ssl_channel_credentials()
            channel = grpc.secure_channel(IMAGERY_SERVICE, credentials=channel_credentials, options=GRPC_CHANNEL_OPTIONS)
        #
        stub = epl_imagery_pb2_grpc.ImageryOperatorsStub(channel)

        request = epl_imagery_pb2.MetadataRequest(satellite_id=satellite_id,
                                                  bounding_box=bounding_box,
                                                  sort_by=sort_by,
                                                  limit=limit,
                                                  sql_filters=sql_filters)

        if cloud_cover:
            if not isinstance(cloud_cover, list):
                cloud_cover = [cloud_cover]
            request.cloud_cover.extend(cloud_cover)

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

    @staticmethod
    def __make_band_definition(band_def):
        if isinstance(band_def, IntEnum):
            grpc_band_def = epl_imagery_pb2.BandDefinition(band_type=band_def)
        elif isinstance(band_def, int):
            grpc_band_def = epl_imagery_pb2.BandDefinition(band_number=band_def)
        elif isinstance(band_def, FunctionDetails):
            grpc_function_details = epl_imagery_pb2.BandFunctionDetails(name=band_def.name,
                                                                        data_type=epl_imagery_pb2.GDALDataType.Value(band_def.data_type.name.upper()),
                                                                        code=band_def.code)

            # https://developers.google.com/protocol-buffers/docs/reference/python-generated#map-fields
            if band_def.arguments:
                for key in band_def.arguments:
                    grpc_function_details.arguments[key] = band_def.arguments[key]

            if band_def.transfer_type:
                grpc_function_details.transfer_type = epl_imagery_pb2.GDALDataType.Value(band_def.transfer_type.name.upper())

            interior_band_list = []
            for interior_band_def in band_def.band_definitions:
                interior_band_list.append(Landsat.__make_band_definition(interior_band_def))

            grpc_function_details.band_definitions.extend(interior_band_list)

            grpc_band_def = epl_imagery_pb2.BandDefinition(band_function=grpc_function_details)

        return grpc_band_def

    def make_imagery_request(self,
                             band_definitions,
                             scale_params: List[List[float]] = None,
                             polygon_boundary_wkb: bytes = None,
                             envelope_boundary: tuple = None,
                             boundary_cs=4326,
                             output_type: DataType = DataType.BYTE,
                             pixel_dimensions: tuple = None,
                             spatial_resolution_m=60) -> epl_imagery_pb2.ImageryRequest:
        request = epl_imagery_pb2.ImageryRequest(spatial_resolution_m=spatial_resolution_m)
        grpc_band_definitions = []
        for index, band_def in enumerate(band_definitions):
            grpc_band_def = Landsat.__make_band_definition(band_def)
            if scale_params and len(scale_params) > index:
                grpc_band_def.scale_params.extend(scale_params[index])
            grpc_band_definitions.append(grpc_band_def)

        request.band_definitions.extend(grpc_band_definitions)
        request.metadata.extend(self.__metadata)
        if polygon_boundary_wkb:
            request.polygon_boundary_wkb = polygon_boundary_wkb

        if envelope_boundary:
            request.envelope_boundary.extend(envelope_boundary)

        if not envelope_boundary or not polygon_boundary_wkb:
            spatial_reference = epl_imagery_pb2.ServiceSpatialReference()
            if isinstance(boundary_cs, int):
                spatial_reference.wkid = boundary_cs
            elif "[" in boundary_cs:
                spatial_reference.esri_wkt = boundary_cs
            else:
                spatial_reference.proj4 = boundary_cs

        request.output_type = epl_imagery_pb2.GDALDataType.Value(output_type.name.upper())

        return request

    def fetch_file(self,
                   band_definitions,
                   scale_params: List[List[float]] = None,
                   polygon_boundary_wkb: bytes = None,
                   envelope_boundary: tuple = None,
                   boundary_cs=4326,
                   output_type: DataType = DataType.BYTE,
                   pixel_dimensions: tuple = None,
                   spatial_resolution_m=60,
                   filename=None):

        if GRPC_SERVICE_HOST == "localhost" or ip_reg.match(GRPC_SERVICE_HOST):
            channel = grpc.insecure_channel(IMAGERY_SERVICE, options=GRPC_CHANNEL_OPTIONS)
        else:
            channel_credentials = grpc.ssl_channel_credentials()
            channel = grpc.secure_channel(IMAGERY_SERVICE, credentials=channel_credentials, options=GRPC_CHANNEL_OPTIONS)
        stub = epl_imagery_pb2_grpc.ImageryOperatorsStub(channel)
        imagery_request = self.make_imagery_request(band_definitions,
                                                    scale_params,
                                                    polygon_boundary_wkb,
                                                    envelope_boundary,
                                                    boundary_cs,
                                                    output_type,
                                                    pixel_dimensions,
                                                    spatial_resolution_m)

        imagery_file_request = epl_imagery_pb2.ImageryFileRequest(file_type=epl_imagery_pb2.JPEG)
        imagery_file_request.imagery_request.CopyFrom(imagery_request)

        big_file_result = stub.ImageryCompleteFile(imagery_file_request)

        temp = tempfile.NamedTemporaryFile(suffix="jpg")

        if filename:
            with open(filename, "wb") as f:
                f.write(big_file_result.data)

        return big_file_result.data

    def fetch_imagery_array(self,
                            band_definitions,
                            scale_params: List[List[float]]=None,
                            polygon_boundary_wkb: bytes=None,
                            envelope_boundary: tuple=None,
                            boundary_cs_wkid=4326,
                            output_type: DataType=DataType.BYTE,
                            pixel_dimensions: tuple=None,
                            spatial_resolution_m=60) -> np.ndarray:

        # https://github.com/grpc/grpc/issues/7927
        # https://github.com/grpc/grpc/issues/11014
        # https://stackoverflow.com/questions/42629047/how-to-increase-message-size-in-grpc-using-python
        # http://nanxiao.me/en/message-length-setting-in-grpc/
        # https://stackoverflow.com/questions/11784329/python-memory-usage-of-numpy-arrays
        # https://stackoverflow.com/questions/21312231/efficiently-converting-java-list-to-matlab-matrix
        # https://stackoverflow.com/questions/31280024/how-to-get-google-protobuf-working-in-matlab
        # https://stackoverflow.com/questions/10440590/using-protocol-buffer-java-bindings-in-matlab
        # https://www.mathworks.com/matlabcentral/answers/27524-compiled-matlab-executables-not-working-correctly-with-java-archives
        # https://stackoverflow.com/a/44576289/445372
        # https://stackoverflow.com/questions/34969446/grpc-image-upload
        # https://jbrandhorst.com/post/grpc-binary-blob-stream/
        # https://ops.tips/blog/sending-files-via-grpc/
        # https://stackoverflow.com/questions/8659471/multi-theaded-numpy-inserts
        # https://stackoverflow.com/questions/40690248/copy-numpy-array-into-part-of-another-array

        if GRPC_SERVICE_HOST == "localhost" or ip_reg.match(GRPC_SERVICE_HOST):
            channel = grpc.insecure_channel(IMAGERY_SERVICE, options=GRPC_CHANNEL_OPTIONS)
        else:
            channel_credentials = grpc.ssl_channel_credentials()
            channel = grpc.secure_channel(IMAGERY_SERVICE, credentials=channel_credentials, options=GRPC_CHANNEL_OPTIONS)

        stub = epl_imagery_pb2_grpc.ImageryOperatorsStub(channel)

        imagery_request = self.make_imagery_request(band_definitions,
                                                    scale_params,
                                                    polygon_boundary_wkb,
                                                    envelope_boundary,
                                                    boundary_cs_wkid,
                                                    output_type,
                                                    pixel_dimensions,
                                                    spatial_resolution_m)
        # TODO chunk array and stream results instead of block for one large result
        # https://stackoverflow.com/questions/40690248/copy-numpy-array-into-part-of-another-array
        result = stub.ImagerySearchNArray(imagery_request)

        # TODO eventually prevent copy by implementing PyArray_SimpleNewFromData
        # https://docs.scipy.org/doc/numpy/reference/c-api.array.html#c.PyArray_SimpleNewFromData
        # https://stackoverflow.com/questions/7543675/how-to-convert-pointer-to-c-array-to-python-array
        # https://stackoverflow.com/questions/33478046/binding-c-array-to-numpy-array-without-copying
        nd_array = None
        if output_type == DataType.BYTE or output_type == DataType.UINT16 or output_type == DataType.UINT32:
            nd_array = np.ndarray(buffer=np.array(result.data_uint32), shape=result.shape, dtype=np.uint64, order='C')
        elif output_type == DataType.INT16 or output_type == DataType.INT32:
            nd_array = np.ndarray(buffer=np.array(result.data_int32), shape=result.shape, dtype=np.int64, order='C')
        elif output_type == DataType.FLOAT32:
            nd_array = np.ndarray(buffer=np.array(result.data_float), shape=result.shape, dtype=np.float, order='C')
        elif output_type == DataType.FLOAT64:
            nd_array = np.ndarray(buffer=np.array(result.data_double), shape=result.shape, dtype=np.double, order='C')

        return nd_array


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
