import unittest
import datetime
import requests
import shapely.geometry

import numpy as np
from osgeo import gdal
from shapely.wkt import loads
from lxml import etree
from datetime import datetime
from datetime import date
from epl.client.imagery.reader import MetadataService, Landsat, SpacecraftID, Band, DataType, FunctionDetails

from math import isclose


def text_compare(t1, t2, tolerance=None):
    if not t1 and not t2:
        return True
    if t1 == '*' or t2 == '*':
        return True
    if tolerance:
        try:
            t1_float = list(map(float, t1.split(",")))
            t2_float = list(map(float, t2.split(",")))
            if len(t1_float) != len(t2_float):
                return False

            for idx, val_1 in enumerate(t1_float):
                if not isclose(val_1, t2_float[idx], rel_tol=tolerance):
                    return False

            return True

        except:
            return False
    return (t1 or '').strip() == (t2 or '').strip()


# https://bitbucket.org/ianb/formencode/src/tip/formencode/doctest_xml_compare.py?fileviewer=file-view-default#cl-70
def xml_compare(x1, x2, tag_tolerances={}):
    tolerance = tag_tolerances[x1.tag] if x1.tag in tag_tolerances else None
    if x1.tag != x2.tag:
        return False, '\nTags do not match: %s and %s' % (x1.tag, x2.tag)
    for name, value in x1.attrib.items():
        tolerance = tag_tolerances[name] if name in tag_tolerances else None
        if not text_compare(x2.attrib.get(name), value, tolerance):
        # if x2.attrib.get(name) != value:
            return False, '\nAttributes do not match: %s=%r, %s=%r' % (name, value, name, x2.attrib.get(name))
    for name in x2.attrib.keys():
        if name not in x1.attrib:
            return False, '\nx2 has an attribute x1 is missing: %s' % name
    if not text_compare(x1.text, x2.text, tolerance):
        return False, '\ntext: %r != %r, for tag %s' % (x1.text, x2.text, x1.tag)
    if not text_compare(x1.tail, x2.tail):
        return False, '\ntail: %r != %r' % (x1.tail, x2.tail)
    cl1 = sorted(x1.getchildren(), key=lambda x: x.tag)
    cl2 = sorted(x2.getchildren(), key=lambda x: x.tag)
    if len(cl1) != len(cl2):
        expected_tags = "\n".join(map(lambda x: x.tag, cl1)) + '\n'
        actual_tags = "\n".join(map(lambda x: x.tag, cl2)) + '\n'
        return False, '\nchildren length differs, %{0} != {1}\nexpected tags:\n{2}\nactual tags:\n{3}'.format(len(cl1), len(cl2), expected_tags, actual_tags)
    i = 0
    for c1, c2 in zip(cl1, cl2):
        i += 1
        result, message = xml_compare(c1, c2, tag_tolerances)
        # if not xml_compare(c1, c2):
        if not result:
            return False, '\nthe children %i do not match: %s\n%s' % (i, c1.tag, message)
    return True, "no errors"

class TestMetaDataSQL(unittest.TestCase):
    def test_scene_id2(self):
        sql_filters = ['scene_id="LC80270312016188LGN00"']
        metadata_service = MetadataService()
        rows = metadata_service.search(SpacecraftID.LANDSAT_8, sql_filters=sql_filters)
        rows = list(rows)
        self.assertEqual(len(rows), 1)

    def test_scene_id(self):
        sql_filters = ['scene_id="LC80390332016208LGN00"']
        metadata_service = MetadataService()
        rows = metadata_service.search(SpacecraftID.LANDSAT_8, sql_filters=sql_filters)
        rows = list(rows)
        self.assertEqual(len(rows), 1)

    def test_start_date(self):
        # gs://gcp-public-data-landsat/LC08/PRE/044/034/LC80440342016259LGN00/
        metadata_service = MetadataService()
        d = date(2016, 6, 24)
        start_date = datetime.combine(d, datetime.min.time())
        rows = metadata_service.search(SpacecraftID.LANDSAT_8, start_date=start_date)
        rows = list(rows)
        self.assertEqual(len(rows), 10)
        for row in rows:
            self.assertEqual(row.spacecraft_id, SpacecraftID.LANDSAT_8)
            d_actual = datetime.strptime(row.date_acquired, '%Y-%m-%d').date()
            self.assertGreaterEqual(d_actual, d)

    def test_end_date(self):
        # gs://gcp-public-data-landsat/LC08/PRE/044/034/LC80440342016259LGN00/
        metadata_service = MetadataService()
        d = date(2016, 6, 24)
        rows = metadata_service.search(SpacecraftID.LANDSAT_7, end_date=d)
        rows = list(rows)
        self.assertEqual(len(rows), 10)
        for row in rows:
            self.assertEqual(row.spacecraft_id, SpacecraftID.LANDSAT_7)
            d_actual = datetime.strptime(row.date_acquired, '%Y-%m-%d').date()
            self.assertLessEqual(d_actual, d)

    def test_one_day(self):
        # gs://gcp-public-data-landsat/LC08/PRE/044/034/LC80440342016259LGN00/
        metadata_service = MetadataService()
        d = date(2016, 6, 24)
        rows = metadata_service.search(SpacecraftID.LANDSAT_8, start_date=d, end_date=d)
        rows = list(rows)
        self.assertEqual(len(rows), 10)
        for row in rows:
            self.assertEqual(row.spacecraft_id, SpacecraftID.LANDSAT_8)
            d_actual = datetime.strptime(row.date_acquired, '%Y-%m-%d').date()
            self.assertEqual(d_actual, d)

    def test_1_year(self):
        # gs://gcp-public-data-landsat/LC08/PRE/044/034/LC80440342016259LGN00/
        metadata_service = MetadataService()
        d_start = date(2015, 6, 24)
        d_end = date(2016, 6, 24)
        rows = metadata_service.search(SpacecraftID.LANDSAT_8, start_date=d_start, end_date=d_end)
        rows = list(rows)
        self.assertEqual(len(rows), 10)
        for row in rows:
            self.assertEqual(row.spacecraft_id, SpacecraftID.LANDSAT_8)
            d_actual = datetime.strptime(row.date_acquired, '%Y-%m-%d').date()
            self.assertLessEqual(d_actual, d_end)
            self.assertGreaterEqual(d_actual, d_start)

    def test_bounding_box_1(self):
        metadata_service = MetadataService()
        d_start = date(2015, 6, 24)
        d_end = date(2016, 6, 24)
        bounding_box = (-115.927734375, 34.52466147177172, -78.31054687499999, 44.84029065139799)
        metadata_rows = metadata_service.search(SpacecraftID.LANDSAT_8,
                                                start_date=d_start,
                                                end_date=d_end,
                                                bounding_box=bounding_box)

        metadata_rows = list(metadata_rows)

        self.assertEqual(len(metadata_rows), 10)
        for row in metadata_rows:
            self.assertEqual(row.spacecraft_id, SpacecraftID.LANDSAT_8)
            d_actual = datetime.strptime(row.date_acquired, '%Y-%m-%d').date()
            self.assertLessEqual(d_actual, d_end)
            self.assertGreaterEqual(d_actual, d_start)
            test_box = row.bounds
            self.assertTrue(
                (bounding_box[0] < test_box[2] < bounding_box[2]) or
                (bounding_box[0] < test_box[0] < bounding_box[2]))
            self.assertTrue(
                (bounding_box[1] < test_box[3] < bounding_box[3]) or
                (bounding_box[1] < test_box[1] < bounding_box[3]))


class TestLandsat(unittest.TestCase):
    base_mount_path = '/grpc'
    metadata_service = None
    metadata_set = []
    r = requests.get("https://raw.githubusercontent.com/johan/world.geo.json/master/countries/USA/NM/Taos.geo.json")
    taos_geom = r.json()
    taos_shape = shapely.geometry.shape(taos_geom['features'][0]['geometry'])

    def setUp(self):
        d_start = date(2017, 3, 12)  # 2017-03-12
        d_end = date(2017, 3, 19)  # 2017-03-20, epl api is inclusive

        self.metadata_service = MetadataService()

        sql_filters = ['collection_number="PRE"']
        metadata_rows = self.metadata_service.search(
            SpacecraftID.LANDSAT_8,
            start_date=d_start,
            end_date=d_end,
            bounding_box=self.taos_shape.bounds,
            limit=10,
            sql_filters=sql_filters)

        for row in metadata_rows:
            self.metadata_set.append(row)


    """
            data, metadata = self.raster.ndarray(
            inputs=['meta_LC80270312016188_v1'],
            bands=['red', 'green', 'blue', 'alpha'],
            resolution=960,
        )"""
    def test_ndarray(self):
        sql_filters = ['scene_id="LC80270312016188LGN00"']
        metadata_rows = self.metadata_service.search(
            SpacecraftID.UNKNOWN_SPACECRAFT,
            sql_filters=sql_filters)

        landsat = Landsat(list(metadata_rows))
        data = landsat.fetch_imagery_array(
            band_definitions=[Band.RED, Band.GREEN, Band.BLUE, Band.ALPHA],
            spatial_resolution_m=960)

        self.assertEqual(data.shape, (249, 245, 4))
        self.assertEqual(data.dtype, np.uint8)



    def test_band_enum(self):
        self.assertTrue(True)
        d_start = date(2016, 7, 20)
        d_end = date(2016, 7, 28)
        rows = self.metadata_service.search(SpacecraftID.LANDSAT_8,
                                            start_date=d_start,
                                            end_date=d_end,
                                            limit=1,
                                            sql_filters=['scene_id="LC80390332016208LGN00"'])
        rows = list(rows)
        metadata = rows[0]
        landsat = Landsat(metadata)
        scale_params = [[0.0, 65535], [0.0, 65535], [0.0, 65535]]
        # nda = landsat.__get_ndarray(band_numbers, metadata, scale_params)
        nda = landsat.fetch_imagery_array(
            [Band.RED, Band.GREEN, Band.BLUE],
            scale_params=scale_params,
            spatial_resolution_m=240)
        self.assertIsNotNone(nda)
        nda2 = landsat.fetch_imagery_array(
            [4, 3, 2],
            scale_params=scale_params,
            spatial_resolution_m=240)
        np.testing.assert_almost_equal(nda, nda2)
        # nda = nda.flatten()
        # nda2 = nda2.flatten()
        # for i in range(0, len(nda)):
        #     self.assertEquals(nda[i], nda2[i], "failure at index{0}")


#
    def test_cutline(self):
        # GDAL helper functions for generating VRT
        landsat = Landsat(self.metadata_set[0])

        # get a numpy.ndarray from bands for specified grpc
        band_numbers = [Band.RED, Band.GREEN, Band.BLUE]
        scale_params = [[0.0, 65535], [0.0, 65535], [0.0, 65535]]
        nda = landsat.fetch_imagery_array(band_numbers, scale_params, self.taos_shape.wkb, spatial_resolution_m=480)
        self.assertIsNotNone(nda)

        # TODO needs shape test

    def test_mosaic(self):
        # GDAL helper functions for generating VRT
        landsat = Landsat(self.metadata_set)

        # get a numpy.ndarray from bands for specified grpc
        band_numbers = [Band.RED, Band.GREEN, Band.BLUE]
        scale_params = [[0.0, 65535], [0.0, 65535], [0.0, 65535]]
        nda = landsat.fetch_imagery_array(band_numbers, scale_params, envelope_boundary=self.taos_shape.bounds)
        self.assertIsNotNone(nda)
        self.assertEqual((1804, 1295, 3), nda.shape)

        # TODO needs shape test

    def test_mosaic_cutline(self):
        # GDAL helper functions for generating VRT
        landsat = Landsat(self.metadata_set)

        # get a numpy.ndarray from bands for specified grpc
        # 'nir', 'swir1', 'swir2'
        band_numbers = [Band.NIR, Band.SWIR1, Band.SWIR2]
        scaleParams = [[0.0, 40000.0], [0.0, 40000.0], [0.0, 40000.0]]
        nda = landsat.fetch_imagery_array(band_numbers, scaleParams, polygon_boundary_wkb=self.taos_shape.wkb)
        self.assertIsNotNone(nda)
        self.assertEqual((1804, 1295, 3), nda.shape)

    def test_mosaic_mem_error(self):
        landsat = Landsat(self.metadata_set)

        # get a numpy.ndarray from bands for specified grpc
        band_numbers = [Band.RED, Band.GREEN, Band.BLUE]
        scaleParams = [[0.0, 40000], [0.0, 40000], [0.0, 40000]]
        nda = landsat.fetch_imagery_array(band_numbers, scaleParams, envelope_boundary=self.taos_shape.bounds)

        self.assertIsNotNone(nda)
        # GDAL helper functions for generating VRT
        landsat = Landsat(self.metadata_set)
        self.assertEqual((1804, 1295, 3), nda.shape)

        # get a numpy.ndarray from bands for specified grpc
        # 'nir', 'swir1', 'swir2'
        band_numbers = [Band.NIR, Band.SWIR1, Band.SWIR2]
        scaleParams = [[0.0, 40000.0], [0.0, 40000.0], [0.0, 40000.0]]
        nda = landsat.fetch_imagery_array(band_numbers, scaleParams, polygon_boundary_wkb=self.taos_shape.wkb)
        self.assertIsNotNone(nda)
        self.assertEqual((1804, 1295, 3), nda.shape)

    def test_datatypes(self):
        landsat = Landsat(self.metadata_set)

        # get a numpy.ndarray from bands for specified grpc
        band_numbers = [Band.RED, Band.GREEN, Band.BLUE]
        scaleParams = [[0.0, 40000], [0.0, 40000], [0.0, 40000]]

        for data_type in DataType:
            if data_type == DataType.CFLOAT32 or data_type == DataType.CFLOAT64:
                continue

            nda = landsat.fetch_imagery_array(band_numbers,
                                              scaleParams,
                                              envelope_boundary=self.taos_shape.bounds,
                                              output_type=data_type,
                                              spatial_resolution_m=240)
            self.assertIsNotNone(nda)
            self.assertGreaterEqual(data_type.range_max, nda.max())
            self.assertLessEqual(data_type.range_min, nda.min())

    def test_vrt_with_alpha(self):
        landsat = Landsat(self.metadata_set)

        # get a numpy.ndarray from bands for specified grpc
        band_numbers = [Band.RED, Band.GREEN, Band.BLUE, Band.ALPHA]
        scaleParams = [[0.0, 40000], [0.0, 40000], [0.0, 40000]]

        nda = landsat.fetch_imagery_array(band_numbers,
                                          scaleParams,
                                          envelope_boundary=self.taos_shape.bounds,
                                          output_type=DataType.UINT16,
                                          spatial_resolution_m=120)
        self.assertIsNotNone(nda)

    def test_rastermetadata_cache(self):
        # GDAL helper functions for generating VRT
        landsat = Landsat(self.metadata_set)

        # get a numpy.ndarray from bands for specified grpc
        # 'nir', 'swir1', 'swir2'
        band_numbers = [Band.NIR, Band.SWIR1, Band.SWIR2]
        scaleParams = [[0.0, 40000.0], [0.0, 40000.0], [0.0, 40000.0]]
        nda = landsat.fetch_imagery_array(band_numbers, scaleParams, polygon_boundary_wkb=self.taos_shape.wkb, spatial_resolution_m=120)
        self.assertIsNotNone(nda)
        self.assertEqual((902, 648, 3), nda.shape)

        band_numbers = [Band.RED, Band.BLUE, Band.GREEN]
        nda = landsat.fetch_imagery_array(band_numbers, scaleParams, polygon_boundary_wkb=self.taos_shape.wkb, spatial_resolution_m=120)
        self.assertIsNotNone(nda)
        self.assertEqual((902, 648, 3), nda.shape)

        band_numbers = [Band.RED, Band.BLUE, Band.GREEN]
        nda = landsat.fetch_imagery_array(band_numbers, scaleParams, spatial_resolution_m=120)
        self.assertIsNotNone(nda)
        self.assertNotEqual((902, 648, 3), nda.shape)

    def test_file_creation(self):
        landsat = Landsat(self.metadata_set[0])

        # get a numpy.ndarray from bands for specified grpc
        band_numbers = [Band.RED, Band.GREEN, Band.BLUE]
        scale_params = [[0.0, 65535], [0.0, 65535], [0.0, 65535]]
        file_name = landsat.fetch_file(band_numbers, scale_params, self.taos_shape.wkb, spatial_resolution_m=480)
        self.assertGreater(len(file_name), 0)


class TestAWSPixelFunctions(unittest.TestCase):
    m_row_data = None
    base_mount_path = '/imagery'
    metadata_service = MetadataService()
    iowa_polygon = None
    metadata_set = []
    r = requests.get("https://raw.githubusercontent.com/johan/world.geo.json/master/countries/USA/NM/Taos.geo.json")
    taos_geom = r.json()
    taos_shape = shapely.geometry.shape(taos_geom['features'][0]['geometry'])

    def setUp(self):
        metadata_service = MetadataService()
        d_start = date(2015, 6, 24)
        d_end = date(2016, 6, 24)
        bounding_box = (-115.927734375, 34.52466147177172, -78.31054687499999, 44.84029065139799)
        sql_filters = ['scene_id="LC80400312016103LGN00"']
        rows = metadata_service.search(SpacecraftID.LANDSAT_8,
                                       start_date=d_start,
                                       end_date=d_end,
                                       bounding_box=bounding_box,
                                       limit=1,
                                       sql_filters=sql_filters)
        rows = list(rows)
        self.m_row_data = rows[0]
        wkt_iowa = "POLYGON((-93.76075744628906 42.32707774458643,-93.47854614257812 42.32707774458643," \
                   "-93.47854614257812 42.12674735753131,-93.76075744628906 42.12674735753131," \
                   "-93.76075744628906 42.32707774458643))"
        self.iowa_polygon = loads(wkt_iowa)
        gdal.SetConfigOption('GDAL_VRT_ENABLE_PYTHON', "YES")

        d_start = date(2017, 3, 12)  # 2017-03-12
        d_end = date(2017, 3, 19)  # 2017-03-20, epl api is inclusive

        sql_filters = ['collection_number="PRE"']
        rows = self.metadata_service.search(
            SpacecraftID.LANDSAT_8,
            start_date=d_start,
            end_date=d_end,
            bounding_box=self.taos_shape.bounds,
            limit=10,
            sql_filters=sql_filters)

        for row in rows:
            self.metadata_set.append(row)

    def test_pixel_1(self):
        metadata = self.m_row_data
        landsat = Landsat(metadata)  # , gsurl[2])

        code = """import numpy as np
def multiply_rounded(in_ar, out_ar, xoff, yoff, xsize, ysize, raster_xsize,
                   raster_ysize, buf_radius, gt, **kwargs):
    factor = float(kwargs['factor'])
    out_ar[:] = np.round_(np.clip(in_ar[0] * factor,0,255))"""

        function_arguments = {"factor": "1.5"}
        pixel_function_details = FunctionDetails(name="multiply_rounded", band_definitions=[2],
                                                 data_type=DataType.FLOAT32, code=code,
                                                 arguments=function_arguments)

        vrt = landsat.get_vrt([pixel_function_details, 3, 2])

        with open('pixel_1_aws.vrt', 'r') as myfile:
            data = myfile.read()
            expected = etree.XML(data)
            actual = etree.XML(vrt)
            result, message = xml_compare(expected, actual, {"GeoTransform": 1e-10})
            self.assertTrue(result, message)


