import unittest
import datetime
import requests
import shapely.geometry

import numpy as np
from shapely.wkt import loads
from datetime import datetime
from datetime import date
from epl.client.imagery.reader import MetadataService, Landsat, SpacecraftID, Band, DataType, FunctionDetails
from epl.native.imagery.metadata_helpers import LandsatQueryFilters, SpacecraftID, LandsatModel

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
        # sql_filters = ['scene_id="LC80270312016188LGN00"']
        landsat_filters = LandsatQueryFilters()
        landsat_filters.scene_id.set_value("LC80270312016188LGN00")
        metadata_service = MetadataService()
        rows = metadata_service.search(SpacecraftID.LANDSAT_8, data_filters=landsat_filters)
        rows = list(rows)
        self.assertEqual(len(rows), 1)

    def test_scene_id(self):
        landsat_filters = LandsatQueryFilters()
        landsat_filters.scene_id.set_value("LC80270312016188LGN00")
        metadata_service = MetadataService()
        rows = metadata_service.search(SpacecraftID.LANDSAT_8, data_filters=landsat_filters)
        rows = list(rows)
        self.assertEqual(len(rows), 1)

    def test_start_date(self):
        # gs://gcp-public-data-landsat/LC08/PRE/044/034/LC80440342016259LGN00/
        metadata_service = MetadataService()
        d = date(2016, 6, 24)
        start_date = datetime.combine(d, datetime.min.time())
        landsat_filters = LandsatQueryFilters()
        landsat_filters.acquired.set_range(start=start_date)
        rows = metadata_service.search(SpacecraftID.LANDSAT_8, data_filters=landsat_filters)
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
        landsat_filters = LandsatQueryFilters()
        landsat_filters.acquired.set_range(end=d)
        rows = metadata_service.search(SpacecraftID.LANDSAT_7, data_filters=landsat_filters)
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
        landsat_filters = LandsatQueryFilters()
        landsat_filters.acquired.set_value(d)
        rows = metadata_service.search(SpacecraftID.LANDSAT_8, data_filters=landsat_filters)
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
        landsat_filters = LandsatQueryFilters()
        landsat_filters.acquired.set_range(start=d_start, end=d_end)
        rows = metadata_service.search(SpacecraftID.LANDSAT_8, data_filters=landsat_filters)
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
        landsat_filters = LandsatQueryFilters()
        landsat_filters.bounds.set_bounds(*bounding_box)
        landsat_filters.acquired.set_range(start=d_start, end=d_end)
        metadata_rows = metadata_service.search(SpacecraftID.LANDSAT_8,
                                                data_filters=landsat_filters)

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

    def test_bounds_repeat_bug(self):
        # grab the Taos, NM county outline from a geojson hosted on github
        r = requests.get("https://raw.githubusercontent.com/johan/world.geo.json/master/countries/USA/NM/Taos.geo.json")
        taos_geom = r.json()
        taos_shape = shapely.geometry.shape(taos_geom['features'][0]['geometry'])
        metadata_service = MetadataService()

        d_start = date(2017, 3, 12)  # 2017-03-12
        d_end = date(2017, 3, 19)  # epl api is inclusive

        # PRE is a collection type that specifies certain QA standards
        # sql_filters = ['collection_number="PRE"']
        landsat_filters = LandsatQueryFilters()
        landsat_filters.collection_number.set_value("PRE")
        landsat_filters.acquired.set_range(start=d_start, end=d_end)
        landsat_filters.bounds.set_bounds(*taos_shape.bounds)
        # search the satellite metadata for images of Taos withing the given date range
        rows = metadata_service.search(
            SpacecraftID.LANDSAT_8,
            limit=10,
            data_filters=landsat_filters)

        # group the scenes together in a list
        for row in rows:
            self.assertEqual(4, len(row.bounds))


class TestLandsat(unittest.TestCase):
    base_mount_path = '/epl_grpc'
    metadata_service = None
    metadata_set = []
    r = requests.get("https://raw.githubusercontent.com/johan/world.geo.json/master/countries/USA/NM/Taos.geo.json")
    taos_geom = r.json()
    taos_shape = shapely.geometry.shape(taos_geom['features'][0]['geometry'])

    def setUp(self):
        d_start = date(2017, 3, 12)  # 2017-03-12
        d_end = date(2017, 3, 19)  # 2017-03-20, epl api is inclusive

        self.metadata_service = MetadataService()

        landsat_filters = LandsatQueryFilters()
        landsat_filters.collection_number.set_value("PRE")
        landsat_filters.acquired.set_range(start=d_start, end=d_end)
        landsat_filters.bounds.set_bounds(*self.taos_shape.bounds)

        metadata_rows = self.metadata_service.search(
            SpacecraftID.LANDSAT_8,
            limit=10,
            data_filters=landsat_filters)

        for row in metadata_rows:
            self.metadata_set.append(row)


    """
            data, metadata = self.raster.ndarray(
            inputs=['meta_LC80270312016188_v1'],
            bands=['red', 'green', 'blue', 'alpha'],
            resolution=960,
        )"""
    def test_ndarray(self):
        # sql_filters = ['scene_id="LC80270312016188LGN00"']
        landsat_filters = LandsatQueryFilters()
        landsat_filters.scene_id.set_value("LC80270312016188LGN00")
        metadata_rows = self.metadata_service.search(
            SpacecraftID.UNKNOWN_SPACECRAFT,
            data_filters=landsat_filters)

        metadata_set = list(metadata_rows)
        landsat = Landsat(metadata_set)
        data = landsat.fetch_imagery_array(
            band_definitions=[Band.RED, Band.GREEN, Band.BLUE, Band.ALPHA],
            spatial_resolution_m=960)

        self.assertEqual(data.shape, (249, 245, 4))
        self.assertEqual(data.dtype, np.uint8)

    def test_band_enum(self):
        self.assertTrue(True)
        d_start = date(2016, 7, 20)
        d_end = date(2016, 7, 28)
        landsat_filters = LandsatQueryFilters()
        landsat_filters.scene_id.set_value("LC80390332016208LGN00")
        landsat_filters.acquired.set_range(start=d_start, end=d_end)
        rows = self.metadata_service.search(SpacecraftID.LANDSAT_8,
                                            limit=1,
                                            data_filters=landsat_filters)
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

        # get a numpy.ndarray from bands for specified epl_grpc
        band_numbers = [Band.RED, Band.GREEN, Band.BLUE]
        scale_params = [[0.0, 65535], [0.0, 65535], [0.0, 65535]]
        nda = landsat.fetch_imagery_array(band_numbers, scale_params, self.taos_shape.wkb, spatial_resolution_m=480)
        self.assertIsNotNone(nda)

        # TODO needs shape test

    def test_mosaic(self):
        # GDAL helper functions for generating VRT
        landsat = Landsat(self.metadata_set)

        # get a numpy.ndarray from bands for specified epl_grpc
        band_numbers = [Band.RED, Band.GREEN, Band.BLUE]
        scale_params = [[0.0, 65535], [0.0, 65535], [0.0, 65535]]
        nda = landsat.fetch_imagery_array(band_numbers, scale_params, envelope_boundary=self.taos_shape.bounds)
        self.assertIsNotNone(nda)
        self.assertEqual((1804, 1295, 3), nda.shape)

        # TODO needs shape test

    def test_mosaic_cutline(self):
        # GDAL helper functions for generating VRT
        landsat = Landsat(self.metadata_set)

        # get a numpy.ndarray from bands for specified epl_grpc
        # 'nir', 'swir1', 'swir2'
        band_numbers = [Band.NIR, Band.SWIR1, Band.SWIR2]
        scaleParams = [[0.0, 40000.0], [0.0, 40000.0], [0.0, 40000.0]]
        nda = landsat.fetch_imagery_array(band_numbers, scaleParams, polygon_boundary_wkb=self.taos_shape.wkb)
        self.assertIsNotNone(nda)
        self.assertEqual((1804, 1295, 3), nda.shape)

    def test_mosaic_mem_error(self):
        landsat = Landsat(self.metadata_set)

        # get a numpy.ndarray from bands for specified epl_grpc
        band_numbers = [Band.RED, Band.GREEN, Band.BLUE]
        scaleParams = [[0.0, 40000], [0.0, 40000], [0.0, 40000]]
        nda = landsat.fetch_imagery_array(band_numbers, scaleParams, envelope_boundary=self.taos_shape.bounds)

        self.assertIsNotNone(nda)
        # GDAL helper functions for generating VRT
        landsat = Landsat(self.metadata_set)
        self.assertEqual((1804, 1295, 3), nda.shape)

        # get a numpy.ndarray from bands for specified epl_grpc
        # 'nir', 'swir1', 'swir2'
        band_numbers = [Band.NIR, Band.SWIR1, Band.SWIR2]
        scaleParams = [[0.0, 40000.0], [0.0, 40000.0], [0.0, 40000.0]]
        nda = landsat.fetch_imagery_array(band_numbers, scaleParams, polygon_boundary_wkb=self.taos_shape.wkb)
        self.assertIsNotNone(nda)
        self.assertEqual((1804, 1295, 3), nda.shape)

    def test_datatypes(self):
        landsat = Landsat(self.metadata_set)

        # get a numpy.ndarray from bands for specified epl_grpc
        band_numbers = [Band.RED, Band.GREEN, Band.BLUE]
        scaleParams = [[0.0, 40000], [0.0, 40000], [0.0, 40000]]

        for data_type in DataType:
            if data_type == DataType.CFLOAT32 or data_type == DataType.CFLOAT64 or data_type == DataType.UNKNOWN_GDAL:
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

        # get a numpy.ndarray from bands for specified epl_grpc
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

        # get a numpy.ndarray from bands for specified epl_grpc
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

        # get a numpy.ndarray from bands for specified epl_grpc
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
        landsat_filters = LandsatQueryFilters()
        landsat_filters.scene_id.set_value("LC80400312016103LGN00")
        landsat_filters.acquired.set_range(start=d_start, end=d_end)
        landsat_filters.bounds.set_bounds(*bounding_box)
        # sql_filters = ['scene_id="LC80400312016103LGN00"']
        rows = metadata_service.search(SpacecraftID.LANDSAT_8,
                                       limit=1,
                                       data_filters=landsat_filters)
        rows = list(rows)
        self.m_row_data = rows[0]
        wkt_iowa = "POLYGON((-93.76075744628906 42.32707774458643,-93.47854614257812 42.32707774458643," \
                   "-93.47854614257812 42.12674735753131,-93.76075744628906 42.12674735753131," \
                   "-93.76075744628906 42.32707774458643))"
        self.iowa_polygon = loads(wkt_iowa)

        d_start = date(2017, 3, 12)  # 2017-03-12
        d_end = date(2017, 3, 19)  # 2017-03-20, epl api is inclusive

        landsat_filters = LandsatQueryFilters()
        landsat_filters.collection_number.set_value("PRE")
        landsat_filters.acquired.set_range(start=d_start, end=d_end)
        landsat_filters.bounds.set_bounds(*self.taos_shape.bounds)
        # sql_filters = ['collection_number="PRE"']
        rows = self.metadata_service.search(
            SpacecraftID.LANDSAT_8,
            limit=10,
            data_filters=landsat_filters)

        for row in rows:
            self.metadata_set.append(row)

    def test_ndvi_taos(self):
        code = """import numpy as np
def ndvi_numpy(in_ar, out_ar, xoff, yoff, xsize, ysize, raster_xsize, raster_ysize, buf_radius, gt, **kwargs):
    with np.errstate(divide = 'ignore', invalid = 'ignore'):
        factor = float(kwargs['factor'])
        output = np.divide((in_ar[1] - in_ar[0]), (in_ar[1] + in_ar[0]))
        output[np.isnan(output)] = 0.0
        # shift range from -1.0-1.0 to 0.0-2.0
        output += 1.0
        # scale up from 0.0-2.0 to 0 to 255 by multiplying by 255/2
        # https://stackoverflow.com/a/1735122/445372
        output *=  factor/2.0
        # https://stackoverflow.com/a/10622758/445372
        # in place type conversion
        out_ar[:] = output.astype(np.int16, copy=False)"""

        code2 = """import numpy as np
def ndvi_numpy2(in_ar, out_ar, xoff, yoff, xsize, ysize, raster_xsize, raster_ysize, buf_radius, gt, **kwargs):
    with np.errstate(divide = 'ignore', invalid = 'ignore'):
        output = (in_ar[1] - in_ar[0]) / (in_ar[1] + in_ar[0])
        output[np.isnan(output)] = 0.0
        out_ar[:] = output"""

        landsat = Landsat(self.metadata_set)
        scale_params = [[0, DataType.UINT16.range_max, -1.0, 1.0]]

        pixel_function_details = FunctionDetails(name="ndvi_numpy",
                                                 band_definitions=[Band.RED, Band.NIR],
                                                 code=code,
                                                 arguments={"factor": DataType.UINT16.range_max},
                                                 data_type=DataType.UINT16)

        nda = landsat.fetch_imagery_array([pixel_function_details],
                                          scale_params=scale_params,
                                          polygon_boundary_wkb=self.taos_shape.wkb,
                                          output_type=DataType.FLOAT32)

        self.assertIsNotNone(nda)
        self.assertGreaterEqual(1.0, nda.max())
        self.assertLessEqual(-1.0, nda.min())

        pixel_function_details = FunctionDetails(name="ndvi_numpy2",
                                                 band_definitions=[Band.RED, Band.NIR],
                                                 code=code2,
                                                 data_type=DataType.FLOAT32)

        nda2 = landsat.fetch_imagery_array([pixel_function_details],
                                           polygon_boundary_wkb=self.taos_shape.wkb,
                                           output_type=DataType.FLOAT32)

        self.assertIsNotNone(nda2)
        self.assertGreaterEqual(1.0, nda2.max())
        self.assertLessEqual(-1.0, nda2.min())


