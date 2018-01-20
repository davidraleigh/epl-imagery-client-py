import unittest
import datetime
import requests
import shapely.geometry

import numpy as np

from datetime import datetime
from datetime import date
from epl.client.imagery.reader import MetadataService, Landsat, SpacecraftID, Band, DataType


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

    def test_cloud_cover(self):
        metadata_service = MetadataService()
        sql_filters = ['cloud_cover=0']
        d_start = date(2015, 6, 24)
        d_end = date(2016, 6, 24)
        bounding_box = (-115.927734375, 34.52466147177172, -78.31054687499999, 44.84029065139799)
        rows = metadata_service.search(
            SpacecraftID.LANDSAT_8,
            start_date=d_start,
            end_date=d_end,
            bounding_box=bounding_box,
            sql_filters=sql_filters)

        rows = list(rows)

        self.assertEqual(len(rows), 10)
        for row in rows:
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

        # mounted directory in docker container
        base_mount_path = '/grpc'

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

    def test_pixel_function_vrt_1(self):
        utah_box = (-112.66342163085938, 37.738141282210385, -111.79824829101562, 38.44821130413263)
        d_start = date(2016, 7, 20)
        d_end = date(2016, 7, 28)

        rows = self.metadata_service.search(SpacecraftID.LANDSAT_8, start_date=d_start, end_date=d_end,
                                            bounding_box=utah_box,
                                            limit=10, sql_filters=['collection_number=="PRE"', "cloud_cover<=5"])
        rows = list(rows)
        self.assertEqual(len(rows), 1)

        #     metadata_row = ['LC80390332016208LGN00', '', 'LANDSAT_8', 'OLI_TIRS', '2016-07-26',
        # '2016-07-26T18:14:46.9465460Z', 'PRE', 'N/A', 'L1T', 39, 33, 1.69,
        # 39.96962, 37.81744, -115.27267, -112.56732, 1070517542,
        # 'gs://gcp-public-data-landsat/LC08/PRE/039/033/LC80390332016208LGN00']
        metadata = rows[0]

        # GDAL helper functions for generating VRT
        landsat = Landsat([metadata])

        # get a numpy.ndarray from bands for specified grpc
        band_numbers = [4, 3, 2]
        scale_params = [[0.0, 65535], [0.0, 65535], [0.0, 65535]]
        nda = landsat.fetch_imagery_array(band_numbers, scale_params)

        self.assertEqual(nda.shape, (3861, 3786, 3))

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

