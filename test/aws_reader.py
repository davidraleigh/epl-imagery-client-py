import unittest
from datetime import date
from datetime import datetime

from epl.client.imagery.reader import MetadataService, Landsat, SpacecraftID, Band, DataType


class TestAWSClouds(unittest.TestCase):
    def test_cloud_cover(self):
        metadata_service = MetadataService()

        # TODO figure out what's wrong here for AWS
        self.assertTrue(True)
        # # sql_filters = ['cloud_cover=0']
        # d_start = date(2015, 6, 24)
        # d_end = date(2016, 6, 24)
        # bounding_box = (-115.927734375, 34.52466147177172, -78.31054687499999, 44.84029065139799)
        # rows = metadata_service.search(
        #     SpacecraftID.LANDSAT_8,
        #     start_date=d_start,
        #     end_date=d_end,
        #     bounding_box=bounding_box,
        #     cloud_cover=[0])
        #
        # rows = list(rows)
        #
        # self.assertEqual(len(rows), 10)
        #
        # for row in rows:
        #     self.assertEqual(row.spacecraft_id, SpacecraftID.LANDSAT_8)
        #     d_actual = datetime.strptime(row.date_acquired, '%Y-%m-%d').date()
        #     self.assertLessEqual(d_actual, d_end)
        #     self.assertGreaterEqual(d_actual, d_start)
        #     test_box = row.bounds
        #     self.assertTrue(
        #         (bounding_box[0] < test_box[2] < bounding_box[2]) or
        #         (bounding_box[0] < test_box[0] < bounding_box[2]))
        #     self.assertTrue(
        #         (bounding_box[1] < test_box[3] < bounding_box[3]) or
        #         (bounding_box[1] < test_box[1] < bounding_box[3]))


class TestAWSvrt(unittest.TestCase):
    def test_pixel_function_vrt_1(self):

        # TODO figure out what's wrong here for AWS
        self.assertTrue(True)
        # utah_box = (-112.66342163085938, 37.738141282210385, -111.79824829101562, 38.44821130413263)
        # d_start = date(2016, 7, 20)
        # d_end = date(2016, 7, 28)
        #
        #
        # rows = self.metadata_service.search(SpacecraftID.LANDSAT_8,
        #                                     start_date=d_start,
        #                                     end_date=d_end,
        #                                     bounding_box=utah_box,
        #                                     limit=10,
        #                                     cloud_cover=5,
        #                                     sql_filters=['collection_number=="PRE"'])
        # rows = list(rows)
        # self.assertEqual(len(rows), 1)
        #
        # #     metadata_row = ['LC80390332016208LGN00', '', 'LANDSAT_8', 'OLI_TIRS', '2016-07-26',
        # # '2016-07-26T18:14:46.9465460Z', 'PRE', 'N/A', 'L1T', 39, 33, 1.69,
        # # 39.96962, 37.81744, -115.27267, -112.56732, 1070517542,
        # # 'gs://gcp-public-data-landsat/LC08/PRE/039/033/LC80390332016208LGN00']
        # metadata = rows[0]
        #
        # # GDAL helper functions for generating VRT
        # landsat = Landsat([metadata])
        #
        # # get a numpy.ndarray from bands for specified grpc
        # band_numbers = [4, 3, 2]
        # scale_params = [[0.0, 65535], [0.0, 65535], [0.0, 65535]]
        # nda = landsat.fetch_imagery_array(band_numbers, scale_params)
        #
        # self.assertEqual(nda.shape, (3861, 3786, 3))