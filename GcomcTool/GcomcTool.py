import os
import time
from datetime import datetime as dt
from datetime import timedelta
from glob import glob

import h5py
import json
import pycrs
import rasterio
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from ftplib import FTP
from osgeo import gdal, osr
from rasterio.mask import mask
from fiona.crs import from_epsg
from rasterio.enums import Resampling
from rasterio.merge import merge
from rasterio.plot import show
from shapely.geometry import box, Polygon


class GcomCpy:
    def __init__(self):
        print('This progdam was tested under the GDAL 3.2.2.')

    def get_meta(self, file_path, sub_dataset):
        self.file_path = file_path
        self.sub_dataset = sub_dataset
        file = gdal.Open(self.file_path)
        self.file_name = file.GetMetadata(
        )['Global_attributes_Product_file_name']
        granule_ID = self.file_name.replace(".h5", "")
        slope = float(file.GetMetadata()["Image_data_" + self.sub_dataset +
                                          "_Slope"])
        offset = float(file.GetMetadata()["Image_data_" + self.sub_dataset +
                                           "_Offset"])
        vtile = int(self.file_name[21:23])
        htile = int(self.file_name[23:25])
        resolution = self.file_name[-9]
        vtilenum = 18
        htilenum = 36
        sub_dataset = self.sub_dataset
        for i in range(len(file.GetSubDatasets())):
            name_of_subdataset = file.GetSubDatasets()[i][0]
            if (sub_dataset in name_of_subdataset) == True:
                image_array = gdal.Open(name_of_subdataset).ReadAsArray()
                break
            else:
                pass
        lintile = image_array.shape[0]
        coltile = image_array.shape[1]
        dlin = 180 / lintile / vtilenum
        dcol = 360 / coltile / htilenum
        d = dlin
        lat0 = 90 - vtile * 10 - d / 2
        lon0 = -180 + htile * 10 + d / 2

        self.lintile = lintile
        self.coltile = coltile

        self.slope = slope
        self.offset = offset

        self.d = d
        self.lat0 = lat0
        self.lon0 = lon0

    def lat_lon_calculator(self, col, lin):
        d = self.d
        lat0 = self.lat0
        lon0 = self.lon0

        lat = lat0 - lin * d
        r = np.cos((lat0 - lin * d) * np.pi / 180)
        lon = (lon0 + col * d) / r
        return lon, lat

    def query_tiles(self, box_coordinates, focus=False, help=True):
        self.box_coordinates = box_coordinates

        image_extent_library = []

        vtilenum = 18
        htilenum = 36
        lintile = 4800
        coltile = 4800
        dlin = 180 / lintile / vtilenum
        dcol = 360 / coltile / htilenum
        d = dlin

        self.d = d

        lat_lon_calculator = self.lat_lon_calculator
        for i in range(18):
            for j in range(36):
                vtile = i
                htile = j
                lat0 = 90 - vtile * 10 - d / 2
                lon0 = -180 + htile * 10 + d / 2
                self.lat0 = lat0
                self.lon0 = lon0

                ul_lon, ul_lat = lat_lon_calculator(0, 0)
                ur_lon, ur_lat = lat_lon_calculator(coltile - 1, 0)
                lr_lon, lr_lat = lat_lon_calculator(coltile - 1, lintile - 1)
                ll_lon, ll_lat = lat_lon_calculator(0, lintile - 1)
                image_extent = Polygon([[ul_lon, ul_lat], [ur_lon, ur_lat],
                                        [lr_lon, lr_lat], [ll_lon, ll_lat]])

                image_extent_library.append([i, j, image_extent])

        image_extent_library = gpd.GeoDataFrame(
            image_extent_library, columns=['vv', 'hh', 'geometry'])

        ul_lat, ul_lon = box_coordinates[3], box_coordinates[0]
        ur_lat, ur_lon = box_coordinates[3], box_coordinates[2]
        lr_lat, lr_lon = box_coordinates[1], box_coordinates[2]
        ll_lat, ll_lon = box_coordinates[1], box_coordinates[0]
        roi = Polygon([[ul_lon, ul_lat], [ur_lon, ur_lat], [lr_lon, lr_lat],
                       [ll_lon, ll_lat]])

        idx = image_extent_library.loc[image_extent_library.intersects(
            roi)].index
        queried_tiles = image_extent_library.loc[idx]

        tile_num_vv = []
        tile_num_hh = []
        for index in idx:
            vv = queried_tiles.loc[index]["vv"].tolist()
            hh = queried_tiles.loc[index]["hh"].tolist()
            print(f"Tile number vv:{vv} hh:{hh}")
            tile_num_vv.append(vv)
            tile_num_hh.append(hh)

        self.tile_num_vv = tile_num_vv
        self.tile_num_hh = tile_num_hh

        fig, ax = plt.subplots(figsize=(15, 10))
        queried_tiles.boundary.plot(ax=ax, color="black")

        world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
        idx = world.loc[world.intersects(roi)].index
        if len(idx) != 0:
            gcom_extent_country = world.loc[idx]
            gcom_extent_country.plot(ax=ax)
        else:
            world.plot(ax=ax)

        x, y = roi.exterior.xy
        ax.plot(x, y, color="red", label='The target area')
        ax.legend()

        if focus == True:
            bounds = queried_tiles.total_bounds
            ax.set_xlim(bounds[0] - 3, bounds[2] + 3)
            ax.set_ylim(bounds[1] - 3, bounds[3] + 3)
        else:
            pass

        if help == True:
            print(
                "The available L2 tile datasets can be found here: https://suzaku.eorc.jaxa.jp/GCOM_C/data/product_std.html"
            )
        else:
            pass

    def tile_calculator(self, vv, hh):
        vv = int(vv)
        hh = int(hh)
        if vv < 10 and hh < 10:
            tile = f"T0{vv}0{hh}"
        elif vv < 10 and hh > 9:
            tile = f"T0{vv}{hh}"
        elif vv > 9 and hh < 10:
            tile = f"T{vv}0{hh}"
        else:
            tile = f"T{vv}{hh}"

        return tile

    def filter_products(self,
                        product_name,
                        date_start,
                        date_end,
                        user_name,
                        orbit="D",
                        product_type="LAND",
                        statistics=True,
                        period="08D",
                        version=2):
        tile_num_vv = self.tile_num_vv
        tile_num_hh = self.tile_num_hh
        tile_calculator = self.tile_calculator

        strdt = dt.strptime(date_start, '%Y-%m-%d')
        enddt = dt.strptime(date_end, '%Y-%m-%d')
        days_num = (enddt - strdt).days + 1
        date_list = []
        for i in range(days_num):
            date = (strdt + timedelta(days=i)).strftime("%Y-%m-%d")
            date = date.replace("-", "")
            date_list.append(date)

        y_start = int(date_start[:4])
        m_start = date_start[5:7]
        d_start = date_start[8:10]

        y_end = int(date_end[:4])
        m_end = date_end[5:7]
        d_end = date_end[8:10]

        ftp = FTP('ftp.gportal.jaxa.jp', user_name, 'anonymous')
        self.ftp = ftp

        print('Filtering process has started...')

        #Date: yy-mm-dd
        if statistics == True:
            target_products = []
            N = len(tile_num_vv)
            for i in range(N):
                tile = tile_calculator(tile_num_vv[i], tile_num_hh[i])
                for date in date_list:
                    y = date[:4]
                    m = date[4:6]
                    d = date[6:8]
                    try:
                        item = ftp.nlst(
                            f"/standard/GCOM-C/GCOM-C.SGLI/L2.{product_type}.{product_name}.Statistics/{version}/{y}/{m}/*{date}{orbit}{period}*{tile}*"
                        )
                        print(item)
                        target_products.extend(item)
                    except:
                        pass
        else:
            target_products = []
            N = len(tile_num_vv)
            for i in range(N):
                tile = tile_calculator(tile_num_vv[i], tile_num_hh[i])
                for date in date_list:
                    y = date[:4]
                    m = date[4:6]
                    d = date[6:8]
                    item = ftp.nlst(
                        f"/standard/GCOM-C/GCOM-C.SGLI/L2.{product_type}.{product_name}/{version}/{y}/{m}/{d}/*{date}{orbit}*{tile}*"
                    )
                    print(item)
                    target_products.extend(item)

        self.target_products = target_products

    def get_products(self, download_path):
        self.download_path = download_path

        ftp = self.ftp
        target_products = self.target_products
        downloaded_products = []
        for file in tqdm(target_products):
            product_name = file[-37:]
            downloaded_products.append(download_path + "/" + product_name)
            with open(download_path + "/" + product_name, "wb") as f:
                ftp.retrbinary(f"RETR {file}", f.write)

        self.downloaded_products = downloaded_products
        ftp.close()

    def show_subdatasets(self):
        download_path = self.download_path
        downloaded_products = self.downloaded_products
        first = downloaded_products[0].replace("\\", "/")
        with h5py.File(first) as opened:
            print(opened['Image_data'].keys())

    def mosaic_images(self, file_list):
        read_product_list = []
        for file in file_list:
            opened = rasterio.open(file)
            read_product_list.append(opened)
        mosaic, out_trans = merge(read_product_list)

        return mosaic, out_trans, read_product_list

    def reproject_all(self,
                      sub_dataset,
                      folder_name="reprojectedFiles",
                      clip=False,
                      merge=True,
                      resampling=False,
                      resampling_method='nearest',
                      resolution_by_meter=None,
                      resolution_by_degree=None):
        download_path = self.download_path
        downloaded_products = self.downloaded_products
        roi = self.box_coordinates

        os.mkdir(download_path + f"/{folder_name}/")

        outputfile_list = []
        originalfile_list = []

        for file in tqdm(downloaded_products):
            inputfile_path = file.replace("\\", "/")

            file_name = file[-37:]
            outputfile_path = download_path + f"/{folder_name}/" + file_name.replace(
                ".h5", ".tif")
            self.reprojection(inputfile_path, sub_dataset, outputfile_path)
            outputfile_list.append(outputfile_path)

            if clip == True:
                self.translate_raster(roi, resampling, resampling_method,
                                      resolution_by_meter,
                                      resolution_by_degree)
                originalfile_list.append(outputfile_path)
            else:
                pass

        allfile_list = glob(download_path + f"/{folder_name}/*")
        allfile_list = [file.replace("\\", "/") for file in allfile_list]
        clippedfile_list = list(set(allfile_list) - set(originalfile_list))

        for file in originalfile_list:
            try:
                os.remove(file)
            except:
                time.sleep(3)
                os.remove(file)

        date_list = []
        for file in clippedfile_list:
            fileDate = file[-46:-38]
            date_list.append(fileDate)

        date_list = list(set(date_list))

        mosaic_images = self.mosaic_images
        for date in date_list:
            files = glob(download_path + f"/{folder_name}/*{date}*")
            mosaic, out_trans, opened_files = mosaic_images(files)

            with rasterio.open(download_path + f"/{folder_name}/" +
                               f"{sub_dataset}_{date}_mosaic.tif",
                               'w',
                               driver='GTiff',
                               width=mosaic[0].shape[1],
                               height=mosaic[0].shape[0],
                               count=1,
                               crs='EPSG:4326',
                               transform=out_trans,
                               dtype=mosaic[0].dtype) as output:
                output.write(mosaic[0], 1)
                output.close()

            for file in opened_files:
                file.close()

            for file in files:
                os.remove(file)

    def clean_up(self):
        downloaded_products = self.downloaded_products
        for file in downloaded_products:
            try:
                os.remove(file)
            except:
                time.sleep(3)
                try:
                    os.remove(file)
                except:
                    pass

    def visualize_extent(self,
                         file_path,
                         sub_dataset,
                         box_coordinates=None,
                         focus=False):
        self.get_meta(file_path, sub_dataset)
        lintile = self.lintile
        coltile = self.coltile
        lat_lon_calculator = self.lat_lon_calculator
        ul_lon, ul_lat = lat_lon_calculator(0, 0)
        ur_lon, ur_lat = lat_lon_calculator(coltile - 1, 0)
        lr_lon, lr_lat = lat_lon_calculator(coltile - 1, lintile - 1)
        ll_lon, ll_lat = lat_lon_calculator(0, lintile - 1)

        fig, ax = plt.subplots(figsize=(15, 10))

        gcom_extent = Polygon([[ul_lon, ul_lat], [ur_lon, ur_lat],
                               [lr_lon, lr_lat], [ll_lon, ll_lat]])
        x, y = gcom_extent.exterior.xy
        ax.plot(x, y, color="red", label='GCOM-C image extent')

        world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
        idx = world.loc[world.intersects(gcom_extent)].index
        gcom_extent_country = world.loc[idx]
        gcom_extent_country.plot(ax=ax)
        ax.legend()

        if box_coordinates != None:

            ul_lat, ul_lon = box_coordinates[3], box_coordinates[0]
            ur_lat, ur_lon = box_coordinates[3], box_coordinates[2]
            lr_lat, lr_lon = box_coordinates[1], box_coordinates[2]
            ll_lat, ll_lon = box_coordinates[1], box_coordinates[0]

            roi = Polygon([[ul_lon, ul_lat], [ur_lon, ur_lat],
                           [lr_lon, lr_lat], [ll_lon, ll_lat]])
            x, y = roi.exterior.xy
            ax.plot(x, y, color="black", label='The target area')
            ax.legend()
        else:
            pass

        if focus == True:
            ax.set_xlim(box_coordinates[0] - 0.3, box_coordinates[2] + 0.3)
            ax.set_ylim(box_coordinates[1] - 0.3, box_coordinates[3] + 0.3)
        else:
            pass

    def reprojection(self, file_path, sub_dataset, outputPath, n_GCPs=1681):
        self.outputPath = outputPath
        self.get_meta(file_path, sub_dataset)
        file = gdal.Open(self.file_path)

        lat_lon_calculator = self.lat_lon_calculator
        slope = self.slope
        offset = self.offset

        sub_dataset = self.sub_dataset

        for i in range(len(file.GetSubDatasets())):
            name_of_subdataset = file.GetSubDatasets()[i][0]
            if (sub_dataset in name_of_subdataset) == True:
                image_array = gdal.Open(name_of_subdataset).ReadAsArray()
                break
            else:
                pass

        image_array = image_array.astype("float")
        image_array[image_array == 65535.0] = np.nan
        image_array[image_array == 65534.0] = np.nan
        image_array = slope * image_array + offset

        dtype = gdal.GDT_Float32
        band = 1

        row = image_array.shape[0]
        col = image_array.shape[1]

        output_file = outputPath

        output = gdal.GetDriverByName('GTiff').Create(output_file, col, row,
                                                      band, dtype)
        output.GetRasterBand(1).WriteArray(image_array)

        gcp = []
        nrow = image_array.shape[0]
        ncol = image_array.shape[1]

        nspace = int(nrow / (np.sqrt(n_GCPs) - 1))

        col_idx = [i for i in range(0, nrow, nspace)]
        row_idx = [j for j in range(0, ncol, nspace)]

        col_idx.append(ncol - 1)
        row_idx.append(nrow - 1)

        for i in row_idx:
            for j in col_idx:
                i = int(i)
                j = int(j)
                lon, lat = lat_lon_calculator(i, j)
                GCP = gdal.GCP(lon, lat, 0, i, j)
                gcp.append(GCP)

        srs = osr.SpatialReference()
        srs.ImportFromEPSG(4326)
        output.SetProjection(srs.ExportToWkt())

        wkt = output.GetProjection()
        output.SetGCPs(gcp, wkt)
        output = gdal.Warp(output_file,
                           output,
                           dstSRS='EPSG:4326',
                           tps=True,
                           outputType=dtype)
        output = None

    def boundary_box(self, box_coordinates=None):
        path = self.outputPath
        src_image = rasterio.open(path)

        if box_coordinates == None:
            box_coordinates = list(src_image.bounds)
            bbox = box(box_coordinates[0], box_coordinates[1],
                       box_coordinates[2], box_coordinates[3])
        else:
            bbox = box(box_coordinates[0], box_coordinates[1],
                       box_coordinates[2], box_coordinates[3])

        geo = gpd.GeoDataFrame({'geometry': bbox},
                               index=[0],
                               crs=src_image.crs)
        coords = [json.loads(geo.to_json())['features'][0]['geometry']]
        self.mapCenter = ((box_coordinates[0] + box_coordinates[2]) / 2,
                          (box_coordinates[1] + box_coordinates[3]) / 2)

        out_img, out_transform = rasterio.mask.mask(src_image,
                                                    coords,
                                                    crop=True,
                                                    filled=False)
        self.width = out_img.shape[2]
        self.height = out_img.shape[1]
        self.transform = out_transform
        self.coords = coords

    def output_raster_meta(self,
                           resolution_by_meter=None,
                           resolution_by_degree=None):
        path = self.outputPath
        src_image = rasterio.open(path)

        if resolution_by_meter == None:
            resolution = src_image.meta['transform'][0]
        else:
            pass

        try:
            if (src_image.meta["crs"] == from_epsg(4326)) == True:
                d = resolution_by_meter / 1000
                r = 6371
                lat = np.radians(self.mapCenter[1])
                resolution = abs(np.degrees(2 * np.arcsin(np.sin(d /
                                                                 (2 * r)))))
            elif (resolution_by_degree != None) == True:
                resolution = resolution_by_degree
            else:
                resolution = resolution_by_meter
        except:
            pass

        src_affine = src_image.meta["transform"]
        src_width = src_image.meta["width"]
        src_height = src_image.meta["height"]
        target_transform, target_width, target_height = rasterio.warp.aligned_target(
            self.transform, self.width, self.height, resolution)

        self.height = target_height
        self.width = target_width
        self.count = src_image.meta["count"]
        self.crs = src_image.meta["crs"]
        self.transform = target_transform

    def resample_raster(self, resampling_method):
        path = self.outputPath
        if resampling_method == "nearest":
            resampling_method_func = Resampling.nearest
        elif resampling_method == "bilinear":
            resampling_method_func = Resampling.bilinear
        else:
            print(
                "Please specify the resampling method either nearest or bilinear."
            )

        name = path.replace(".tif", "")
        with rasterio.open(f"{name}_clipped.tif") as dataset:
            dataset = dataset.read(out_shape=(self.height, self.width),
                                   resampling=resampling_method_func)
        return dataset

    def write_clipped_raster(self):
        path = self.outputPath
        coords = self.coords

        data = rasterio.open(path)
        out_img, out_transform = rasterio.mask.mask(data,
                                                    coords,
                                                    crop=True,
                                                    filled=False)

        name = path.replace(".tif", "")
        dtype = out_img.dtype

        with rasterio.open(f"{name}_clipped.tif",
                           'w',
                           driver='GTiff',
                           width=out_img.shape[2],
                           height=out_img.shape[1],
                           count=data.count,
                           crs=data.crs,
                           transform=out_transform,
                           dtype=dtype) as output:
            for i in range(data.count):
                output.write(out_img[i], i + 1)
            output.close()
        data.close()

    def write_resampled_raster(self, raster):
        path = self.outputPath
        dtype = raster.dtype
        name = path.replace(".tif", "")
        with rasterio.open(f"{name}_clipped_resampled.tif",
                           'w',
                           driver='GTiff',
                           width=self.width,
                           height=self.height,
                           count=self.count,
                           crs=self.crs,
                           transform=self.transform,
                           dtype=dtype) as output:
            for i in range(self.count):
                output.write(raster[i], i + 1)
            output.close()

    def translate_raster(self,
                         box_coordinates=None,
                         resampling=False,
                         resampling_method='nearest',
                         resolution_by_meter=None,
                         resolution_by_degree=None):
        path = self.outputPath

        boundary_box = self.boundary_box
        output_raster_meta = self.output_raster_meta
        write_clipped_raster = self.write_clipped_raster
        resample_raster = self.resample_raster
        write_resampled_raster = self.write_resampled_raster

        boundary_box(box_coordinates)
        output_raster_meta(resolution_by_meter=resolution_by_meter,
                           resolution_by_degree=resolution_by_degree)
        write_clipped_raster()

        if resampling == True:
            dataset = resample_raster(resampling_method)
            write_resampled_raster(dataset)
        else:
            pass