import os
import time
import shutil
import warnings
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
import paramiko
from osgeo import gdal, osr
from rasterio.mask import mask
from fiona.crs import from_epsg
from rasterio.enums import Resampling
from rasterio.merge import merge
from rasterio.plot import show
from shapely.geometry import box, Polygon

warnings.simplefilter('ignore', RuntimeWarning)


class GcomCpy:
    def __init__(self):
        print('This program was tested under the GDAL 3.2.2.')

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

    def query_tiles(self, box_coordinates, focus=False, help=True,
                    show_map=False #To be fixed
                   ):
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
        
        if show_map==True:
            fig, ax = plt.subplots(figsize=(15, 10))
            queried_tiles.boundary.plot(ax=ax, color="black")

        #world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
        #idx = world.loc[world.intersects(roi)].index
        #if len(idx) != 0:
        #    gcom_extent_country = world.loc[idx]
        #    if show_map==True:
        #        gcom_extent_country.plot(ax=ax)
        #else:
        #    if show_map==True:
        #        world.plot(ax=ax)

        x, y = roi.exterior.xy
        
        if show_map==True:
            ax.plot(x, y, color="red", label='The target area')
            ax.legend()

        if focus == True:
            bounds = queried_tiles.total_bounds
            if show_map==True:
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

    def filter_products_tile(self,
                             product_name,
                             date_start,
                             date_end,
                             user_name,
                             password,
                             orbit="D",
                             product_type="LAND",
                             statistics=True,
                             period="08D",
                             version=3,
                             resolution='Q',
                            port=2051):
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

        #ftp = FTP('ftp.gportal.jaxa.jp', user_name, 'anonymous')
        #self.ftp = ftp
        transport=paramiko.Transport(('ftp.gportal.jaxa.jp',port))
        transport.connect(username=user_name,password=password)
        sftp=paramiko.SFTPClient.from_transport(transport)
        self.sftp=sftp

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
                        sftp_path=f"/standard/GCOM-C/GCOM-C.SGLI/L2.{product_type}.{product_name}.Statistics/{version}/{y}/{m}"
                        file_list=sftp.listdir(sftp_path)
                        pattern=[f'{date}{orbit}{period}',f'{tile}',f'{product_name}{resolution}']
                        item=[f for f in file_list if all([(r in f) for r in pattern])]
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
                    sftp_path=f"/standard/GCOM-C/GCOM-C.SGLI/L2.{product_type}.{product_name}/{version}/{y}/{m}/{d}"
                    file_list=sftp.listdir(sftp_path)
                    pattern=[f'{date}{orbit}{period}',f'{tile}',f'{product_name}{resolution}']
                    item=[f for f in file_list if all([(r in f) for r in pattern])]
                    print(item)
                    target_products.extend(item)

        self.target_products = target_products
        self.sftp_path=sftp_path

    def filter_products_global(self,
                               product_name,
                               date_start,
                               date_end,
                               user_name,
                               orbit="D",
                               product_type="ATMOS",
                               level=2,
                               version=2,
                               projection="D",
                               period="01M"):

        strdt = dt.strptime(date_start, '%Y-%m-%d')
        enddt = dt.strptime(date_end, '%Y-%m-%d')
        strdt = dt.strptime(date_start, '%Y-%m-%d')
        enddt = dt.strptime(date_end, '%Y-%m-%d')
        days_num = (enddt - strdt).days + 1
        date_list = []
        for i in range(days_num):
            date = (strdt + timedelta(days=i)).strftime("%Y-%m-%d")
            date = date.replace("-", "")
            date_list.append(date)

        ftp = FTP('ftp.gportal.jaxa.jp', user_name, 'anonymous')
        self.ftp = ftp

        print('Filtering process has started...')

        if level == 2:
            target_products = []
            for date in date_list:
                y = date[:4]
                m = date[4:6]
                d = date[6:8]
                try:
                    item = ftp.nlst(
                        f"/standard/GCOM-C/GCOM-C.SGLI/L2.{product_type}.{product_name}.Global/{version}/{y}/{m}/*{date}{orbit}*"
                    )
                    print(item)
                    target_products.extend(item)
                except:
                    pass
        else:
            target_products = []
            for date in date_list:
                y = date[:4]
                m = date[4:6]
                d = date[6:8]
                try:
                    item = ftp.nlst(
                        f"/standard/GCOM-C/GCOM-C.SGLI/L3.{product_type}.{product_name}/{version}/{y}/{m}/*{date}{orbit}{period}*{projection}0000*"
                    )
                    print(item)
                    target_products.extend(item)
                except:
                    pass
        self.target_products = target_products

    def get_products(self, download_path):
        self.download_path = download_path

        sftp = self.sftp
        target_products = self.target_products
        downloaded_products = []
        date_list=[]
        for file in tqdm(target_products):
            fileDate = os.path.splitext(os.path.basename(file))[0][7:15]

            date_list.append(fileDate)
            product_name = file[-37:]
            downloaded_products.append(download_path + "/" + product_name)
            localpath=dwnlooad_path+'/'+product_name
            if not os.path.exists(localpath):
                sftp.get(remotepath=self.sftp_path+'/'+file,localpath=localpath)

        self.downloaded_products = downloaded_products
        self.date_list=date_list
        ftp.close()

    def show_subdatasets(self, batch=True, path_to_product=None):
        if batch == True:
            download_path = self.download_path
            downloaded_products = self.downloaded_products
            first = downloaded_products[0].replace("\\", "/")
        else:
            first = path_to_product

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
                
        date_list=self.date_list

        #date_list = []
        #for file in downloaded_products:
            #fileDate = os.path.splitext(os.path.basename(file))[0][0:7]
            #date_list.append(fileDate)

        date_list = list(set(date_list))
        self.date_list = date_list

        mosaic_images = self.mosaic_images
        

        if clip == True:
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
        else:
            pass

    def reproject_all_global(self,
                             sub_dataset,
                             folder_name="reprojectedFiles",
                             temporal_merge=True,
                             merge_method='mean',
                             merge_image_name='merged_by_'):

        download_path = self.download_path
        downloaded_products = self.downloaded_products
        projection = downloaded_products[0][-24]
        os.mkdir(download_path + f"/{folder_name}/")

        flag = 0

        if projection == "D":
            reprojection_function = self.global_eqr
            flag = 1
        elif projection == "X":
            reprojection_function = self.global_eqa_1dim
        elif projection == "A":
            reprojection_function = self.global_eqa
        elif projection == "N":
            reprojection_function = self.polar_stereo
        elif projection == "S":
            reprojection_function = self.polar_stereo
        else:
            pass

        output_file_path_list = []

        for file_path in downloaded_products:
            file_path = file_path.replace("\\", "/")
            file_name = file_path[-37:]
            output_path = download_path + f"/{folder_name}/" + file_name.replace(
                ".h5", ".tif")
            reprojection_function(file_path, sub_dataset, output_path)
            output_file_path_list.append(output_path)

        if temporal_merge == True:
            if flag == 0:
                file_list = glob(download_path + f"/{folder_name}/*")
                with rasterio.open(file_list[0]) as opened:
                    shape = opened.read(1).shape
                opened_array_list = []
                for file in file_list:
                    with rasterio.open(file) as opened:
                        opened_array = opened.read(1).flatten()
                        opened_array_list.append(opened_array)

                if merge_method == 'mean':
                    output_array = np.nanmean(opened_array_list, axis=0)
                elif merge_method == 'max':
                    output_array = np.nanmax(opened_array_list, axis=0)
                elif merge_method == 'min':
                    output_array = np.nanmin(opened_array_list, axis=0)
                else:
                    print(
                        'The available merging methods are mean, max, or min.')

                output_array = output_array.reshape(shape)
                image = Image.fromarray(output_array)
                image.save(
                    download_path +
                    f"/{folder_name}/{merge_image_name}{merge_method}.tif")

            else:
                file_list = glob(download_path + f"/{folder_name}/*")
                with rasterio.open(file_list[0]) as opened:
                    ref = opened
                    shape = opened.read(1).shape
                opened_array_list = []
                for file in file_list:
                    with rasterio.open(file) as opened:
                        opened_array = opened.read(1).flatten()
                        opened_array_list.append(opened_array)

                if merge_method == 'mean':
                    output_array = np.nanmean(opened_array_list, axis=0)
                elif merge_method == 'max':
                    output_array = np.nanmax(opened_array_list, axis=0)
                elif merge_method == 'min':
                    output_array = np.nanmin(opened_array_list, axis=0)
                else:
                    print(
                        'The available merging methods are mean, max, or min.')
                output_array = output_array.reshape(shape)

                with rasterio.open(
                        download_path +
                        f"/{folder_name}/{merge_image_name}{merge_method}.tif",
                        'w',
                        driver='GTiff',
                        width=output_array.shape[1],
                        height=output_array.shape[0],
                        count=1,
                        crs='EPSG:4326',
                        transform=ref.transform,
                        dtype=output_array.dtype) as output:
                    output.write(output_array, 1)
                    output.close()
            for file in output_file_path_list:
                try:
                    os.remove(file)
                except:
                    time.sleep(3)
                    os.remove(file)

        else:
            pass

    def combine_rsrf_tile(self,
                          output_folder_name='combined',
                          bands_250m=[
                              'Rs_VN01', 'Rs_VN02', 'Rs_VN03', 'Rs_VN04',
                              'Rs_VN05', 'Rs_VN06', 'Rs_VN07', 'Rs_VN08',
                              'Rs_VN10', 'Rs_VN11', 'Rs_SW03', 'Tb_TI01',
                              'Tb_TI02'
                          ],
                         bands_1000m=[]):
        download_path = self.download_path
        os.mkdir(download_path + '/temp_250m')
        reproject_func = self.reproject_all
        align_raster=self.align_raster
        date_list = self.date_list
        
        for band in bands_250m:
            print(f'Processing band {band}...')
            reproject_func(band,
                           folder_name=f"temp_250m/{band}",
                           clip=True,
                           merge=True)
        
        band = bands_250m[0]
        with rasterio.open(glob(download_path + f"/temp_250m/{band}/*")[0]) as ref:
            ref_array = ref.read(1)
            height = ref.shape[0]
            width = ref.shape[1]
            transform = ref.transform
        
        
        if len(bands_1000m)!=0:
            os.mkdir(download_path + '/temp_1000m')
            for band in bands_1000m:
                os.mkdir(download_path + f'/temp_250m/{band}')
                print(f'Processing band {band}...')
                reproject_func(band,folder_name=f"temp_1000m/{band}",clip=True,merge=True)
                
            ref_image_path=glob(download_path + f"/temp_250m/{bands_250m[0]}/*")[0]
            for band in bands_1000m:
                for i in range(len(date_list)):
                    date=date_list[i]
                    target_path=download_path + '/' +f"temp_1000m/{band}/{band}_{date}_mosaic.tif"
                    align_raster(ref_image_path,target_path)
                    
            for band in bands_1000m:
                for i in range(len(date_list)):
                    shutil.move(download_path + '/' +f"temp_1000m/{band}/{band}_{date}_mosaic_aligned.tif",
                               download_path + f'/temp_250m/{band}')
                    os.rename(download_path + '/' +f"temp_250m/{band}/{band}_{date}_mosaic_aligned.tif",
                             download_path + '/' +f"temp_250m/{band}/{band}_{date}_mosaic.tif")
        else:
            pass
        
        bands=bands_250m+bands_1000m
        
        os.mkdir(download_path + '/' + output_folder_name)
        for date in date_list:
            with rasterio.open(download_path + '/' + output_folder_name + '/' +
                               f'RSRF_{date}.tif',
                               'w',
                               driver='GTiff',
                               count=len(bands),
                               width=width,
                               height=height,
                               transform=transform,
                               crs='EPSG:4326',
                               dtype=ref_array.dtype) as output:
                cnt = 1
                for sub_dataset in bands:
                    with rasterio.open(
                            download_path + '/' +
                            f"temp_250m/{sub_dataset}/{sub_dataset}_{date}_mosaic.tif"
                    ) as opened:
                        array = opened.read(1)
                    output.write(array, cnt)
                    output.set_band_description(cnt, sub_dataset)
                    cnt += 1
                output.close()
        shutil.rmtree(download_path + '/temp_250m')
        if len(bands_1000m)!=0:
            shutil.rmtree(download_path + '/temp_1000m')
        else:
            pass


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

        #if box_coordinates != None:
        if len(box_coordinates) != 0:

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

        srs = osr.SpatialReference()
        srs.ImportFromEPSG(4326)
        output.SetProjection(srs.ExportToWkt())

        output = None

    def reprojection_scene(self,
                           file_path,
                           subdataset,
                           output_path,
                           interval=80):
        opened = gdal.Open(file_path)

        for i in range(len(opened.GetSubDatasets())):
            if 'Longitude' in opened.GetSubDatasets()[i][0]:
                lon_array = gdal.Open(
                    opened.GetSubDatasets()[i][0]).ReadAsArray()
            else:
                pass

        for i in range(len(opened.GetSubDatasets())):
            if 'Latitude' in opened.GetSubDatasets()[i][0]:
                lat_array = gdal.Open(
                    opened.GetSubDatasets()[i][0]).ReadAsArray()
            else:
                pass

        for i in range(len(opened.GetSubDatasets())):
            if subdataset in opened.GetSubDatasets()[i][0]:
                image_array = gdal.Open(
                    opened.GetSubDatasets()[i][0]).ReadAsArray().astype(
                        np.float)
            else:
                pass

        error_dn = int(
            opened.GetMetadata()[f'Image_data_{subdataset}_Error_DN'].replace(
                'd ', ''))
        try:
            slope = float(
                opened.GetMetadata()[f'Image_data_{subdataset}_Slope'].replace(
                    'd ', ''))
            offset = float(opened.GetMetadata()
                           [f'Image_data_{subdataset}_Offset'].replace(
                               'd ', ''))
        except:
            pass

        image_array[image_array == error_dn] = np.nan

        try:
            phys_value_array = slope * image_array + offset
        except:
            phys_value_array = image_array

        gcp = []

        nspace = int(interval / 10)

        col = phys_value_array.shape[1]
        row = phys_value_array.shape[0]
        band = 1
        dtype = gdal.GDT_Float32

        output_file = output_path
        output = gdal.GetDriverByName('GTiff').Create(output_file, col, row,
                                                      band, dtype)
        output.GetRasterBand(1).WriteArray(phys_value_array)

        row_idx = [j for j in range(0, lat_array.shape[0], nspace)]
        col_idx = [i for i in range(0, lat_array.shape[1], nspace)]

        row_idx.append(lon_array.shape[0] - 1)
        col_idx.append(lat_array.shape[1] - 1)

        for i in row_idx:
            for j in col_idx:
                i = int(i)
                j = int(j)

                col = 10 * i - 1
                row = 10 * j - 1

                col = int(col)
                row = int(row)

                if col > 0 and row > 0:
                    pass
                elif row > 0 and col < 0:
                    col = 0
                elif col > 0 and row < 0:
                    row = 0
                else:
                    row = 0
                    col = 0

                lat = lat_array[i, j].astype(np.float64)
                lon = lon_array[i, j].astype(np.float64)

                GCP = gdal.GCP(lon, lat, 0, row, col)
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

        srs = osr.SpatialReference()
        srs.ImportFromEPSG(4326)
        output.SetProjection(srs.ExportToWkt())

        output = None

    def boundary_box(self, box_coordinates=None):
        path = self.outputPath
        src_image = rasterio.open(path)

        if len(box_coordinates) == 0:
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

    def global_eqa(self, file_path, subdataset, output_path):
        opened = gdal.Open(file_path)
        error_dn = int(
            opened.GetMetadata()[f'Image_data_{subdataset}_Error_DN'].replace(
                'd ', ''))
        slope = float(opened.GetMetadata()[f'Image_data_{subdataset}_Slope'])
        offset = float(opened.GetMetadata()[f'Image_data_{subdataset}_Offset'])

        for i in range(len(opened.GetSubDatasets())):
            if subdataset in opened.GetSubDatasets()[i][0]:
                image_array = gdal.Open(
                    opened.GetSubDatasets()[i][0]).ReadAsArray().astype(
                        np.float)
            else:
                pass

        image_array[image_array == error_dn] = np.nan
        phys_value_array = slope * image_array + offset

        img = Image.fromarray(phys_value_array)
        img.save(output_path)

        del opened

    def polar_stereo(self, file_path, subdataset, output_path):
        opened = gdal.Open(file_path)
        error_dn = int(
            opened.GetMetadata()[f'Image_data_{subdataset}_Error_DN'].replace(
                'd ', ''))
        slope = float(
            opened.GetMetadata()[f'Image_data_{subdataset}_Slope'].replace(
                'd ', ''))
        offset = float(
            opened.GetMetadata()[f'Image_data_{subdataset}_Offset'].replace(
                'd ', ''))

        for i in range(len(opened.GetSubDatasets())):
            if subdataset in opened.GetSubDatasets()[i][0]:
                image_array = gdal.Open(
                    opened.GetSubDatasets()[i][0]).ReadAsArray().astype(
                        np.float)
            else:
                pass

        image_array[image_array == error_dn] = np.nan
        phys_value_array = slope * image_array + offset
        image = Image.fromarray(phys_value_array)
        image.save(output_path)

        del opened

    def global_eqr(self, file_path, subdataset, output_path):
        opened = gdal.Open(file_path)
        error_dn = int(
            opened.GetMetadata()[f'Image_data_{subdataset}_Error_DN'].replace(
                'd ', ''))
        slope = float(
            opened.GetMetadata()[f'Image_data_{subdataset}_Slope'].replace(
                'd ', ''))
        offset = float(
            opened.GetMetadata()[f'Image_data_{subdataset}_Offset'].replace(
                'd ', ''))

        for i in range(len(opened.GetSubDatasets())):
            if subdataset in opened.GetSubDatasets()[i][0]:
                image_array = gdal.Open(
                    opened.GetSubDatasets()[i][0]).ReadAsArray().astype(
                        np.float)
            else:
                pass

        image_array[image_array == error_dn] = np.nan
        phys_value_array = slope * image_array + offset
        transform = rasterio.transform.from_bounds(-180, -90, 180, 90,
                                                   image_array.shape[1],
                                                   image_array.shape[0])

        with rasterio.open(output_path,
                           'w',
                           driver='GTiff',
                           width=phys_value_array.shape[1],
                           height=phys_value_array.shape[0],
                           count=1,
                           crs='EPSG:4326',
                           transform=transform,
                           dtype=phys_value_array.dtype) as output:
            output.write(phys_value_array, 1)
            output.close()

        del opened

    def num_bin_calculator(self, x, res, nrow):
        return np.round(2 * nrow * np.cos(
            (90 - ((x - 1) * (res) + (res) * (1 / 2))) * np.pi / 180))
    
    def align_raster(self,path_to_ref_image,path_to_target_image,remove=True,nodata=np.nan):
        with rasterio.open(path_to_ref_image) as ref:
            bounds=ref.bounds
            height=ref.meta["height"]
            width=ref.meta["width"]
            crs=ref.meta["crs"]
            transform=ref.meta["transform"]

        bbox = box(bounds[0], bounds[1], bounds[2], bounds[3])
        geo = gpd.GeoDataFrame({'geometry': bbox}, index=[0], crs=crs)
        geo = geo.to_crs(crs=crs)
        coords=[json.loads(geo.to_json())['features'][0]['geometry']]
        
        dst_path=path_to_target_image.replace(os.path.basename(path_to_target_image),'')
        filename=os.path.splitext(os.path.basename(path_to_target_image))[0]
        os.mkdir(dst_path+'/temp')
        
        with rasterio.open(path_to_target_image) as src:
            out_img, out_transform = rasterio.mask.mask(src, coords, crop=True,filled=False,nodata=nodata)
            src_crs=src.crs
            src_count=src.count
            dtype=out_img.dtype
        
        with rasterio.open(dst_path+'/temp'+f'/{filename}_clipped.tif','w',driver='GTiff',
                           width=out_img.shape[2],
                           height=out_img.shape[1],
                           count=src_count,
                           crs=src_crs,
                           transform=out_transform,
                           dtype=dtype) as output:
            for i in range(src_count):
                output.write(out_img[i],i+1)
            output.close()
        
        with rasterio.open(dst_path+'/temp'+f'/{filename}_clipped.tif') as src_clipped:
            clipped_resampled=src_clipped.read(out_shape=(height,width),resampling=Resampling.nearest)
        
        with rasterio.open(dst_path+f'/{filename}_aligned.tif',
                           'w',
                           driver='GTiff',
                           width=width,
                           height=height,
                           count=src_count,
                           crs=crs,
                           transform=transform,
                           dtype=dtype) as dst:
            for i in range(src_count):
                dst.write(clipped_resampled[i],i+1)
            dst.close()
            
        shutil.rmtree(dst_path+'/temp')
        
        if remove==True:
            os.remove(path_to_target_image)
        else:
            pass
        
    def global_eqa_1dim(self, file_path, subdataset, output_path):
        opened_gdal = gdal.Open(file_path)
        error_dn = int(opened_gdal.GetMetadata()
                       [f'Image_data_{subdataset}_Error_DN'].replace('d ', ''))
        slope = float(opened_gdal.GetMetadata()
                      [f'Image_data_{subdataset}_Slope'].replace('d ', ''))
        offset = float(opened_gdal.GetMetadata()
                       [f'Image_data_{subdataset}_Offset'].replace('d ', ''))

        opened = h5py.File(file_path)
        opened_vector = opened['Image_data'][subdataset][()]

        num_bin_calculator = self.num_bin_calculator

        resolution = file_path[-9]

        if resolution == 'F':
            nrow = 4320
        else:
            nrow = 2160

        ncol = int(nrow * 2)
        output_E = np.zeros([int(nrow), int(ncol / 2)])
        output_W = np.zeros([int(nrow), int(ncol / 2)])

        containor = 0
        for i in tqdm(range(nrow)):
            bins = num_bin_calculator(i + 1, 1 / 24, nrow)
            ith_row_values = opened_vector[int(containor):int(containor +
                                                              bins)]
            ith_row_values_E = opened_vector[int(containor):int(containor +
                                                                (bins / 2 +
                                                                 bins % 2))]
            ith_row_values_W = opened_vector[int(containor + bins / 2 +
                                                 bins % 2):int(containor +
                                                               bins)]
            output_E[i:i + 1,
                     int(ncol / 2 -
                         len(ith_row_values_E)):ncol] = ith_row_values_E
            output_W[i:i + 1, 0:len(ith_row_values_W)] = ith_row_values_W
            containor = containor + bins

        output_E = np.fliplr(output_E)
        output_W = np.fliplr(output_W)

        output_combined = np.concatenate([output_W, output_E], axis=1)
        output_combined = np.flipud(output_combined)
        output_combined = np.fliplr(output_combined)
        output_combined[output_combined == error_dn] = np.nan
        output_combined = slope * output_combined + offset

        img = Image.fromarray(output_combined.astype(np.float))
        img.save(output_path)

        del opened_gdal, opened
       
