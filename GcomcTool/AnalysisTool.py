import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point
import rasterio
import math
from tqdm import tqdm
import pandas as pd
from glob import glob
import os
from shapely import wkt


class AnalysisTool:
    def __init__(self):
        pass

    def train_point_handler(self, shapefile_path, image_path, num_points=1000):
        raster = rasterio.open(image_path)
        gdf = gpd.read_file(shapefile_path)

        training_points_candidates = []

        while len(training_points_candidates) < num_points:
            idx = np.random.randint(0, gdf['geometry'].count())
            bounds = gdf[gdf['geometry'].index == idx].total_bounds

            xmin = bounds[0]
            xmax = bounds[2]
            ymin = bounds[1]
            ymax = bounds[3]

            x = np.random.uniform(xmin, xmax)
            y = np.random.uniform(ymin, ymax)
            point = Point(x, y)

            if len(gdf[gdf['geometry'].index == idx].loc[gdf[
                    gdf['geometry'].index == idx].intersects(
                        point)].index) == 0:
                pass
            else:
                training_points_candidates.append(point)

        counts = raster.meta['count']
        values = []
        training_point = []

        for point in tqdm(training_points_candidates):
            lon = point.x
            lat = point.y
            val_list = []

            for i in range(counts):
                array = raster.read(i + 1)
                val = array[raster.index(lon, lat)]
                val_list.append(val)

            if np.isnan(val_list).any() == True:
                pass
            else:
                values.append(val_list)
                training_point.append(point)

        del raster

        return values, training_point

    def train_data_process(self,
                           shapefile_dir_path,
                           image_path,
                           num_points=1000):
        train_point_handler = self.train_point_handler
        shapefile_list = glob(shapefile_dir_path + '/*.shp')

        extracted_values = []
        lulc_class = []
        lulc_name = []
        lat=[]
        lon=[]

        cnt = 1
        for shapefile in shapefile_list:
            filename = os.path.basename(shapefile).replace('.shp', '')
            values, training_point = train_point_handler(
                shapefile, image_path, num_points)

            extracted_values.extend(values)
            
            lat.extend([point.x for point in training_point])
            lon.extend([point.y for point in training_point])

            lulc_name.extend([filename for i in range(len(values))])
            lulc_class.extend([cnt for i in range(len(values))])
            cnt += 1

        output_df = pd.DataFrame({
            'lulc_name': lulc_name,
            'lulc_class': lulc_class,
            'extracted_values': extracted_values,
            'Latitude': lat,
            'Longitude': lon
        })
        
        output_gdf=gpd.GeoDataFrame(output_df,geometry=gpd.points_from_xy(output_df.Latitude,output_df.Longitude))
        
        return output_gdf