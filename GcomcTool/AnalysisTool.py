import geopandas as gpd
import warnings
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
from sklearn.decomposition import PCA
from rasterio.plot import show
from PIL import Image, ImageOps

from sklearn import metrics

from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier

warnings.simplefilter(action='ignore',category=FutureWarning)

class AnalysisTool:
    def __init__(self):
        pass

    def train_point_handler(self, shapefile_path, image_path, num_points=1000):
        raster = rasterio.open(image_path)
        gdf = gpd.read_file(shapefile_path)
        
        counts = raster.meta['count']
        values = []
        training_points = []
        

        while len(training_points) < num_points:
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
                val_list = []
                for i in range(counts):
                    array = raster.read(i + 1)
                    val = array[raster.index(x, y)]
                    val_list.append(val)
                if np.isnan(val_list).any() == True:
                    pass
                else:
                    values.append(val_list)
                    training_points.append(point)


        del raster

        return values, training_points

    def train_data_process(self,
                           shapefile_dir_path,
                           image_path=None,
                           num_points=1000,
                           write_train_data=False,
                           output_train_data_path=None,
                           output_train_data_epsg=4326):
        train_point_handler = self.train_point_handler
        shapefile_list = glob(shapefile_dir_path + '/*.shp')

        extracted_values = []
        lulc_class = []
        lulc_name = []
        lat = []
        lon = []

        cnt = 1
        for shapefile in tqdm(shapefile_list):
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

        output_gdf = gpd.GeoDataFrame(output_df,
                                      geometry=gpd.points_from_xy(
                                          output_df.Latitude,
                                          output_df.Longitude))
        output_gdf.set_crs(epsg=4326,inplace=True)
        
        if write_train_data==True:
            output_gdf=output_gdf.to_crs({'init': f'epsg:{output_train_data_epsg}'})
            output_gdf.drop('extracted_values',axis=1).to_file(output_train_data_path)
        else:
            pass

        return output_gdf

    def pca_image(self,
                  image_path,
                  RGBoption=False,
                  return_arrays=False,
                  n_components=10,
                  exception=[],nanvalue=np.nan):
        raster = rasterio.open(image_path)
        width = raster.meta["width"]
        height = raster.meta["height"]
        shape = (height, width)

        counts = raster.meta['count']
        bands_opened = []
        cnt = 1
        for i in range(counts):
            if (cnt in exception) == True:
                pass
            else:
                array = raster.read(cnt).flatten()
                array = np.nan_to_num(array)
                bands_opened.append(array)
            cnt += 1

        bands_stacked = np.dstack(tuple(bands_opened))
        pca = PCA(n_components)
        pca.fit(bands_stacked[0])
        variance = pca.explained_variance_
        totalVariance = np.sum(variance)
        variance = variance / totalVariance * 100

        fig, ax = plt.subplots()

        ax.plot(variance)
        ax.set_xlabel("Components of PCA")
        ax.set_ylabel("Variance explained [%]")

        pca_image = pca.transform(bands_stacked[0])

        first_component = self.nan_mask(image_path,pca_image[:, 0].reshape(shape),nanvalue)
        second_component = self.nan_mask(image_path,pca_image[:, 1].reshape(shape),nanvalue)
        third_component = self.nan_mask(image_path,pca_image[:, 2].reshape(shape),nanvalue)

        if RGBoption == True:
            self.visualize([0, 1, 2],
                           array=np.stack([
                               first_component, second_component,
                               third_component
                           ]))
        else:
            first_component = first_component
            show(first_component)

        if return_arrays == True:
            return [
                pca_image[:, i].reshape(shape) for i in range(n_components)
            ]
        else:
            pass

    def visualize(self, rgb_band, array=None, image_path=None, nanvalue=np.nan):
        if image_path != None:
            raster = rasterio.open(image_path)
            R = raster.read(rgb_band[0])
            G = raster.read(rgb_band[1])
            B = raster.read(rgb_band[2])
            
            R=self.nan_mask(image_path,R,nanvalue)
            G=self.nan_mask(image_path,G,nanvalue)
            B=self.nan_mask(image_path,B,nanvalue)

            R = (255 / np.nanmax(R) * R).astype(np.uint8)
            G = (255 / np.nanmax(G) * G).astype(np.uint8)
            B = (255 / np.nanmax(B) * B).astype(np.uint8)

            image_stack = Image.fromarray(np.dstack((R, G, B)))
            image = ImageOps.equalize(image_stack)
            image.show()
        else:
            R = array[rgb_band[0]]
            G = array[rgb_band[1]]
            B = array[rgb_band[2]]

            R = (255 / np.nanmax(R) * R).astype(np.uint8)
            G = (255 / np.nanmax(G) * G).astype(np.uint8)
            B = (255 / np.nanmax(B) * B).astype(np.uint8)

            image_stack = Image.fromarray(np.dstack((R, G, B)))
            image = ImageOps.equalize(image_stack)
            image.show()

    def un_supervised_classification(
            self,
            path_to_image,
            output_path,
            train_data_path,
            method='GaussianMixture',
            n=30,
            num_points=1000,
            preview=True,
            output_file_name='un_supervised_classification',
            params=None,
            nanvalue=np.nan):

        input_data = self.train_data_process(train_data_path, path_to_image,
                                             num_points)

        data = input_data['extracted_values']
        data = [spectrum for spectrum in data]
        X = np.array(data)

        if method == 'GaussianMixture':
            if params != None:
                model = GaussianMixture(**params).fit(X)
            else:
                model = GaussianMixture(n_components=n).fit(X)
        elif method == 'KMeans':
            if params != None:
                model = KMeans(**params).fit(X)
            else:
                model = KMeans(n_clusters=n).fit(X)
        else:
            print(
                'The available methods are either GaussianMixture or KMeans.')

        raster = rasterio.open(path_to_image)
        counts = raster.meta['count']

        image_array = []
        for i in range(counts):
            image_array.append(raster.read(i + 1))

        cluster_output = []

        for i in tqdm(range(image_array[0].shape[0])):
            spectrum_data_at_row_i = []
            spectrum_data_at_row_i_nan_check = []
            for m in range(counts):
                spectrum_data_at_row_i.append(np.nan_to_num(image_array[m][i]))
                spectrum_data_at_row_i_nan_check.append(image_array[m][i])

            spectrum_data_at_row_i_T = np.transpose(
                np.array(spectrum_data_at_row_i))
            spectrum_data_at_row_i_nan_check_T = np.transpose(
                np.array(spectrum_data_at_row_i_nan_check))

            nan_ind = np.where(np.isnan(spectrum_data_at_row_i_nan_check_T))[0]
            classified = np.transpose(
                model.predict(spectrum_data_at_row_i_T)).astype(np.float)
            classified[nan_ind] = nanvalue
            cluster_output.append(classified)

        output_image = np.array(cluster_output)

        output_tif_name = output_file_name + '_' + method + '.tif'

        with rasterio.open(output_path + '/' + output_tif_name,
                           'w',
                           height=output_image.shape[0],
                           width=output_image.shape[1],
                           driver='GTiff',
                           count=1,
                           crs=raster.crs,
                           transform=raster.transform,
                           dtype=output_image.dtype) as output:
            output.write(output_image, 1)
            output.close()

        if preview == True:
            fig, ax = plt.subplots(figsize=(15, 15))
            rasterio.plot.show(output_image, ax=ax, cmap='hsv')
        else:
            pass

        del raster

    def supervised_classification(self,
                                  path_to_image,
                                  output_path,
                                  train_data_path,
                                  method='RandomForest',
                                  num_points=1000,
                                  preview=True,
                                  output_file_name='supervised_classification',
                                  test_size=0.2,
                                  params={},
                                  nanvalue=np.nan):

        input_data = self.train_data_process(train_data_path, path_to_image,
                                             num_points)

        data = input_data['extracted_values']
        data = [spectrum for spectrum in data]
        label = input_data['lulc_class']
        label = [l for l in label]

        X = np.array(data)
        y = np.array(label)

        X_train, X_test, y_train, y_test = train_test_split(X,
                                                            y,
                                                            test_size=0.2)

        if method == 'RandomForest':
            model = RandomForestClassifier(**params)
        elif method == 'NeuralNetwork':
            model = MLPClassifier(**params)
        elif method == 'KNeighbors':
            model = KNeighborsClassifier(**params)
        elif method == 'GradientBoosting':
            model = GradientBoostingClassifier(**params)
        else:
            print(
                'The available methods are either RandomForest, NeuralNetwork, Kneighbors, or GradientBoosting.'
            )

        model.fit(X_train, y_train)

        confusion_matrix = metrics.confusion_matrix(y_test,
                                                    model.predict(X_test))
        kappa = metrics.cohen_kappa_score(y_test, model.predict(X_test))

        print(f'Confusion matrix is:')
        print(confusion_matrix)
        print(f'Kappa coefficient {kappa:.5f}')

        raster = rasterio.open(path_to_image)
        counts = raster.meta['count']

        image_array = []
        for i in range(counts):
            image_array.append(raster.read(i + 1))

        classified_output = []

        for i in tqdm(range(image_array[0].shape[0])):
            spectrum_data_at_row_i = []
            spectrum_data_at_row_i_nan_check = []
            for m in range(counts):
                spectrum_data_at_row_i.append(np.nan_to_num(image_array[m][i]))
                spectrum_data_at_row_i_nan_check.append(image_array[m][i])

            spectrum_data_at_row_i_T = np.transpose(
                np.array(spectrum_data_at_row_i))
            spectrum_data_at_row_i_nan_check_T = np.transpose(
                np.array(spectrum_data_at_row_i_nan_check))

            nan_ind = np.where(np.isnan(spectrum_data_at_row_i_nan_check_T))[0]

            classified = np.transpose(
                model.predict(spectrum_data_at_row_i_T)).astype(np.float)
            classified[nan_ind] = nanvalue
            classified_output.append(classified)

        output_image = np.array(classified_output)
        output_tif_name = output_file_name + '_' + method + '.tif'

        with rasterio.open(output_path + '/' + output_tif_name,
                           'w',
                           height=output_image.shape[0],
                           width=output_image.shape[1],
                           driver='GTiff',
                           count=1,
                           crs=raster.crs,
                           transform=raster.transform,
                           dtype=output_image.dtype) as output:
            output.write(output_image, 1)
            output.close()

        if preview == True:
            fig, ax = plt.subplots(figsize=(15, 15))
            rasterio.plot.show(output_image, ax=ax, cmap='jet')
        else:
            pass

        del raster
    
    def nan_mask(self,path_to_image,input_array,nanvalue):
        raster = rasterio.open(path_to_image)
        counts = raster.meta['count']

        image_array = []
        for i in range(counts):
            image_array.append(raster.read(i + 1))
            
        nan_mask=[]
        
        for i in range(image_array[0].shape[0]):
            spectrum_data_at_row_i_nan_check = []
            mask=np.zeros(image_array[0].shape[1])
            
            for m in range(counts):
                spectrum_data_at_row_i_nan_check.append(image_array[m][i])

            spectrum_data_at_row_i_nan_check_T = np.transpose(
                np.array(spectrum_data_at_row_i_nan_check))

            nan_ind = np.where(np.isnan(spectrum_data_at_row_i_nan_check_T))[0]

            mask[nan_ind] = 1
            nan_mask.append(mask)

        nan_mask=np.array(nan_mask)
        
        idx=np.where(nan_mask==1)
        input_array[idx]=nanvalue

        return input_array
        
