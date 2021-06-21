# GcomcTool
This module was developed to make the GCOM-C data processing easy.
It can do the following jobs:
1.Reprojection of the original EQA hdf5 data to the WGS84 lat/lon GeoTIFF data.
2.Query and download GCOM-C data in an automated way.
3.Visualization of the target areas.
4.Clip and merge the input data into continuous images.

Although the output products seem to have good accuracy of reprojection, since it is an unofficial tool made by one student, so the accuracy of reprojection and the other attributes are not guaranteed by the Japan Aerospace Exploration Agency (JAXA) and other official data providers.
Currently, this code supports the tile products that contain most of the level-2 land, atmosphere, and cryosphere datasets. The following products are not supported in this module, but I'm considering including them in the future. *level-1 and level-2 ocean scene products *level-3 EQA and EQR products
