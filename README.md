# GcomcTool
This module was developed to make the GCOM-C data processing easy.

Installation
```
pip install git+https://github.com/Shima-shoki/GcomcTool
```

It can do the following jobs on the GCOM-C products:<br>
1. Reprojection of the original EQA hdf5 data to the WGS84 lat/lon GeoTIFF data (level-2 tile products, level-2 scene products, and level-1B TOA products).<br>
2. Query and download GCOM-C data in an automated way (level-2 tile products, and level-3 global products).<br>
3. Visualization of the target areas (level-2 tile products).<br>
4. Clip and merge the input data into continuous images (level-2 tile products).<br>
5. Process the original data to an easily accessible tiff format (all the products).<br>

Although the output products seem to have good accuracy of reprojection, this is only an unofficial tool made by one student. So the accuracy of reprojection and the other attributes are not guaranteed by the Japan Aerospace Exploration Agency (JAXA) and other official data providers.

Any kind of help will be very much appriciated!
