# Pose inversion guided by multi-view contours for high-precision orthodontic tooth movement monitoring

## preprocessing
Tooth Coordinate System Establishment - Please refer to ToothCoordinate.py

## Usage
1） Put the intraoral iamges and its contour extraction results of VOC-format into the folder in ./label.

2） Put the teeth model into the folder in ./mesh,  each tooth named according to its FDI number.

3） To run orthodontic tooth movement monitoring demo: "python main.py"

## Overview
Overview of the remote orthodontic tooth movement monitoring framework using pose inversion guided by multi-view contours. a Input data: Initial intraoral scan (first visit) and multi-view intraoral images captured during treatment. b Parameter optimization: Alternates between matching corresponding points and minimizing the loss between projected model silhouettes (using current camera parameters) and actual image contours. This process inverts both camera and 6-DoF tooth pose parameters. c Output: Applying the optimized pose parameters to the initial scan model accurately reflects current tooth positions, enabling quantitative treatment outcome evaluation.
<p align="center">
    <img src=".\log\assets\teeth_movement_monitoring_framework.jpg" alt="teeth monitoring framework" width="800"/>
</p>

## Reconstruction result
<p align="center">
    <img src=".\log\assets\monitoring_results.jpg" alt="teeth monitoring results" width="800"/>
</p>


## Requirements
- cycpd==0.25
- h5py==3.1.0
- keras==2.6.0
- numpy==1.19.5
- open3d==0.16.0
- opencv-contrib-python==4.6.0.66
- opencv-python==4.6.0.66
- pandas==1.4.3
- ray==2.0.1
- scikit-image==0.19.3
- scikit-learn==1.1.1
- scikit-optimize==0.9.0
- scipy==1.8.1
- Shapely==1.8.5.post1
- tensorflow-addons==0.16.1
- tensorflow-gpu==2.6.0
- trimesh==3.15.5
