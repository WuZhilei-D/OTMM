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
