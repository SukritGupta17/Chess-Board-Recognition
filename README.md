# Chess-Board-Recognition

This project highlights approaches taken to process
an image of a chessboard and identify the configuration of the
board using **computer vision techniques**. Although, the use of a
chessboard detection for camera calibration is a classic vision
problem, existing techniques on piece recognition work under
a controlled environment. The procedures are customized for
a chosen colored chessboard and a particular set of pieces.
The methods used in this project supplements existing
research by using **clustering** to segment the chessboard and
pieces irrespective of color schemes. For piece recognition, the
method introduces a novel approach of using a **R-CNN** to train
a robust classifier to work on different kinds of chessboard
pieces. The method performs better on different kinds of pieces
as compared to a **SIFT** based classifier. If extended, this work
could be useful in recording moves and training chess AI for
predicting the best possible move for a particular chessboard
configuration.

**Approach Stack**:
![Approach](https://github.com/SukritGupta17/Chess-Board-Recognition/blob/master/Results/Approach.png)

**Clusters Obtained**:
![cluster1](https://github.com/SukritGupta17/Chess-Board-Recognition/blob/master/Results/cluster1.png)
![cluster2](https://github.com/SukritGupta17/Chess-Board-Recognition/blob/master/Results/cluster2.png)

**Detected Lines**
![detected lines](https://github.com/SukritGupta17/Chess-Board-Recognition/blob/master/Results/detected%20lines.png)

**Pieces extracted**
![extracted pieces](https://github.com/SukritGupta17/Chess-Board-Recognition/blob/master/Results/pieces.png)

**Recognition**:
![final recognition](https://github.com/SukritGupta17/Chess-Board-Recognition/blob/master/Results/deteected.png)


_NOTE: For more details refer to the report._
