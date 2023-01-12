# Data

This directory contains the data for the BFW and RFW sets.  It also contains the license for both datasets, which should be abided by.  Only research may be done with both datasets, BFW allows education too.

To set up this directory correctly with the code, extract the BFW dataset into a folder called `TBD` and extract RFW into `TBD`.

The Table below was copied from the repository [README](../README.md)
| Dataset | Open-source | URL |
|---------|-------------|-----|
| BFW     | No, register through Google Forms | https://github.com/visionjo/facerec-bias-bfw |
| RFW     | No, mail for research access | http://whdeng.cn/RFW/testing.html |

# BFW

## Data structure

### CSV file

[From](https://github.com/visionjo/facerec-bias-bfw/blob/master/data/README.md#data-structure)
| ID |  fold | p1  | p2  | label  | id1  | id2	| att1  | att2  | vgg16  | resnet50   | senet50   | a1   | a2   | g1   | g2 | e1   | e2   | sphereface   |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0  | 1  |  asian\_females/n000009/0010\_01.jpg | asian\_females/n000009/0043\_01.jpg | 1     | 0   | 0   | asian\_females | asian\_females | 0.820 | 0.703 | 0.679 | AF | AF | F  | F  | A  | A  | 0.393   |
| 1  | 1  | asian\_females/n000009/0010\_01.jpg | asian\_females/n000009/0120\_01.jpg | 1     | 0   | 0   | asian\_females | asian\_females | 0.719 | 0.524 | 0.594 | AF | AF | F  | F  | A  | A  | 0.354  |
| 2  | 1  |  asian\_females/n000009/0010\_01.jpg | asian\_females/n000009/0122\_02.jpg | 1     | 0   | 0   | asian\_females | asian\_females | 0.732 | 0.528 | 0.644  | AF | AF | F  | F  | A  | A  | 0.302  |
| 3 | 1    | asian\_females/n000009/0010\_01.jpg | asian\_females/n000009/0188\_01.jpg | 1     | 0   | 0   | asian\_females | asian\_females | 0.607 | 0.348 | 0.459 | AF | AF | F  | F  | A  | A  | \-0.009 |
| 4 | 1    | asian\_females/n000009/0010\_01.jpg | asian\_females/n000009/0205\_01.jpg | 1     | 0   | 0   | asian\_females | asian\_females | 0.629 | 0.384 | 0.495 | AF | AF | F  | F  | A  | A  | 0.133  |
<br>

* **ID** : index (i.e., row number) of dataframe ([0, *N*], where *N* is pair count).
* **fold** : fold number of five-fold experiment [1, 5].
* **p1**  and **p2** : relative image path of face
* **label** : ground-truth ([0, 1] for non-match and match, respectively)
* **id1** and **id2** : subject ID for faces in pair ([0, *M*], where *M* is number of unique subjects)
* **att1** and **att2** : attributee of subjects in pair.
* **vgg16**, **resnet50**, **senet50**, and **sphereface** : cosine similarity score for respective model.
* **a1** and **a2** : abbreviated attribute tag of subjects in pair [AF, AM, BF, BM, IF, IM, WF, WM].
* **g1** and **g2** : abbreviated gender tag of subjects in pair [F, M].
* **e1** and **e2** : abbreviate ethnicity tag of subjects in pair [A, B, I, W].

### PKL file

`dict[str: list[ dict[str: Any]] ]`
A dictionary mapping paths (such as `asian_males/n006698/0278_01.jpg`) to metrics `box`, `confidence`, `keypoints`.

* **Box** `list[ int ]`, a list of coordinates or `x,y,h,w` for the bounding box of the detected face
* **confident** Confidence score for detected face
* **keypoints** `dict[ str, tuple[float] ]` A dictionary mapping key points `left_eye`, `right_eye`, `nose`, `mouth_left`, `mouth_right` to a tuple of coordinates `x,y`