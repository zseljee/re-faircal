# Data

This directory contains the data for the BFW and RFW sets.  It also contains the license for both datasets, which should be abided by.  Only research may be done with both datasets, BFW allows education too.

To set up this directory correctly with the code, extract the BFW dataset into a folder called `TBD` and extract RFW into `TBD`.

The Table below was copied from the repository [README](../README.md)
| Dataset | Open-source | URL |
|---------|-------------|-----|
| BFW     | No, register through Google Forms | https://github.com/visionjo/facerec-bias-bfw |
| RFW     | No, mail for research access | http://whdeng.cn/RFW/testing.html |

## RFW described

If you download the `test.tar.gz` file through the link you receive after you request access you will see that contains two folders.  The first folder is called `data` and contains all images of the four different testing race subsets.  The second folder is called `txts` and contains all necessary metadata about using the images for training.

To get back to the first folder `data`; It contains four folders four the different subsets.  Each of these subsets then contains many folders: one for each different individual in the dataset.  These folders contain from X to Y different images in which the face is very recognisable.

The second folder `txts` is necessary to obtain the cross validation folds for the pairs.  It also contains four folders for the same different subsets.  This time, the folders contain four files with information about the data folder.  The website for the RFW dataset describes these files as:

* `<set>_images.txt`: The testing image list and label, e.g. `m.0xnkj_0002.jpg  0`
* `<set>_lmk.txt`: Estimated 5 facial landmarks on the provided loosely cropped faces.
* `<set>_pairs.txt`: 10 disjoint splits of image pairs are given, and each contains 300 positive pairs and 300 negative pairs similar to LFW.
* `<set>_people.txt`: The overlapped identities between RFW and MS-Celeb-1M and the number of images per identity.

Note that the landmarks are actually 9 columns.  I expect the last 5 to be the actual landmarks.  In that case I'm not sure what the first four would be.  The reason I think this is that the second and fourth columns have a fixed value of `180.0` for all images.

This is a more detailed explanation of the files:

* `<set>_images.txt`: A file that creates a mapping from filename to ID for a person (within a data subset).
* `<set>_lmk.txt`: 4 Unknown columns. And estimated 5 facial landmarks on the provided loosely cropped faces.
* `<set>_pairs.txt`: File providing positive and negative pairs including how to separate them into 10 fold cross-validation.  See description below.
* `<set>_people.txt`: For people that also occur in the MS-Celeb-1M dataset, this lists the number of images in the RFW dataset.

So the most important part to understand is the structure of `<set>_pairs.txt`:

```txt
person_ID (folder name) <tab> image_1_number (int) <tab> image_2_number (int)

...
person_ID (folder) <tab> image_1_number (int) <tab> image_2_number (int)
person_ID_1 (folder) <tab> image_number (int) <tab> person_ID_2 (folder) <tab> image_number (int)
...
person_ID_1 (folder) <tab> image_number (int) <tab> person_ID_2 (folder) <tab> image_number (int)
person_ID (folder) <tab> image_1_number (int) <tab> image_2_number (int)
...
```

What this means is that it first contains 300 pairs of positive examples, identical `person_ID` for both image_numbers, and then 300 negative examples, which have a different `person_ID`.  This list of 600 is then repeated with different data for the folds, but just appended on the previous data.

Finally, it should be noted that there are 13 pairs of identical IDs that are shared between different subgroups.  The person ID does not uniquely identify a person without their race.

## BFW data structure

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
