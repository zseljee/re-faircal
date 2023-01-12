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
