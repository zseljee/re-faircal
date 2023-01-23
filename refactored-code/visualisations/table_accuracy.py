import numpy as np
import pandas as pd

def table_accuracy():
    columnNames = [
        ('RFW', 'FaceNet (VGGFace2)', 'TPR @ 0.1\% FPR', 'Salvador'),
        ('RFW', 'FaceNet (VGGFace2)', 'TPR @ 0.1\% FPR', 'FACT-AI'),

        ('RFW', 'FaceNet (VGGFace2)', 'TPR @ 1.0\% FPR', 'Salvador'),
        ('RFW', 'FaceNet (VGGFace2)', 'TPR @ 1.0\% FPR', 'FACT-AI'),

        ('RFW', 'FaceNet (WebFace)', 'TPR @ 0.1\% FPR', 'Salvador'),
        ('RFW', 'FaceNet (WebFace)', 'TPR @ 0.1\% FPR', 'FACT-AI'),

        ('RFW', 'FaceNet (WebFace)', 'TPR @ 1.0\% FPR', 'Salvador'),
        ('RFW', 'FaceNet (WebFace)', 'TPR @ 1.0\% FPR', 'FACT-AI'),


        ('BFW', 'FaceNet (WebFace)', 'TPR @ 0.1\% FPR', 'Salvador'),
        ('BFW', 'FaceNet (WebFace)', 'TPR @ 0.1\% FPR', 'FACT-AI'),

        ('BFW', 'FaceNet (WebFace)', 'TPR @ 1.0\% FPR', 'Salvador'),
        ('BFW', 'FaceNet (WebFace)', 'TPR @ 1.0\% FPR', 'FACT-AI'),

        ('BFW', 'ArcFace', 'TPR @ 0.1\% FPR', 'Salvador'),
        ('BFW', 'ArcFace', 'TPR @ 0.1\% FPR', 'FACT-AI'),

        ('BFW', 'ArcFace', 'TPR @ 1.0\% FPR', 'Salvador'),
        ('BFW', 'ArcFace', 'TPR @ 1.0\% FPR', 'FACT-AI'),
    ]
    columnIndex = pd.MultiIndex.from_tuples(columnNames, names=["Dataset", "Feature", "Metric", "By"])

    data_salvador = np.ones((3,8))#np.array([[18.42, 34.88, ]]) # 3,8
    data_factai = np.zeros((3, 8))

    data = np.zeros((3,16))
    data[:,::2] = data_salvador
    data[:,1::2] = data_factai

    df = pd.DataFrame(data, columns=columnIndex, index=['Baseline', 'FairCal', 'Oracle'])

    print(df.style.to_latex())

table_accuracy()
