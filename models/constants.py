import os

DATA_FOLDER = os.path.abspath( './data/' )
DATA_ROOT = {
    'rfw': os.path.join( DATA_FOLDER, 'rfw/' ),
    'bfw': os.path.join( DATA_FOLDER, 'bfw/' ),
}
RAW_IMAGE_ROOT = {
    'rfw': os.path.join( DATA_ROOT['rfw'], 'data/' ),
    'bfw': os.path.join( DATA_ROOT['bfw'], 'uncropped-face-samples/' ),
}
CROPPED_IMAGE_ROOT = {
    'rfw': os.path.join( DATA_ROOT['rfw'], 'data_cropped/' ),
    'bfw': os.path.join( DATA_ROOT['bfw'], 'data_cropped/' ),
}

BFW_RAW_CSV = os.path.join( DATA_ROOT['bfw'], 'bfw-v0.1.5-datatable.csv' )

EMBEDDING_FORMAT = "{}_embeddings.pickle"

EMBEDDING_FOLDER = {
    'rfw': DATA_ROOT['rfw'],
    'bfw': DATA_ROOT['bfw'],
}

OUTPUT_CSV = {
    'rfw': os.path.join( DATA_ROOT['rfw'], 'rfw.csv' ),
    'bfw': os.path.join( DATA_ROOT['bfw'], 'bfw.csv' ),
}
