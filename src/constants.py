import os

from preprocess.constants import (
    DATA_FOLDER,
    EMBEDDING_FOLDER,
    EMBEDDING_FORMAT,
    OUTPUT_CSV as PREPROCESS_OUTPUT_CSV,
)

EXPERIMENT_FOLDER = os.path.abspath( os.path.join( os.path.dirname(__file__), '../', 'experiments/' ) )

AVAILABLE_DATASETS = {
    'rfw': {
        'data_root': os.path.join( DATA_FOLDER, 'rfw' ),
        'csv': PREPROCESS_OUTPUT_CSV['rfw'],
        'embeddings': os.path.join( EMBEDDING_FOLDER['rfw'], EMBEDDING_FORMAT ),
        'sensitive_attributes': {
            'ethnicity': {
                'cols': ['ethnicity', 'ethnicity'],
                'values': ['African', 'Asian', 'Caucasian', 'Indian']}
            }
    },
    'bfw': {
        'data_root': os.path.join( DATA_FOLDER, 'bfw' ),
        'csv': PREPROCESS_OUTPUT_CSV['bfw'],
        'embeddings': os.path.join( EMBEDDING_FOLDER['bfw'], EMBEDDING_FORMAT),
        'sensitive_attributes': {
            'ethnicity': {
                'cols': ['e1', 'e2'],
                'values': ['B', 'A', 'W', 'I'],
            },
            'gender': {
                'cols': ['g1', 'g2'],
                'values': ['F', 'M'],
            }
        }
    }
}
