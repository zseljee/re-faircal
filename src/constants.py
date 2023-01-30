import os

EXPERIMENT_FOLDER = os.path.abspath( './experiments' )

DATA_FOLDER = os.path.abspath( './data/' )

AVAILABLE_DATASETS = {
    'rfw': {
        'data_root': os.path.join( DATA_FOLDER, 'rfw' ),
        'csv': os.path.join( DATA_FOLDER, 'rfw', 'rfw.csv' ),
        'embeddings': os.path.join( DATA_FOLDER, 'rfw', '{}_embeddings.pickle' ),
        'sensitive_attributes': { 
            'ethnicity': {
                'cols': ['ethnicity', 'ethnicity'], 
                'values': ['African', 'Asian', 'Caucasian', 'Indian']}
            }
    },
    'bfw': {
        'data_root': os.path.join( DATA_FOLDER, 'bfw' ),
        'csv': os.path.join( DATA_FOLDER, 'bfw', 'bfw.csv' ),
        'embeddings': os.path.join( DATA_FOLDER, 'bfw', '{}_embeddings.pickle' ),
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