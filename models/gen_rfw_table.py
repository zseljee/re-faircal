import os
import pandas as pd


def retreive_text_files(base_dir:str) -> list:
    """
        A function that searches for the text files in the data that
        contain the pair information between the pictures
        It assumes that the pairs text is in a structure like: 'European/European_pairs.txt'

        input:
            base_dir = (string) the base directory of the data

        output:
            file_name_lst = (list of strings) strings of the folder and text file name
    """
    file_name_lst = []
    walker = os.walk(base_dir)
    _ = next(walker)

    for (root, _, files) in walker:
        if any('pair' in name for name in files):
            folder_name = root.split('/')[-1]
            pair_filename = folder_name + '/' + folder_name + '_pairs.txt'
            file_name_lst.append(pair_filename)

    return file_name_lst



def gen_df(base_dir:str) -> pd.DataFrame:
    """
        Function that uses one file of unrecognized faces to check the pairs of images in the other file.

        input:
            base_dir: (string) base directory of txts
            file_name_list: (list) of directory names (strings) to itterate over and read the pair text file from

        output:
            df: pd.DataFrame, contains the information of the image pairs. contains columns:
                id1: unique id of the left face
                id2: unique id of the right face (same as id1 for positive samples)
                path1: path to the image + image name of the left image, composed of ethnicity, id and num
                path2: path to the image + image name of the right image, composed of ethnicity, id and num
                same: same if the images are from the same person or not (1: True, 0: False)
                fold: the fold of the image pair (1, ..., 10)
                ethnicity: ethnicity of both images (same ethnicity of each pair)
                num1: indicator of left image used (a positive integer)
                num2: indicator of right image used (a positive integer)
    """

    filename_list = retreive_text_files(base_dir)

    info_dict = {'id1': [],
                'id2': [],
                'path1': [],
                'path2':[],
                'same': [],
                'fold': [],
                'ethnicity': [],
                'num1': [],
                'num2': [],
                'pair': []}

    in_current_fold = False

    for filename in filename_list:
        fold_nr = 0
        ethnicity = filename.split('/')[0]

        with open(base_dir + '/' + filename, 'r') as f:
            for line in f.readlines():

                striped_line = [item.strip() for item in line.split()]

                # handle the case where the same person is used
                if len(striped_line) == 3:

                    # check the folds of the images
                    if not in_current_fold:
                        fold_nr += 1
                        in_current_fold = True

                    facefile_names = [f'{striped_line[0]}_{int(striped_line[1]):04}',
                                      f'{striped_line[0]}_{int(striped_line[2]):04}']

                    same = 1
                    pair = 'Genuine'
                    id1 = id2 = striped_line[0]
                    img_id1, img_id2 = striped_line[1], striped_line[2]
                    face1_filepath = ethnicity + '/' + striped_line[0] + '/' + facefile_names[0] + '.jpg'
                    face2_filepath = ethnicity + '/' + striped_line[0] + '/' + facefile_names[1] + '.jpg'

                # handle the case where pictures of two different persons are used
                elif len(striped_line) == 4:
                    in_current_fold = False
                    facefile_names = [f'{striped_line[0]}_{int(striped_line[1]):04}',
                                      f'{striped_line[2]}_{int(striped_line[3]):04}']
                    same = 0
                    pair = 'Imposter'
                    id1 = striped_line[0]
                    id2 = striped_line[2]
                    img_id1, img_id2 = striped_line[1], striped_line[3]
                    face1_filepath = ethnicity + '/' + striped_line[0] + '/' + facefile_names[0] + '.jpg'
                    face2_filepath = ethnicity + '/' + striped_line[2] + '/' + facefile_names[1] + '.jpg'

                # catch edge cases
                else:
                    raise Exception(f'line: {line} raised an exception!')

                info_dict['id1'].append(id1)
                info_dict['id2'].append(id2)
                info_dict['path1'].append(face1_filepath)
                info_dict['path2'].append(face2_filepath)
                info_dict['same'].append(same)
                info_dict['pair'].append(pair)
                info_dict['fold'].append(fold_nr)
                info_dict['ethnicity'].append(ethnicity)
                info_dict['num1'].append(img_id1)
                info_dict['num2'].append(img_id2)

    df = pd.DataFrame(data=info_dict)

    return df
