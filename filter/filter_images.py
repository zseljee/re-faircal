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



def read_files(base_dir:str, filename_list:list, unrecognized_faces_filename:str, unique_filenames:bool= False) -> any:
    """
        Function that uses one file of unrecognized faces to check the pairs of images in the other file.

        input:
            base_dir: (string) base directory for the data relative to the python file
            file_name_list: (list) of directory names (strings) to itterate over and read the pair text file from
            unrecognized_faces_filename: (string) of relative path and filename of a text file with names of pictures with unrecognized faces

        output:
            info_dict: (dictionary of lists), contains the information of the filtered faces. e.g.:
                id1: unique id of the first face
                id2: unique id of the second face (can be the same as id1)
                filepath1: path to the image + image name of the first image
                filepath2: path to the image + image name of the second image
                label: label if the images are from the same person or not (1: True, 0: False)
            
            optional:
                set_filenames: (set) of filepaths + filename of all the filtered faces. To aid in the creation of embeddings
    """
    # create list of filenames of unrecognized faces for easy comparison
    with open(unrecognized_faces_filename) as unrecog_faces_file:
        unrecognized_faces_names_lst = [line.split('/')[-1][:-5] for line in unrecog_faces_file.readlines()] # remove ".jpg/n"
    
    info_dict = {'id1': [],
                'id2': [],
                'filepath1': [], 
                'filepath2':[], 
                'label': []}
    if unique_filenames:
        set_filenames = set()

    for filename in filename_list:
        with open(base_dir + '/' + filename, 'r') as f:
            for line in f.readlines():

                striped_line = [item.strip() for item in line.split()]

                # handle the case where the same person is used
                if len(striped_line) == 3:
                    facefile_names = [f'{striped_line[0]}_{int(striped_line[1]):04}', 
                                      f'{striped_line[0]}_{int(striped_line[2]):04}']
                    label = 1
                    id1 = id2 = striped_line[0]
                    face1_filepath = base_dir[:-4] + 'data/' + filename.split('/')[0] + striped_line[0] + facefile_names[0] + '.jpg'
                    face2_filepath = base_dir[:-4] + 'data/' + filename.split('/')[0] + striped_line[0] + facefile_names[1] + '.jpg'

                # handle the case where pictures of two different persons are used
                elif len(striped_line) == 4:
                    facefile_names = [f'{striped_line[0]}_{int(striped_line[1]):04}', 
                                      f'{striped_line[2]}_{int(striped_line[3]):04}']
                    label = 0
                    id1 = striped_line[0]
                    id2 = striped_line[2]
                    face1_filepath = base_dir[:-4] + 'data/' + filename.split('/')[0] + striped_line[0] + facefile_names[0] + '.jpg'
                    face2_filepath = base_dir[:-4] + 'data/' + filename.split('/')[0] + striped_line[2] + facefile_names[1] + '.jpg'

                # catch edge cases
                else:
                    raise Exception(f'line: {line} raised an exception!')
                
                # filter the images
                if all(item not in unrecognized_faces_names_lst for item in facefile_names):

                    info_dict['id1'].append(id1)
                    info_dict['id2'].append(id2)
                    info_dict['filepath1'].append(face1_filepath)
                    info_dict['filepath2'].append(face2_filepath)
                    info_dict['label'].append(label)

                    if unique_filenames:
                        set_filenames.add(face1_filepath)
                        set_filenames.add(face2_filepath)

    if unique_filenames:
        return info_dict, set_filenames
    return info_dict


def test_files(info_dict,  set_filenames = None):
    """
        function that tests if certain faces are NOT in the filtered set
        The images are selected from the 'unrecognised-faces.txt' file.

    """
    assert not any('m.0gy3f_0003.jpg' in item for item in set_filenames)
    assert not any('m.08zncx_0003.jpg' in item for item in set_filenames)
    assert not any('m.0r8ntwm_0002.jpg' in item for item in set_filenames)
    assert not any('m.07k59_k_0001.jpg' in item for item in set_filenames)
    assert not any('m.0cy6q8_0001.jpg' in item for item in set_filenames)
    assert not any('m.0cy6q8_0002.jpg' in item for item in set_filenames)
    assert not any('m.026p3wd_0004.jpg' in item for item in set_filenames)
    assert not any('m.044hmr_0003.jpg' in item for item in set_filenames)

def main():
    BASE_DIR = 'data/rfw/txts'

    lst_filename = retreive_text_files(BASE_DIR)
    output = read_files(BASE_DIR, lst_filename, 'data/rfw/txts/unrecognised-faces.txt', True)
    # test_files(*output)

    if len(output) == 2:
        info_dict, set_of_filenames = output
        with open(BASE_DIR + '/unique_picture_links.txt', 'w') as f:
            for item in set_of_filenames:
                f.write(item + '\n')
    
    else:
        info_dict = output[0]
    df = pd.DataFrame({key: pd.Series(val) for key, val in info_dict.items()})
    df.to_csv(BASE_DIR + '/filtered_pairs.csv', index=False)


if __name__ == '__main__':
    main()