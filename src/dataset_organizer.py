import os
import shutil


def organize(
        data_path='/home/yilmaz/school/Scene-Classifier/dataset/Images/',
        files_path='/home/yilmaz/school/Scene-Classifier/dataset/', train_file='train.txt', test_file='test.txt'):
    # Parse the files
    train_lines = open(os.path.join(files_path, train_file)).read().splitlines()
    test_lines = open(os.path.join(files_path, test_file)).read().splitlines()

    # Remove the existing data if exist
    if os.path.exists('/tmp/b21327694_dataset'):
        shutil.rmtree('/tmp/b21327694_dataset')

    # Copy the train data
    file_copy(data_path, '/tmp/b21327694_dataset/train/', train_lines)

    # Copy the test data
    file_copy(data_path, '/tmp/b21327694_dataset/test/', test_lines)


def file_copy(source_base_path, destination_base_path, file_list):

    for file_name in file_list:

        # Check if the directory exist
        if not os.path.isdir(os.path.dirname(os.path.join(destination_base_path, file_name))):
            os.makedirs(os.path.dirname(os.path.join(destination_base_path, file_name)))

        # Copy the file
        shutil.copyfile(
            src=os.path.join(source_base_path, file_name),
            dst=os.path.join(destination_base_path, file_name)
        )
