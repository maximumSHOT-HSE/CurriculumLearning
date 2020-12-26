import shutil
from time import sleep
import os


def without_two_greatest(checkpoints):
    return sorted(checkpoints, key=lambda x: int(x.split('-')[1]))[:-2]


if __name__ == '__main__':
    main_folder = '/home/aomelchenko/Bachelor-s-Degree/Logs'
    while True:
        for folder in os.listdir(main_folder):
            current_folder = main_folder + os.path.sep + folder
            to_delete = without_two_greatest(os.listdir(current_folder))
            for checkpoint_folder in to_delete:
                folder_with_logs = current_folder + os.path.sep + checkpoint_folder

                files_to_del = ['optimizer.pt', 'pytorch_model.bin']
                full_paths = [folder_with_logs + os.path.sep + file for file in files_to_del]

                for file in full_paths:
                    if os.path.exists(file):
                        os.remove(file)

        sleep(60 * 60)
