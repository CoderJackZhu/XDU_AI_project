from utils import create_data_lists

if __name__ == '__main__':
    create_data_lists(train_folders=['./data/Train','./data/BSD100'],
                      test_folders=['./data/Set5',
                                    './data/Set14'],
                      min_size=100,
                      output_folder='./data/')
