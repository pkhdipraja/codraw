import os


class PATH(object):
    """
    Configuration object for dataset paths.
    """
    def __init__(self):
        # Datasets path
        self.CODRAW_JSON_PATH = '/home/users/pkahardipraja/project/codraw_project/CoDraw/dataset/CoDraw_1_0.json'

    def init_path(self):
        self.RESULTS_PATH = './results'
        self.CKPTS_PATH = './ckpts'

        if 'ckpts' not in os.listdir('./'):
            os.mkdir(self.CKPTS_PATH)

        if 'results' not in os.listdir('./'):
            os.mkdir(self.RESULTS_PATH)
