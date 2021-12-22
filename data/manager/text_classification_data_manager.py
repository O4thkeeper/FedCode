from tqdm import tqdm

from data.manager.base.base_data_manager import BaseDataManager


class TextClassificationDataManager(BaseDataManager):
    """Data manager for text classification"""

    def __init__(self, args, model_args, preprocessor, num_workers=1):

        super(TextClassificationDataManager, self).__init__(args, model_args, num_workers)
        self.attributes = self.load_attributes(args.data_file_path)
        self.preprocessor = preprocessor

    def read_instance_from_h5(self, data_file, index_list, desc=""):
        X = list()
        y = list()
        for idx in tqdm(index_list, desc="Loading data from h5 file." + desc):
            X.append(data_file["X"][str(idx)][()].decode("utf-8"))
            y.append(data_file["Y"][str(idx)][()].decode("utf-8"))
        return {"X": X, "y": y}