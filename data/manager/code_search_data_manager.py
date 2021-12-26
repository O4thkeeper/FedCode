from tqdm import tqdm

from data.manager.base.abstract_data_manager import AbstractDataManager


class CodeSearchDataManager(AbstractDataManager):
    def __init__(self, args, model_args, preprocessor, data_type, data_path, batch_size, partition_path=None):
        super(CodeSearchDataManager, self).__init__(args, model_args, data_type, data_path, batch_size, partition_path)
        self.attributes = self.load_attributes(data_path)
        self.preprocessor = preprocessor

    def read_instance_from_h5(self, data_file, index_list=None, desc=""):
        X = dict()
        X['text_a'] = list()
        X['text_b'] = list()
        y = list()
        if index_list is None:
            for idx in tqdm(range(len(data_file["Y"])), desc="Loading data from h5 file." + desc):
                X['text_a'].append(data_file["X"]['text_a'][str(idx)][()].decode("utf-8"))
                X['text_b'].append(data_file["X"]['text_b'][str(idx)][()].decode("utf-8"))
                y.append(data_file["Y"][str(idx)][()].decode("utf-8"))
        else:
            for idx in tqdm(index_list, desc="Loading data from h5 file." + desc):
                X['text_a'].append(data_file["X"]['text_a'][str(idx)][()].decode("utf-8"))
                X['text_b'].append(data_file["X"]['text_b'][str(idx)][()].decode("utf-8"))
                y.append(data_file["Y"][str(idx)][()].decode("utf-8"))
        return {"X": X, "y": y}
