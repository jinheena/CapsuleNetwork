# copy and modify from https://github.com/IBM/pytorch-seq2seq/blob/master/seq2seq/util/checkpoint.py

from __future__ import print_function
import os
import time
import shutil

import torch

class Checkpoint(object):
    """
    The Checkpoint class manages the saving and loading of a model during training. It allows training to be suspended
    and resumed at a later time (e.g. when running on a cluster using sequential jobs).
    To make a checkpoint, initialize a Checkpoint object with the following args; then call that object's save() method
    to write parameters to disk.
    Args:
        model : model being trained
        optimizer (Optimizer): stores the state of the optimizer
        epoch (int): current epoch (an epoch is a loop through the full training data)
    Attributes:
        CHECKPOINT_DIR_NAME (str): name of the checkpoint directory
        TRAINER_STATE_NAME (str): name of the file storing trainer states
        MODEL_NAME (str): name of the file storing model
    """

    CHECKPOINT_DIR_NAME = 'checkpoints'
    TRAINER_STATE_NAME = 'trainer_states.pt'
    MODEL_NAME = 'model.pt'

    def __init__(self, model, optimizer, epoch, path=None):
        self.model = model
        self.optimizer = optimizer
        self.epoch = epoch
        self._path = path

    @property
    def path(self):
        if self._path is None:
            raise LookupError("The checkpoint has not been saved.")
        return self._path

    def save(self, experiment_dir):
        """
        Saves the current model and related training parameters into a subdirectory of the checkpoint directory.
        The name of the subdirectory is the current local time in Y_M_D_H_M_S format.
        Args:
            experiment_dir (str): path to the experiment root directory
        Returns:
             str: path to the saved checkpoint subdirectory
        """
        date_time = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())

        self._path = os.path.join(experiment_dir, self.CHECKPOINT_DIR_NAME, date_time)
        path = self._path

        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path)
        torch.save({'epoch': self.epoch,
                    'optimizer': self.optimizer
                   },
                   os.path.join(path, self.TRAINER_STATE_NAME))
        torch.save(self.model, os.path.join(path, self.MODEL_NAME))

        return path

    @classmethod
    def load(cls, path):
        """
        Loads a Checkpoint object that was previously saved to disk.
        Args:
            path (str): path to the checkpoint subdirectory
        Returns:
            checkpoint (Checkpoint): checkpoint object with fields copied from those stored on disk
        """
        if torch.cuda.is_available():
            resume_checkpoint = torch.load(os.path.join(path, cls.TRAINER_STATE_NAME))
            model = torch.load(os.path.join(path, cls.MODEL_NAME))
        else:
            resume_checkpoint = torch.load(os.path.join(path, cls.TRAINER_STATE_NAME), map_location=lambda storage, loc: storage)
            model = torch.load(os.path.join(path, cls.MODEL_NAME), map_location=lambda storage, loc: storage)

        optimizer = resume_checkpoint['optimizer']
        return Checkpoint(model=model,
                          optimizer=optimizer,
                          epoch=resume_checkpoint['epoch'],
                          path=path)

    @classmethod
    def get_latest_checkpoint(cls, experiment_path):
        """
        Given the path to an experiment directory, returns the path to the last saved checkpoint's subdirectory.
        Precondition: at least one checkpoint has been made (i.e., latest checkpoint subdirectory exists).
        Args:
            experiment_path (str): path to the experiment directory
        Returns:
             str: path to the last saved checkpoint's subdirectory
        """
        checkpoints_path = os.path.join(experiment_path, cls.CHECKPOINT_DIR_NAME)
        all_times = sorted(os.listdir(checkpoints_path), reverse=True)
        return os.path.join(checkpoints_path, all_times[0])