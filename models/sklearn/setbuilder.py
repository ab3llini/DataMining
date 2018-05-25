import dataset.dataset as ds_handler
import dataset.utility as ds_util
import random
import numpy as np
import preprocessing.preprocessing_utils as preprocessing

# Note that the set builder will auto exclude the target, there is no need to insert it here
default_excluded = [
                'StoreID',
                'Date',
                'IsOpen',
                'Region',
                'CloudCover',
                'Max_Sea_Level_PressurehPa',
                'WindDirDegrees',
                'Max_Dew_PointC',
                'Mean_Sea_Level_PressurehPa',
                'Min_Sea_Level_PressurehPa',
                'Day'
            ]

class SetBuilder:

    # Can pass a different training/testing split tuple
    # If default is set to false no attributes are excluded
    # Provide only one target, default is nr of sales
    def __init__(self, split=(3, 2016, 12, 2017, 1, 2018, 2, 2018), default=True, target='NumberOfSales', dataset='mean_var_pre_imputed_per_day.csv'):

        self.split = split

        self.target = target

        self.xtr = None
        self.ytr = None
        self.xts = None
        self.yts = None

        self.dataset = dataset

        self.frame = None

        if default:
            self.excluded = default_excluded
        else:
            self.excluded = []

    def exclude(self, attr):
        self.excluded.append(attr)

        return self

    def random_sampling(self, percentage):

        print("Random sampling")

        size = round(self.xtr.shape[0] * percentage)
        idxs = []
        sample_xtr = []
        sample_ytr = []
        # Bagging with bootstraping
        for i in range(0, size):
            idxs.append(random.randint(0, self.xtr.shape[0]))

        sample_xtr = [self.xtr[i] for i in idxs]
        sample_ytr = [self.ytr[i] for i in idxs]

        self.xtr = np.array(sample_xtr)
        self.ytr = np.array(sample_ytr)

        print('Done.\nTraining set has %s samples\nTesting set has %s samples' % (len(self.ytr), len(self.yts)))

        return self

    def build(self):

        if self.target not in self.excluded:
            self.exclude(self.target)

        print("Building training and testing set")
        print("Excluded attributes = %s" % self.excluded)

        self.frame = ds_handler.read_dataset(name=self.dataset)

        self.xtr = ds_util.get_frame_in_range(self.frame, self.split[0], self.split[1], self.split[2], self.split[3])
        self.xts = ds_util.get_frame_in_range(self.frame, self.split[4], self.split[5], self.split[6], self.split[7])

        self.xtr = ds_handler.to_numpy(self.xtr.drop(columns=self.excluded))
        self.xts = ds_handler.to_numpy(self.xts.drop(columns=self.excluded))

        self.ytr = ds_util.get_frame_in_range(self.frame, self.split[0], self.split[1], self.split[2], self.split[3])
        # ytr = ytr.drop(ytr[ytr.IsOpen == 0].index)
        self.ytr = ds_handler.to_numpy(self.ytr[[self.target]])

        self.yts = ds_util.get_frame_in_range(self.frame, self.split[4], self.split[5], self.split[6], self.split[7])
        # yts = yts.drop(yts[yts.IsOpen == 0].index)
        self.yts = ds_handler.to_numpy(self.yts[[self.target]])

        print('Done.\nTraining set has %s samples\nTesting set has %s samples'
              % (len(self.ytr), len(self.yts)))

        return self
