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
    def __init__(self, split=(3, 2016, 12, 2017, 1, 2018, 2, 2018), df=None, autoexclude=False, target='NumberOfSales', dataset='mean_var_pre_imputed_per_day.csv'):

        self.split = split

        self.target = target

        self.xtr = None
        self.ytr = None
        self.xts = None
        self.yts = None

        self.filter = None

        self.dataset = dataset
        self.frame = df

        if self.frame is None:
            self.frame = ds_handler.read_dataset(name=self.dataset)
        else:
            if autoexclude:
                print("Important: autoexclude is ON, program might crash when accessing non existing attributes")

        if autoexclude:
            self.excluded = default_excluded
        else:
            self.excluded = []

    def exclude_list(self, list):
        for e in list:
            if e != 'Date':
                self.excluded.append(e)
        return self

    def exclude(self, *attr):
        for e in attr:
            if e != 'Date':
                self.excluded.append(e)
        return self

    # Apply before BUILD
    def only(self, columns):
        columns.append(self.target)
        if 'Date' not in columns:
            columns.append('Date')
            self.frame = self.frame[columns]

        return self

    # Apply after BUILD
    def random_sampling(self, percentage):

        print("Random sampling")

        size = round(self.xtr.shape[0] * percentage)
        idxs = []
        # Bagging with bootstraping
        for i in range(0, size):
            idxs.append(random.randint(0, self.xtr.shape[0]-1))

        sample_xtr = [self.xtr[i] for i in idxs]
        sample_ytr = [self.ytr[i] for i in idxs]

        print('Done.\nTraining set has %s samples\nTesting set has %s samples' % (len(self.ytr), len(self.yts)))

        return np.array(sample_xtr), np.array(sample_ytr)



    def build(self):

        if self.filter is not None:
            self.frame = self.frame[self.filter]

        if 'Date' in self.excluded:
            self.excluded.remove('Date')

        try:
            self.frame = self.frame.drop(columns=self.excluded)
        except Exception as e:
            print("Error: trying to access a non existing attribute.. Did you turn off autoexclude?")
            return None

        print("Building training and testing set")
        print("Excluded attributes = %s" % self.excluded)

        self.xtr = ds_util.get_frame_in_range(self.frame, self.split[0], self.split[1], self.split[2], self.split[3])
        self.xts = ds_util.get_frame_in_range(self.frame, self.split[4], self.split[5], self.split[6], self.split[7])

        self.xtr = ds_handler.to_numpy(self.xtr.drop(columns=['Date', self.target]))
        self.xts = ds_handler.to_numpy(self.xts.drop(columns=['Date', self.target]))

        self.ytr = ds_util.get_frame_in_range(self.frame, self.split[0], self.split[1], self.split[2], self.split[3])
        self.ytr = ds_handler.to_numpy(self.ytr[[self.target]])

        self.yts = ds_util.get_frame_in_range(self.frame, self.split[4], self.split[5], self.split[6], self.split[7])
        self.yts = ds_handler.to_numpy(self.yts[[self.target]])

        print('Done.\nTraining set has %s samples\nTesting set has %s samples'
              % (len(self.ytr), len(self.yts)))

        return self
