import dataset.dataset as ds_handler
import dataset.utility as ds_util
import random
import numpy as np
import pandas as pd
import preprocessing.preprocessing_utils as preprocessing

# Note that the set builder will auto exclude the target, there is no need to insert it here

class SetBuilder:

    # Can pass a different training/testing split tuple
    # If default is set to false no attributes are excluded
    # Provide only one target, default is nr of sales
    # split might be an array of array containing multiple splits like:
    # split [
    # [(train tuple interval one), (training tuple interval two)],
    # [(test tuple interval one), (test tuple interval two)]
    # ]

    def __init__(self, split=(3, 2016, 12, 2017, 1, 2018, 2, 2018), df=None, autoexclude=False, target='NumberOfSales', dataset='best_for_customers.csv'):

        self.default_excluded = ['StoreID', 'Date', 'IsOpen', 'Region', 'CloudCover', 'Max_Sea_Level_PressurehPa',
            'WindDirDegrees', 'Max_Dew_PointC', 'Mean_Sea_Level_PressurehPa', 'Min_Sea_Level_PressurehPa', 'Day']

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
            self.excluded = self.default_excluded
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

        print('Random sampling returned %s samples' % len(sample_ytr))

        return np.array(sample_xtr), np.array(sample_ytr)

    def get_training(self, interval):
        xtr = ds_util.get_frame_in_range(self.frame, interval[0], interval[1], interval[2], interval[3])
        ytr = xtr.copy()
        xtr = xtr.drop(columns=['Date', self.target])
        ytr = ytr[[self.target]]

        return xtr, ytr

    def get_testing(self, interval):
        xts = ds_util.get_frame_in_range(self.frame, interval[0], interval[1], interval[2], interval[3])
        yts = xts.copy()
        xts = xts.drop(columns=['Date', self.target])
        yts = yts[[self.target]]

        return xts, yts

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

        if type(self.split) is tuple:

            print("Split strategy = sequential")

            self.xtr, self.ytr = self.get_training((self.split[0], self.split[1], self.split[2], self.split[3]))
            self.xts, self.yts = self.get_training((self.split[4], self.split[5], self.split[6], self.split[7]))

        else:

            print("Split strategy = interleaved")

            training_tuples = self.split[0]
            testing_tuples = self.split[1]

            for interval in training_tuples:

                if self.xtr is None:
                    self.xtr, self.ytr = self.get_training(interval)
                else:
                    delta_xtr, delta_ytr = self.get_training(interval)
                    self.xtr = pd.concat([self.xtr, delta_xtr])
                    self.ytr = pd.concat([self.ytr, delta_ytr])

            for interval in testing_tuples:
                if self.xts is None:
                    self.xts, self.yts = self.get_testing(interval)
                else:
                    delta_xts, delta_yts = self.get_testing(interval)
                    self.xts = pd.concat([self.xts, delta_xts])
                    self.yts = pd.concat([self.yts, delta_yts])

        self.xtr = ds_handler.to_numpy(self.xtr)
        self.ytr = ds_handler.to_numpy(self.ytr)
        self.xts = ds_handler.to_numpy(self.xts)
        self.yts = ds_handler.to_numpy(self.yts)

        print('Setbuilder: #xtr = %s, #ytr = %s, #xts = %s, #yts = %s' % (len(self.xtr), len(self.ytr), len(self.xts), len(self.yts)))

        return self
