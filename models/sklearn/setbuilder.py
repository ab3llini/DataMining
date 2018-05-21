import dataset.dataset as ds_handler
import dataset.utility as ds_util
import preprocessing.preprocessing_utils as preprocessing

# Note that the set builder will auto exclude the target, there is no need to insert it here
default_excluded = [
                'Region',
                'Date',
                'WindDirDegrees',
                'Max_Wind_SpeedKm_h',
                'Max_Sea_Level_PressurehPa',
                'Mean_Sea_Level_PressurehPa',
                'Min_Sea_Level_PressurehPa'
            ]

selected_dataset = 'mean_var_pre_imputed.csv'

class SetBuilder:

    # Can pass a different training/testing split tuple
    # If default is set to false no attributes are excluded
    # Provide only one target, default is nr of sales
    def __init__(self, split=(3, 2016, 12, 2017, 1, 2018, 2, 2018), default=True, target='NumberOfSales', avoid_closed=False):

        self.split = split

        self.target = target

        self.xtr = None
        self.ytr = None
        self.xts = None
        self.yts = None

        self.avoid_closed = avoid_closed

        self.frame = None

        if default:
            self.excluded = default_excluded
        else:
            self.excluded = []

    def exclude(self, attr):
        self.excluded.append(attr)

        return self

    def build(self):


        if self.target not in self.excluded:
            self.exclude(self.target)

        print("Building training and testing set")
        print("Excluded attributes = %s" % self.excluded)

        self.frame = ds_handler.read_dataset(name=selected_dataset)

        if self.avoid_closed:
            self.frame = preprocessing.eliminate_IsOpen_zeros(self.frame)

        self.xtr = ds_util.get_frame_in_range(self.frame, self.split[0], self.split[1], self.split[2], self.split[3])
        # xtr = xtr.drop(xtr[xtr.IsOpen == 0].index)
        self.xts = ds_util.get_frame_in_range(self.frame, self.split[4], self.split[5], self.split[6], self.split[7])
        # xts = xts.drop(xts[xts.IsOpen == 0].index)

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
