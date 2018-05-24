from dataexploration.graphs.graphs import monthlysalesplot
from dataset.dataset import read_imputed_onehot_dataset
from dataexploration.graphs.graphs import monthlycustomersplot
from dataexploration.graphs.graphs import opendaybeforegeneralplot
from dataexploration.graphs.graphs import opendaybeforeonweekplot

# monthlysalesplot(read_imputed_onehot_dataset(), 3, 2016, 2, 2018)
# monthlycustomersplot(read_imputed_onehot_dataset(), 3, 2016, 2, 2018)
opendaybeforegeneralplot(read_imputed_onehot_dataset(), 1111)
opendaybeforeonweekplot(read_imputed_onehot_dataset(), 1111)
