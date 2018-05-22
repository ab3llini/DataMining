from dataexploration.graphs.graphs import monthlysalesplot
from dataset.dataset import read_imputed_onehot_dataset
from dataexploration.graphs.graphs import monthlycustomersplot

monthlysalesplot(read_imputed_onehot_dataset(), 3, 2016, 2, 2018)
monthlycustomersplot(read_imputed_onehot_dataset(), 3, 2016, 2, 2018)