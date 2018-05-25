from dataset.dataset import read_imputed_onehot_dataset
import dataexploration.graphs.graphs as gr

df=read_imputed_onehot_dataset()
# gr.monthlyplot(read_imputed_onehot_dataset(), 3, 2016, 2, 2018)
# gr.monthlyplot(read_imputed_onehot_dataset(), 3, 2016, 2, 2018, "NumberOfCustomers")
# gr.opendaybeforegeneralplot(read_imputed_onehot_dataset(), 1000)
# gr.opendaybeforeonweekplot(read_imputed_onehot_dataset(), 1000)
# gr.competitorplot(read_imputed_onehot_dataset())
# gr.competitorplot(read_imputed_onehot_dataset(), "NumberOfCustomers")
# gr.frequencypershop(df, 0, daily=True, target="NumberOfCustomers")
# gr.scattertargets(df, "Region")
gr.availabilityplot(df)