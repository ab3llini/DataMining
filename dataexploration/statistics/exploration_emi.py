import dataset.dataset as ds
import dataexploration.graphs.graphs as gr

df_ni= ds.read_dataset()
df=ds.read_imputed_onehot_dataset()
# gr.monthlyplot(df, 3, 2016, 2, 2018)
# gr.monthlyplot(df, 3, 2016, 2, 2018, "NumberOfCustomers")
# gr.opendaybeforegeneralplot(df, 1000)
# gr.opendaybeforeonweekplot(df, 1000)
# gr.competitorplot(df)
# gr.competitorplot(df, "NumberOfCustomers")
# gr.frequencypershop(df[df["StoreType_Hyper Market"] == 1], 0, target="NumberOfCustomers", region=True)
# gr.scattertargets(df, "Region")
# gr.availabilityplot(df)
# gr.frequencypershop(df, storeID=0, target="NumberOfCustomers", events=True)
# gr.barplot(df=df_ni, x="StoreType", y=hue="Region")
# gr.frequencystatplot(df)
# gr.monthlyplot(df, storetype=True, target="NumberOfCustomers")
gr.monthlyplot(df, regions=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], target="NumberOfCustomers")