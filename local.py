import dataset.dataset as ds_handler
import dataset.utility as util
import preprocessing.preprocessing_utils as prep
import pandas as p
import preprocessing.imputation as imp

df = ds_handler.read_imputed_onehot_dataset()
df['Date'] = p.to_datetime(df['Date'], format='%d/%m/%Y')
df['Day'] = df['Date'].dt.weekday_name

train = util.get_frame_in_range(df, 3, 2016, 12, 2017)
test = util.get_frame_in_range(df, 1, 2018, 2, 2018)

test = prep.prepare_ds_to_customer_prediction(test, train)

train = prep.prepare_ds_to_customer_prediction(train, train)

final = p.concat([train, test])

# df['CustomersMeanPerStore'] = df.StoreID.map(df.groupby(['StoreID']).NumberOfCustomers.mean())
# df['CustomersVariancePerStore'] = df.StoreID.map(df.groupby(['StoreID']).NumberOfCustomers.var())


ds_handler.save_dataset(final, "mean_var_on_cust_from_tain.csv")