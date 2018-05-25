import dataset.dataset as datasetfun
import seaborn as sb
import matplotlib.pyplot as pl
import pandas as pd
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler


if __name__ == "__main__":

    correlation_analysis = False
    PCA_analysis = False
    PCA_correlation_attributes = False
    PCA_analysis_attribute = 'Region_AreaKM2'

    sb.set_style("whitegrid")
    sb.set(style="white", color_codes=True)
    sb.set_context(rc={"font.family": 'sans', "font.size": 5, "axes.titlesize": 8, "axes.labelsize": 8})

    data = datasetfun.read_imputed_onehot_dataset()

    data_nominal = datasetfun.nominal_only(data)

    data_numeric = datasetfun.numeric_only(data)

    if correlation_analysis == True:

        numeric_corr = data_numeric.corr(method="pearson")

        numeric_heatmap = sb.heatmap(numeric_corr, square=True, annot=True, cmap="Blues")
        pl.show()

        numeric_clustermap = sb.clustermap(numeric_corr, square="True", cmap="Blues", annot=True)
        pl.show()

    if PCA_analysis == True:

        pca = PCA(n_components = 2)
        projected_data = pca.fit_transform(data_numeric)

        print("original shape:", data_numeric.shape)
        print('###################################')

        dataframe_projected = pd.DataFrame(data=projected_data, columns = ['component_1', 'component_2'])

        print("projected shape:", dataframe_projected.shape)
        print("projected head:", dataframe_projected.head(10))

        min_max_scaler = MinMaxScaler()
        data_projected_scaled = min_max_scaler.fit_transform(dataframe_projected)
        df_projected_scaled = pd.DataFrame(data=data_projected_scaled, columns = ['component_1', 'component_2'])

        print("projected scaled shape:", df_projected_scaled.shape)
        print("projected scaled head:", df_projected_scaled.head(10))

        marker_size = 10
        pl.scatter(data_projected_scaled[:, 0], data_projected_scaled[:, 1], marker_size,  alpha=0.5, c = data_numeric[PCA_analysis_attribute], cmap=pl.cm.get_cmap('spectral', 10))

        # delete the comment in order to use the projected data NOT normalized
        # pl.scatter(projected_data[:, 0], projected_data[:, 1], marker_size,  alpha=0.5, c = data_numeric[PCA_analysis_attribute], cmap=pl.cm.get_cmap('spectral', 10))
        pl.colorbar()
        pl.xlabel('component 1')
        pl.ylabel('component 2')
        pl.title('Principal Component Analysis')
        pl.show()

        if PCA_correlation_attributes == True:

            # can be changed between 'component_2' and 'component_1'. We allow to plot one component at time for sake
            # of clarity, to not plot to many attributes
            PCA_component_to_plot = 'component_1'

            print("dataframe PCA scaled:", df_projected_scaled.shape)
            print("dataframe numeric only:", data_numeric.shape)

            new_df = data_numeric.drop(['StoreID', 'IsOpen'], axis=1)

            new_df = pd.concat([new_df, df_projected_scaled[PCA_component_to_plot]], axis=1)

            print("merged dataframe:", new_df.shape)

            new_df_corr = new_df.corr(method="pearson")

            numeric_clustermap = sb.clustermap(new_df_corr, square="True", cmap="Blues", annot=True)
            pl.figure(figsize=(20,20))
            pl.show()


    ax = sb.jointplot(x="NearestCompetitor", y="NumberOfSales", data=data_numeric, marker='+')
    pl.show()
    ax = sb.jointplot(x="NearestCompetitor", y="NumberOfCustomers", data=data_numeric, marker='+')
    pl.show()
