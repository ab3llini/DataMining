import dataset.dataset as d
import dataexploration.statistics.statistics as st
import preprocessing.imputation as imp
import numpy as np
import dataexploration.graphs.graphs as gr


calculate_miss_correlations = False


if __name__ == '__main__':
    ds = d.read_dataset()
    print("###### DS READ SUCCESFULLY ######")
    print("NUMERIC: ")
    st.describe(d.numeric_only(ds))
    print("NOMINAL: ")
    st.describe(d.nominal_only(ds))
    print("###################################################################")
    st.missing(ds, list(ds), printa=True)
    print("###################################################################")
    if calculate_miss_correlations:
        misses = st.misscorrelations(ds, 0.3)
        # {'CloudCover>Events': 0.6865787620504602,
        # 'CloudCover>Max_Gust_SpeedKm_h': 0.8914547971151745,
        # 'Events>Max_Gust_SpeedKm_h': 0.8572902061274155,
        # 'Max_VisibilityKm>CloudCover': 0.8677897336390897,
        # 'Max_VisibilityKm>Events': 0.9578408890456871,
        # 'Max_VisibilityKm>Max_Gust_SpeedKm_h': 0.8377138825189627,
        # 'Max_VisibilityKm>Mean_VisibilityKm': 1.0,
        # 'Max_VisibilityKm>Min_VisibilitykM': 1.0,
        # 'Mean_VisibilityKm>CloudCover': 0.8677897336390897,
        # 'Mean_VisibilityKm>Events': 0.9578408890456871,
        # 'Mean_VisibilityKm>Max_Gust_SpeedKm_h': 0.8377138825189627,
        # 'Mean_VisibilityKm>Max_VisibilityKm': 1.0,
        # 'Mean_VisibilityKm>Min_VisibilitykM': 1.0,
        # 'Min_VisibilitykM>CloudCover': 0.8677897336390897,
        # 'Min_VisibilitykM>Events': 0.9578408890456871,
        # 'Min_VisibilitykM>Max_Gust_SpeedKm_h': 0.8377138825189627,
        # 'Min_VisibilitykM>Max_VisibilityKm': 1.0,
        # 'Min_VisibilitykM>Mean_VisibilityKm': 1.0}

        print("MISS CORRELATIONS: ")
        print(misses)
    print("###################################################################")
    ds = imp.full_preprocess(ds)
    print("PREPROCESSING COMPLETE")

    print("###################################################################")
    st.missing(ds, list(ds), printa=True)

    print("###################################################################")
    for attr in d.nominal_only(ds):
        print("VALUES OF " + attr + ": " + str(d.values_of(ds, attr)))

    print("###################################################################")
    for attr in d.numeric_only(ds):
        print("DISTINCT NUMBER OF " + attr + ": " + str(len(d.values_of(ds, attr))))

    for attr in list(d.numeric_only(ds)):
        gr.scatterplot(ds, attr, 'NumberOfSales')

