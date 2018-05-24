import seaborn as sb
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from dataset.utility import get_frame_in_range
from dataset.dataset import to_numpy

def scatterplot(df, x=None, y=None, colour=None, regression=False):
    if x is None or y is None:
        raise AttributeError
    sb.lmplot(x, y, df, colour, fit_reg=regression)
    plt.show()


def pairplot(df, colour=None, marker='+'):
    sb.pairplot(df, colour, markers=marker)
    plt.show()


def showimage(image):
    """displays a single image"""
    plt.figure()
    plt.imshow(image)
    plt.show()


def correlation(df):
    corr = df.corr()
    sb.heatmap(corr, annot=True)
    plt.show()


def boxplot(df, x, y, jitter=False):
    sb.boxplot(x, y, data=df)
    sb.swarmplot(x, y, data=df, color=".25")
    plt.show()

def monthlysalesplot(df, bm, by, em, ey):
    month_x = []
    sales_y = []
    while bm != em or by != ey:
        dfframe = get_frame_in_range(df, bm, by, bm, by)
        month_x.append(str(bm) + "-" + str(by))
        sales_y.append(dfframe["NumberOfSales"].sum())
        bm += 1
        if bm == 13:
            bm = 1
            by += 1

    sb.barplot(x=month_x, y=sales_y, hue_order=None).set_title("Monthly Number of Sales (All shops)")
    plt.show()


def monthlycustomersplot(df, bm, by, em, ey):
    if em == 12:
        em = 1
        ey += 1
    else:
        em += 1

    month_x = []
    sales_y = []
    while bm != em or by != ey:
        dfframe = get_frame_in_range(df, bm, by, bm, by)
        month_x.append(str(bm) + "-" + str(by))
        sales_y.append(dfframe["NumberOfCustomers"].sum())
        bm += 1
        if bm == 13:
            bm = 1
            by += 1

    sb.barplot(x=month_x, y=sales_y, hue_order=None).set_title("Monthly Number of Customers (All shops)")
    plt.show()


def opendaybeforegeneralplot(df, storeID):
    IsOpenList = list(to_numpy(df["IsOpen"]).squeeze())
    IsOpenList.pop()
    IsOpenList.insert(0, 1)
    dftoplot = pd.DataFrame(np.array(IsOpenList).reshape(523021, 1), columns=["OpenDayBefore"])
    dftoplot = df.assign(OpenDayBefore=dftoplot)
    dftoplotpershop = dftoplot[dftoplot["StoreID"] == storeID]
    sb.boxplot(x="OpenDayBefore", y="NumberOfSales", data=dftoplotpershop).set_title("Sales / Shop Availability")
    plt.show()


def opendaybeforeonweekplot(df, storeID):
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
    df['Day'] = df['Date'].dt.weekday_name
    df = df.drop(df[df["Day"] == "Sunday"].index)
    IsOpenList = list(to_numpy(df["IsOpen"]).squeeze())
    IsOpenList.pop()
    IsOpenList.insert(0, 1)
    dftoplot = pd.DataFrame(np.array(IsOpenList).reshape(448375, 1), columns=["OpenDayBefore"])
    dftoplot = df.assign(OpenDayBefore=dftoplot)
    dftoplotpershop = dftoplot[dftoplot["StoreID"] == storeID]
    sb.boxplot(x="OpenDayBefore", y="NumberOfSales", data=dftoplotpershop).set_title("Sales / Shop Availability (-Sun)")
    plt.show()


if __name__ == '__main__':
    import dataset.dataset as d
    import dataset.utility as utils
    import models.keras.evaluation as eva
    import pandas as pd
    ds = d.read_imputed_onehot_dataset()
    y = 2016
    m = 3
    while y != 2018 or m != 3:
        sub_ds = utils.get_frame_in_range(ds, m, y, m, y)
        expected_out = d.to_numpy(sub_ds[['NumberOfSales']]).squeeze()
        print(str(m) + "/" + str(y) + ": ", expected_out.sum())
        m += 1
        if m == 13:
            m = 1
            y += 1
