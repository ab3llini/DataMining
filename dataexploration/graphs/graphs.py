import seaborn as sb
import matplotlib.pyplot as plt
import numpy
from dataset.utility import get_frame_in_range

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

    sb.barplot(x=month_x, y=sales_y, hue_order=None)
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

    sb.barplot(x=month_x, y=sales_y, hue_order=None)
    plt.show()
