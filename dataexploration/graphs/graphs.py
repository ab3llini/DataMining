import seaborn as sb
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from dataset.utility import get_frame_in_range
import dataset.dataset as ds
from dataset.dataset import values_of
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


def barplot(df, x, y, hue=None):
    sb.barplot(x=x, y=y, data=df, hue=hue).set_title(x + "/" + y)
    plt.show()


def monthlyplot(df, bm=3, by=2016, em=2, ey=2018, regions=None, storetype=False, target="NumberOfSales"):
    if em == 12:
        em = 1
        ey += 1
    else:
        em += 1

    base_bm = bm
    base_by = by
    base_em = em
    base_ey = ey

    if regions is None and not storetype:
        month_x = []
        sales_y = []
        while bm != em or by != ey:
            dfframe = get_frame_in_range(df, bm, by, bm, by)
            month_x.append(str(bm) + "-" + str(by))
            sales_y.append(dfframe[target].sum())
            bm += 1
            if bm == 13:
                bm = 1
                by += 1

        sb.pointplot(x=month_x, y=sales_y, hue="Region", hue_order=None).set_title("Monthly " + target + " (All shops)")
    elif storetype:
        col = 0
        palette = sb.color_palette("hls", 4)
        month_x = []
        sales_y_1 = []
        sales_y_2 = []
        sales_y_3 = []
        sales_y_4 = []
        while bm != em or by != ey:
            dfframe = get_frame_in_range(df, bm, by, bm, by)
            month_x.append(str(bm) + "-" + str(by))
            sales_y_1.append(dfframe[dfframe["StoreType_Hyper Market"] == 1][target].mean())
            sales_y_2.append(dfframe[dfframe["StoreType_Super Market"] == 1][target].mean())
            sales_y_3.append(dfframe[dfframe["StoreType_Standard Market"] == 1][target].mean())
            sales_y_4.append(dfframe[dfframe["StoreType_Shopping Center"] == 1][target].mean())
            bm += 1
            if bm == 13:
                bm = 1
                by += 1

        pdf1 = pd.DataFrame(data={"Month": month_x, target: sales_y_1, "Type": "Hyper Market"})
        pdf2 = pd.DataFrame(data={"Month": month_x, target: sales_y_2, "Type": "Super Market"})
        pdf3 = pd.DataFrame(data={"Month": month_x, target: sales_y_3, "Type": "Standard Market"})
        pdf4 = pd.DataFrame(data={"Month": month_x, target: sales_y_4, "Type": "Shopping Center"})
        pdf = pd.concat([pdf1, pdf2, pdf3, pdf4])
        sb.pointplot(x="Month", y=target, data=pdf, hue="Type")
    else:
        pdf = []
        palette = sb.color_palette("hls", 11)
        for i in regions:
            month_x = []
            sales_y = []
            while bm != em or by != ey:
                dfframe = get_frame_in_range(df[df["Region"] == i], bm, by, bm, by)
                month_x.append(str(bm) + "-" + str(by))
                sales_y.append(dfframe[target].mean())
                bm += 1
                if bm == 13:
                    bm = 1
                    by += 1
            pdf.append(pd.DataFrame(data={"Month":month_x, target:sales_y, "Region":i}))

            bm = base_bm
            by = base_by
            em = base_em
            ey = base_ey
        pdf = pd.concat(pdf)
        sb.pointplot(x="Month", y=target, data=pdf, hue="Region")

    plt.legend()
    plt.show()


def opendaybeforegeneralplot(df, storeID):
    IsOpenList = list(to_numpy(df["IsOpen"]).squeeze())
    IsOpenList.pop()
    IsOpenList.insert(0, 1)
    dftoplot = pd.DataFrame(np.array(IsOpenList).reshape(523021, 1), columns=["OpenDayBefore"])
    dftoplot = df.assign(OpenDayBefore=dftoplot)
    dftoplotpershop = dftoplot[dftoplot["StoreID"] == storeID]
    dftoplotpershop = dftoplot[dftoplot["IsOpen"] == 1]
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
    dftoplotpershop = dftoplot[dftoplot["IsOpen"] == 1]
    sb.boxplot(x="OpenDayBefore", y="NumberOfSales", data=dftoplotpershop).set_title("Sales / Shop Availability (-Sun)")
    plt.show()


# If storeID = 0, ignore shop selection.
def competitorplot(df, target="NumberOfSales"):
    title = target + " / Nearest Competitor"
    sb.barplot(x="NearestCompetitor", y=target, data=df).set_title(title)
    plt.show()


def frequencypershop(df, storeID, target="NumberOfSales", daily=False, shoptype=False, cloudcover=False, events=False, region=False):
    title = target + " Distribution"
    if daily:
        df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
        df['Day'] = df['Date'].dt.weekday_name

    if storeID != 0:
        df = df[df["StoreID"] == storeID]
        title = title + " - Shop" + str(storeID)

    if daily:
        sb.distplot(a=df[df["Day"] == "Monday"][target], label="Monday").set_title(title)
        sb.distplot(a=df[df["Day"] == "Tuesday"][target], label="Tuesday").set_title(title)
        sb.distplot(a=df[df["Day"] == "Wednesday"][target], label="Wednesday").set_title(title)
        sb.distplot(a=df[df["Day"] == "Thursday"][target], label="Thursday").set_title(title)
        sb.distplot(a=df[df["Day"] == "Friday"][target], label="Friday").set_title(title)
        sb.distplot(a=df[df["Day"] == "Saturday"][target], label="Saturday").set_title(title)
        sb.distplot(a=df[df["Day"] == "Sunday"][target], label="Sunday").set_title(title)
        plt.legend()
    elif shoptype:
        sb.distplot(a=df[df["StoreType_Hyper Market"] == 1][target], label="Hyper Market").set_title(title)
        sb.distplot(a=df[df["StoreType_Super Market"] == 1][target], label="Super Market").set_title(title)
        sb.distplot(a=df[df["StoreType_Standard Market"] == 1][target], label="Standard Market").set_title(title)
        sb.distplot(a=df[df["StoreType_Shopping Center"] == 1][target], label="Shopping Center").set_title(title)
        plt.legend()
    elif cloudcover:
        sb.distplot(a=df[df["CloudCover"] == 0][target], label="0").set_title(title)
        sb.distplot(a=df[df["CloudCover"] == 1][target], label="1").set_title(title)
        sb.distplot(a=df[df["CloudCover"] == 2][target], label="2").set_title(title)
        sb.distplot(a=df[df["CloudCover"] == 3][target], label="3").set_title(title)
        sb.distplot(a=df[df["CloudCover"] == 4][target], label="4").set_title(title)
        sb.distplot(a=df[df["CloudCover"] == 5][target], label="5").set_title(title)
        sb.distplot(a=df[df["CloudCover"] == 6][target], label="6").set_title(title)
        sb.distplot(a=df[df["CloudCover"] == 7][target], label="7").set_title(title)
        sb.distplot(a=df[df["CloudCover"] == 8][target], label="8").set_title(title)
        plt.legend()
    elif events:
        sb.distplot(a=df[df["Events_none"] == 1][target], label="None").set_title(title)
        sb.distplot(a=df[df["Events_Fog"] == 1][target], label="Fog").set_title(title)
        sb.distplot(a=df[df["Events_Rain"] == 1][target], label="Rain").set_title(title)
        sb.distplot(a=df[df["Events_Thunderstorm"] == 1][target], label="Thunderstorm").set_title(title)
        sb.distplot(a=df[df["Events_Snow"] == 1][target], label="Snow").set_title(title)
        sb.distplot(a=df[df["Events_Hail"] == 1][target], label="Hail").set_title(title)
        plt.legend()
    elif region:
        sb.distplot(a=df[df["Region"] == 0][target], label="0").set_title(title)
        sb.distplot(a=df[df["Region"] == 1][target], label="1").set_title(title)
        sb.distplot(a=df[df["Region"] == 2][target], label="2").set_title(title)
        sb.distplot(a=df[df["Region"] == 3][target], label="3").set_title(title)
        sb.distplot(a=df[df["Region"] == 4][target], label="4").set_title(title)
        sb.distplot(a=df[df["Region"] == 5][target], label="5").set_title(title)
        sb.distplot(a=df[df["Region"] == 6][target], label="6").set_title(title)
        sb.distplot(a=df[df["Region"] == 7][target], label="7").set_title(title)
        sb.distplot(a=df[df["Region"] == 8][target], label="8").set_title(title)
        sb.distplot(a=df[df["Region"] == 9][target], label="9").set_title(title)
        sb.distplot(a=df[df["Region"] == 10][target], label="10").set_title(title)
        plt.legend()

    else:
        sb.distplot(a=df[target]).set_title(title)
    plt.show()


def availabilityplot(df):
    title = "Availability Distribution Per Day"
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
    df['Day'] = df['Date'].dt.weekday_name

    sb.distplot(a=df[df["Day"] == "Monday"]["IsOpen"], label="Monday").set_title(title)
    sb.distplot(a=df[df["Day"] == "Tuesday"]["IsOpen"], label="Tuesday").set_title(title)
    sb.distplot(a=df[df["Day"] == "Wednesday"]["IsOpen"], label="Wednesday").set_title(title)
    sb.distplot(a=df[df["Day"] == "Thursday"]["IsOpen"], label="Thursday").set_title(title)
    sb.distplot(a=df[df["Day"] == "Friday"]["IsOpen"], label="Friday").set_title(title)
    sb.distplot(a=df[df["Day"] == "Saturday"]["IsOpen"], label="Saturday").set_title(title)
    sb.distplot(a=df[df["Day"] == "Sunday"]["IsOpen"], label="Sunday").set_title(title)

    plt.legend()
    plt.show()


def scattertargets(df, hue):
    if hue == "Day":
        df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
        df['Day'] = df['Date'].dt.weekday_name

    sb.lmplot(x="NumberOfCustomers", y="NumberOfSales", data=df, hue=hue)
    plt.show()


def cloudcoverbarplot(df, target="NumberOfSales"):
    sb.barplot(x="CloudCover", y=target, data=df).set_title(target + "Per CloudCover (All shops)")
    plt.show()


def frequencystatplot(df):
    regions = [0,1,2,3,4,5,6,7,8,9,10]
    compstat = []
    for i in regions:
        compstat.append(len(values_of(df[df["Region"] == i], "StoreID")))

    sb.barplot(x=regions, y=compstat)
    plt.show()




if __name__ == '__main__':
    import dataset.dataset as d
    import dataset.utility as utils
    import pandas as pd
    ds = d.read_imputed_onehot_dataset()
    monthlyplot(ds)
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
