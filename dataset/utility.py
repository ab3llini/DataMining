import pandas as pd
import datetime
import calendar

# Returns a masked frame that holds data between 2 dates
# f = Frame
# bm = beginning month
# bi = beginning year
# em = ending month
# ei = ending year


def get_frame_in_range(f, bm, by, em, ey):
    try :
        f["Date"] = pd.to_datetime(f["Date"], format="%d/%m/%Y")
    except Exception as e:
        print("*** Next time use the standard data format %d/%m/%Y, i was going to make this program crash :)")
        try:
            f["Date"] = pd.to_datetime(f["Date"], format="%Y-%m-%d")
        except TypeError as e2:
            print("*** can't get frame in range ! wtf format did you use for the date ??!")
            raise e2

    mask = (f['Date'] > datetime.date(by, bm, 1)) & (f['Date'] <= datetime.date(ey, em, calendar.monthrange(ey, em)[1]))
    return f.loc[mask]


'''
------------------------------------------------------------------------------------
# Sample usage to plot a bar chart representing the number of entries per each month

yrs = [2018, 2017, 2016]
mts = range(1, 13)
monthly_entries = {}

for y in yrs:
    for m in mts:
        if (y == 2016 and m < 3) or (y == 2018 and m > 2):
            continue
        x = get_frame_in_range(
            frame,
            y, m, y, m
        )
        count = x.shape[0]
        print("For %s/%s we have %s entries" % (m, y, count))
        key = "%s/%s" % (y, m)
        monthly_entries[key] = count


plt.xticks(rotation='vertical')
plt.bar(monthly_entries.keys(), monthly_entries.values())
plt.show()
------------------------------------------------------------------------------------
'''
