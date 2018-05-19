import pandas as pd
import datetime
import calendar

# Returns a masked frame that holds data between 2 dates
# f = Frame
# bm = beginning month
# bi = beginning year
# em = ending month
# ei = ending year


def get_frame_in_range(f, bm, bi, em, ei):
    f["Date"] = pd.to_datetime(f["Date"], format="%d/%m/%Y")
    mask = (f['Date'] > datetime.date(bi, bm, 1)) & (f['Date'] <= datetime.date(ei, em, calendar.monthrange(ei, em)[1]))
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

