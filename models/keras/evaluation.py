import dataset.dataset as d
import numpy as np


def tss(df, attr):
    arr = d.to_numpy(df[[attr]]).squeeze()
    mean = arr.mean()
    return np.sum(np.square(arr-mean))


def rss(df, preds, attr):
    arr = d.to_numpy(df[[attr]]).squeeze()
    return np.sum(np.square(arr-preds))


def r2(df, preds, attr):
    return 1 - rss(df, preds, attr) / tss(df, attr)


def region_error(real, pred, regions, shop_ids, dates):
    months = np.unique(np.array([date.split("-")[1] for date in dates]))
    monthsmap = map_values_pos(months)
    shops = np.unique(shop_ids)
    shopsmap = map_values_pos(shops)
    regionsunique = np.unique(regions)
    regionsmap = map_values_pos(regionsunique)
    errorsshape = [len(shops), len(months)]
    total_real = calculate_total_sales_per_month_per_shop(real, shop_ids, dates, monthsmap, shopsmap, errorsshape)
    total_pred = calculate_total_sales_per_month_per_shop(pred, shop_ids, dates, monthsmap, shopsmap, errorsshape)
    shop_region_map = map_shop_region(shop_ids, regions)
    regions_errors = np.zeros(regionsunique.shape)
    regions_totals = np.zeros(regionsunique.shape)
    for id in shops:
        for m in months:
            regions_errors[regionsmap[str(shop_region_map[str(id)])]] += \
                abs(total_real[shopsmap[str(id)]][monthsmap[str(m)]] - total_pred[shopsmap[str(id)]][monthsmap[str(m)]])
            regions_totals[regionsmap[str(shop_region_map[str(id)])]] += total_real[shopsmap[str(id)]][monthsmap[str(m)]]
    for i in range(len(regions_errors)):
        regions_errors[i] = regions_errors[i] / regions_totals[i]
    return regions_errors, total_real, total_pred


def getmonth(date):
    return date.split("-")[1]


def map_values_pos(vals):
    i = 0
    mapping = dict()
    for m in vals:
        mapping[str(m)] = i
        i += 1
    return mapping


def map_shop_region(shopids, regions):
    ris = dict()
    for i in range(len(shopids)):
        ris[str(shopids[i])] = regions[i]
    return ris


def calculate_total_sales_per_month_per_shop(values, shop_ids, dates, monthsmap, shopsmap, risshape):
    ris = np.zeros(shape=risshape)
    tot = len(values)
    for i in range(tot):
        ris[shopsmap[str(shop_ids[i])]][monthsmap[str(getmonth(dates[i]))]] += values[i]
    return ris
