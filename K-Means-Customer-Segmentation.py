import pandas as pd
import numpy as np
import seaborn as sns
import seaborn as sns
import datetime as dt
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from yellowbrick.cluster import KElbowVisualizer

import warnings

pd.set_option('display.max_columns', None); pd.set_option('display.max_rows', None);
pd.set_option('display.float_format', lambda x: '%.0f' % x)


#Data set reading process was performed.
df_2010_2011 = pd.read_excel("datasets/online_retail_II.xlsx", sheet_name = "Year 2010-2011")
df = df_2010_2011.copy()
df.head()
# data preprocessing
# create rfm data

def create_rfm(dataframe):

    # VERIYI HAZIRLAMA
    dataframe["TotalPrice"] = dataframe["Quantity"] * dataframe["Price"]
    dataframe.dropna(inplace=True)
    dataframe = dataframe[~dataframe["Invoice"].str.contains("C", na=False)]

    # RFM METRIKLERININ HESAPLANMASI
    today_date = dt.datetime(2011, 12, 11)
    rfm = dataframe.groupby('Customer ID').agg({'InvoiceDate': lambda date: (today_date - date.max()).days,
                                                'Invoice': lambda num: num.nunique(),
                                                "TotalPrice": lambda price: price.sum()})
    rfm.columns = ['recency', 'frequency', "monetary"]
    rfm = rfm[(rfm['monetary'] > 0)]

    # RFM SKORLARININ HESAPLANMASI
    rfm["recency_score"] = pd.qcut(rfm['recency'], 5, labels=[5, 4, 3, 2, 1])
    rfm["frequency_score"] = pd.qcut(rfm["frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
    # Monetary segment tanımlamada kullanılmadığı için işlemlere alınmadı.

    # SEGMENTLERIN ISIMLENDIRILMESI
    rfm['rfm_segment'] = rfm['recency_score'].astype(str) + rfm['frequency_score'].astype(str)

    seg_map = {
        r'[1-2][1-2]': 'hibernating',
        r'[1-2][3-4]': 'at_risk',
        r'[1-2]5': 'cant_loose',
        r'3[1-2]': 'about_to_sleep',
        r'33': 'need_attention',
        r'[3-4][4-5]': 'loyal_customers',
        r'41': 'promising',
        r'51': 'new_customers',
        r'[4-5][2-3]': 'potential_loyalists',
        r'5[4-5]': 'champions'
    }

    rfm['rfm_segment'] = rfm['rfm_segment'].replace(seg_map, regex=True)
    rfm = rfm[["recency", "frequency", "monetary", "rfm_segment"]]
    return rfm

rfm = create_rfm(df)
rfm
rfm = rfm.loc[:,"recency":"monetary"]

# Clustering with K-Means algorithm

sc = MinMaxScaler((0,1))
df = sc.fit_transform(rfm)
kmeans = KMeans(n_clusters = 10)
k_fit = kmeans.fit(df)
k_fit.labels_

# determining number of clusters

kmeans = KMeans()
k_fit = kmeans.fit(df)
ssd = []

K = range(1,30)

for k in K:
    kmeans = KMeans(n_clusters = k).fit(df)
    ssd.append(kmeans.inertia_)

plt.plot(K, ssd, "bx-")
plt.xlabel("Distance Residuals according to chosen k")
plt.title("Elbow method for optimum k ")
plt.show()

# second way to do this

kmeans = KMeans()
elbow = KElbowVisualizer(kmeans, k=(2, 20))
elbow.fit(df)
elbow.show()

# final Kmeans model with optimum number of clusters

kmeans = KMeans(n_clusters = 6).fit(df)
kumeler = kmeans.labels_

pd.DataFrame({"Customer ID": rfm.index, "Kumeler": kumeler})

# final rfm dataframe after segmentation with K-means algorithm

rfm["cluster_no"] = kumeler
rfm["cluster_no"] = rfm["cluster_no"] + 1
rfm.head()

