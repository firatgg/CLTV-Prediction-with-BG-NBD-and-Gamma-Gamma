###############################################################
# TASK 1: Data Preparation
###############################################################
import pandas as pd
import datetime as dt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from sklearn.preprocessing import MinMaxScaler
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.2f' % x)
pd.options.mode.chained_assignment = None

# 1. Read the OmniChannel.csv data. Create a copy of the dataframe.
df_ = pd.read_csv("C:/projects/pythonProject/data_sets/flo_data_20k.csv")
df = df_.copy()

# 2. Define the necessary functions outlier_thresholds and replace_with_thresholds to suppress outlier values.
# Note: When calculating CLTV, frequency values should be integers. Therefore, round their lower and upper limits using round().


def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    # noinspection SpellCheckingInspection
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = round(low_limit, 0)
    dataframe.loc[(dataframe[variable] > up_limit), variable] = round(up_limit, 0)

# 3. Suppress the outlier values of the variables "order_num_total_ever_online", "order_num_total_ever_offline", "customer_value_total_ever_offline", "customer_value_total_ever_online".
columns = ["order_num_total_ever_online", "order_num_total_ever_offline", "customer_value_total_ever_offline", "customer_value_total_ever_online"]
for col in columns:
    replace_with_thresholds(df, col)

# 4. Create new variables for the total number of purchases and spending for each customer who shops both online and offline.
df["order_num_total"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
df["customer_value_total"] = df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]

# 5. Examine variable types. Convert the types of variables representing dates to date.
date_columns = df.columns[df.columns.str.contains("date")]
df[date_columns] = df[date_columns].apply(pd.to_datetime)

###############################################################
# TASK 2: Creating CLTV Data Structure
###############################################################

# 1. Take the date two days after the last purchase date in the dataset as the analysis date.
last_order_date = df["last_order_date"].max()  # 2021-05-30
analysis_date = last_order_date + pd.DateOffset(days=2)

# 2. Create a new cltv dataframe containing customer_id, recency_cltv_weekly, T_weekly, frequency, and monetary_cltv_avg.
cltv_df = pd.DataFrame()
cltv_df["customer_id"] = df["master_id"]
cltv_df["recency_cltv_weekly"] = ((last_order_date - df["first_order_date"]).dt.days) / 7
cltv_df["T_weekly"] = ((analysis_date - df["first_order_date"]).dt.days) / 7
cltv_df["frequency"] = df["order_num_total"]
cltv_df["monetary_cltv_avg"] = df["customer_value_total"] / df["order_num_total"]

cltv_df.head()

###############################################################
# TASK 3: Building BG/NBD and Gamma-Gamma Models, Calculating 6-month CLTV
###############################################################

# 1. Build the BG/NBD model.
bgf = BetaGeoFitter(penalizer_coef=0.001)
bgf.fit(cltv_df['frequency'],
        cltv_df['recency_cltv_weekly'],
        cltv_df['T_weekly'])

# Predict expected purchases from customers within 3 months and add it to the cltv dataframe as exp_sales_3_month.
cltv_df["exp_sales_3_month"] = bgf.predict(4*3,
                                       cltv_df['frequency'],
                                       cltv_df['recency_cltv_weekly'],
                                       cltv_df['T_weekly'])

# Predict expected purchases from customers within 6 months and add it to the cltv dataframe as exp_sales_6_month.
cltv_df["exp_sales_6_month"] = bgf.predict(4*6,
                                       cltv_df['frequency'],
                                       cltv_df['recency_cltv_weekly'],
                                       cltv_df['T_weekly'])

# 3. Analyze the top 10 customers with the most purchases in the 3rd and 6th months. Is there a difference?
top_10_3_month = cltv_df.sort_values("exp_sales_3_month", ascending=False)[:10]
top_10_6_month = cltv_df.sort_values("exp_sales_6_month", ascending=False)[:10]

print("Top 10 Customers for 3 Months:")
print(top_10_3_month)

print("\nTop 10 Customers for 6 Months:")
print(top_10_6_month)

# 2. Fit the Gamma-Gamma model. Predict the expected average value that customers will leave and add it to the cltv dataframe as exp_average_value.
ggf = GammaGammaFitter(penalizer_coef=0.01)
ggf.fit(cltv_df['frequency'], cltv_df['monetary_cltv_avg'])
cltv_df["exp_average_value"] = ggf.conditional_expected_average_profit(cltv_df['frequency'],
                                                                cltv_df['monetary_cltv_avg'])
cltv_df.head()

# 3. Calculate the 6-month CLTV and add it to the dataframe with the name cltv.
cltv = ggf.customer_lifetime_value(bgf,
                                   cltv_df['frequency'],
                                   cltv_df['recency_cltv_weekly'],
                                   cltv_df['T_weekly'],
                                   cltv_df['monetary_cltv_avg'],
                                   time=6,
                                   freq="W",
                                   discount_rate=0.01)
cltv_df["cltv"] = cltv

# Observe the top 20 customers by CLTV.
top_20_cltv = cltv_df.sort_values("cltv", ascending=False)[:20]
print("\nTop 20 Customers by CLTV:")
print(top_20_cltv)

###############################################################
# TASK 4: Creating Segments According to CLTV
###############################################################

# 1. Divide all your customers into 4 groups (segments) based on the 6-month standardized CLTV and add group names to the dataset.
# Assign with the name cltv_segment.
cltv_df["cltv_segment"] = pd.qcut(cltv_df["cltv"], 4, labels=["D", "C", "B", "A"])
cltv_df.head()

# 2. Is it logical to categorize customers into 4 groups according to CLTV scores? Should it be less or more? Comment.

# 3. Briefly suggest 6-month actions for management for 2 selected groups from the 4 groups.

###############################################################
# BONUS: Generalize the entire process with a function.
###############################################################

def create_cltv_df(dataframe):

    # Data Preparation
    columns = ["order_num_total_ever_online", "order_num_total_ever_offline", "customer_value_total_ever_offline","customer_value_total_ever_online"]
    for col in columns:
        replace_with_thresholds(dataframe, col)

    dataframe["order_num_total"] = dataframe["order_num_total_ever_online"] + dataframe["order_num_total_ever_offline"]
    dataframe["customer_value_total"] = dataframe["customer_value_total_ever_offline"] + dataframe["customer_value_total_ever_online"]
    dataframe = dataframe[~((dataframe["customer_value_total"] == 0) | (dataframe["order_num_total"] == 0))]
    date_columns = dataframe.columns[dataframe.columns.str.contains("date")]
    dataframe[date_columns] = dataframe[date_columns].apply(pd.to_datetime)

    # Creating CLTV Data Structure
    last_order_date = dataframe["last_order_date"].max()
    analysis_date = dt.datetime(2021, 6, 1)
    cltv_df = pd.DataFrame()
    cltv_df["customer_id"] = dataframe["master_id"]
    cltv_df["recency_cltv_weekly"] = ((dataframe["last_order_date"] - dataframe["first_order_date"]).dt.days) / 7
    cltv_df["T_weekly"] = ((analysis_date - dataframe["first_order_date"]).dt.days) / 7
    cltv_df["frequency"] = dataframe["order_num_total"]
    cltv_df["monetary_cltv_avg"] = dataframe["customer_value_total"] / dataframe["order_num_total"]
    cltv_df = cltv_df[(cltv_df['frequency'] > 1)]

    # Building BG-NBD Model
    bgf = BetaGeoFitter(penalizer_coef=0.001)
    bgf.fit(cltv_df['frequency'],
            cltv_df['recency_cltv_weekly'],
            cltv_df['T_weekly'])
    cltv_df["exp_sales_3_month"] = bgf.predict(4 * 3,
                                               cltv_df['frequency'],
                                               cltv_df['recency_cltv_weekly'],
                                               cltv_df['T_weekly'])
    cltv_df["exp_sales_6_month"] = bgf.predict(4 * 6,
                                               cltv_df['frequency'],
                                               cltv_df['recency_cltv_weekly'],
                                               cltv_df['T_weekly'])

    # Building Gamma-Gamma Model
    ggf = GammaGammaFitter(penalizer_coef=0.01)
    ggf.fit(cltv_df['frequency'], cltv_df['monetary_cltv_avg'])
    cltv_df["exp_average_value"] = ggf.conditional_expected_average_profit(cltv_df['frequency'],
                                                                           cltv_df['monetary_cltv_avg'])

    # CLTV Prediction
    cltv = ggf.customer_lifetime_value(bgf,
                                       cltv_df['frequency'],
                                       cltv_df['recency_cltv_weekly'],
                                       cltv_df['T_weekly'],
                                       cltv_df['monetary_cltv_avg'],
                                       time=6,
                                       freq="W",
                                       discount_rate=0.01)
    cltv_df["cltv"] = cltv

    # CLTV Segmentation
    cltv_df["cltv_segment"] = pd.qcut(cltv_df["cltv"], 4, labels=["D", "C", "B", "A"])

    return cltv_df

cltv_df = create_cltv_df(df)

cltv_df.head(10)
