# CLTV Prediction with BG-NBD and Gamma-Gamma

## Business Problem
FLO aims to set a roadmap for its sales and marketing activities. To plan the company's medium to long-term strategies, it is crucial to predict the potential value that existing customers will bring to the company in the future.

## Dataset Story
The dataset consists of information derived from the past shopping behaviors of customers who made their last purchases in 2020-2021 through OmniChannel (both online and offline shopping).

- `master_id`: Unique customer number
- `order_channel`: Channel used for shopping (Android, iOS, Desktop, Mobile, Offline)
- `last_order_channel`: Channel where the last purchase was made
- `first_order_date`: Date of the customer's first purchase
- `last_order_date`: Date of the customer's last purchase
- `last_order_date_online`: Date of the customer's last online purchase
- `last_order_date_offline`: Date of the customer's last offline purchase
- `order_num_total_ever_online`: Total number of purchases made by the customer online
- `order_num_total_ever_offline`: Total number of purchases made by the customer offline
- `customer_value_total_ever_offline`: Total amount spent by the customer in offline purchases
- `customer_value_total_ever_online`: Total amount spent by the customer in online purchases
- `interested_in_categories_12`: List of categories the customer shopped in the last 12 months

## Tasks

### Task 1: Data Preparation
1. Read the `flo_data_20K.csv` data. Create a copy of the dataframe.
2. Define the necessary functions `outlier_thresholds` and `replace_with_thresholds` to suppress outlier values.
   - Note: When calculating CLTV, frequency values should be integers. Therefore, round their lower and upper limits using `round()`.
3. Suppress the outlier values of the variables `"order_num_total_ever_online"`, `"order_num_total_ever_offline"`, `"customer_value_total_ever_offline"`, `"customer_value_total_ever_online"`.
4. Create new variables for the total number of purchases and spending for each customer who shops both online and offline.
5. Examine variable types. Convert the types of variables representing dates to `date`.

### Task 2: Creating CLTV Data Structure
1. Take the date two days after the last purchase date in the dataset as the analysis date.
2. Create a new cltv dataframe containing `customer_id`, `recency_cltv_weekly`, `T_weekly`, `frequency`, and `monetary_cltv_avg`.
   - Monetary value will be the average value per purchase, and recency and tenure values will be expressed in weeks.

### Task 3: Building BG/NBD and Gamma-Gamma Models, Calculating CLTV
1. Fit the BG/NBD model.
   - a. Predict expected purchases from customers within 3 months and add as `exp_sales_3_month` to the cltv dataframe.
   - b. Predict expected purchases from customers within 6 months and add as `exp_sales_6_month` to the cltv dataframe.
2. Fit the Gamma-Gamma model. Predict the average value that customers will leave and add it as `exp_average_value` to the cltv dataframe.
3. Calculate the 6-month CLTV and add it to the dataframe as `cltv`.
   - a. Standardize the calculated CLTV values and create a variable named `scaled_cltv`.
   - b. Observe the top 20 individuals with the highest CLTV values.

### Task 4: Creating Segments According to CLTV
1. Divide all your customers into 4 groups based on the standardized CLTV for the last 6 months and add group names to the dataset as `cltv_segment`.
2. Provide brief 6-month action recommendations to management for two selected groups from the 4 segments.

### Task 5: Functionize the Entire Process
