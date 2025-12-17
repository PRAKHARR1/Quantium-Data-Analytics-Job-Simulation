#Task 1: Customer Analytics & Purchasing Behaviour (Python Only)
#Data Loading & Initial Checks (Python)
import pandas as pd
import numpy as np

transactions = pd.read_excel("QVI_transaction_data.xlsx")
customers = pd.read_csv("QVI_purchase_behaviour.csv")

transactions.head()
customers.head()

#Data Quality Checks
transactions.info()
customers.info()

transactions.isnull().sum()
customers.isnull().sum()

#Feature Engineering
#Create Pack Size
transactions['PACK_SIZE'] = (
    transactions['PROD_NAME']
    .str.extract('(\d+)')
    .astype(float)
)

#Create Brand Name
transactions['BRAND'] = transactions['PROD_NAME'].str.split().str[0]

#Merge Transaction & Customer Data
data = transactions.merge(
    customers,
    how='left',
    on='LYLTY_CARD_NBR'
)

#Customer Segment Analysis
#Sales by Lifestage & Premium Customer
segment_summary = data.groupby(
    ['LIFESTAGE', 'PREMIUM_CUSTOMER']
).agg(
    total_sales=('TOT_SALES', 'sum'),
    total_qty=('PROD_QTY', 'sum'),
    customers=('LYLTY_CARD_NBR', 'nunique')
).reset_index()

segment_summary['avg_spend_per_customer'] = (
    segment_summary['total_sales'] / segment_summary['customers']
)
segment_summary['avg_units_per_customer'] = (
    segment_summary['total_qty'] / segment_summary['customers']
)

#Brand & Pack Size Preferences
#Most Popular Brands by Segment
brand_pref = data.groupby(
    ['LIFESTAGE', 'PREMIUM_CUSTOMER', 'BRAND']
).agg(
    total_qty=('PROD_QTY', 'sum')
).reset_index()

#TASK 2

# Task 2: Experimentation & Uplift Testing

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load Data
df = pd.read_csv("QVI_data.csv")

df['DATE'] = pd.to_datetime(df['DATE'])
df['MONTH'] = df['DATE'].dt.to_period('M')

#Create Monthly Metrics
monthly = df.groupby(['STORE_NBR', 'MONTH']).agg(
    total_sales=('TOT_SALES', 'sum'),
    customers=('LYLTY_CARD_NBR', 'nunique'),
    transactions=('TXN_ID', 'count')
).reset_index()

monthly['avg_txn_per_customer'] = (
    monthly['transactions'] / monthly['customers']
)

#Function to Select Control Store
#    (Using Pearson Correlation)
def find_control_store(trial_store, metric='total_sales'):
    trial_data = monthly[monthly['STORE_NBR'] == trial_store][metric].values
    scores = {}

    for store in monthly['STORE_NBR'].unique():
        if store == trial_store:
            continue
        control_data = monthly[monthly['STORE_NBR'] == store][metric].values
        min_len = min(len(trial_data), len(control_data))
        corr = np.corrcoef(trial_data[:min_len], control_data[:min_len])[0,1]
        scores[store] = corr

    return max(scores, key=scores.get)
# Identify Control Stores
trial_stores = [77, 86, 88]
control_stores = {}

for store in trial_stores:
    control_stores[store] = find_control_store(store)

print("Trial vs Control Stores:")
print(control_stores)

# Visual Comparison
for trial, control in control_stores.items():
    trial_data = monthly[monthly['STORE_NBR'] == trial]
    control_data = monthly[monthly['STORE_NBR'] == control]

    plt.figure()
    plt.plot(trial_data['MONTH'].astype(str), trial_data['total_sales'], label=f"Trial {trial}")
    plt.plot(control_data['MONTH'].astype(str), control_data['total_sales'], label=f"Control {control}")
    plt.xticks(rotation=90)
    plt.title(f"Total Sales: Trial Store {trial} vs Control Store {control}")
    plt.legend()
    plt.tight_layout()
    plt.show()

# Trial Period Comparison
trial_period = monthly['MONTH'] >= '2019-02'

for trial, control in control_stores.items():
    trial_sales = monthly[
        (monthly['STORE_NBR'] == trial) & trial_period
    ]['total_sales'].sum()

    control_sales = monthly[
        (monthly['STORE_NBR'] == control) & trial_period
    ]['total_sales'].sum()

    print(f"\nStore {trial} vs Control {control}")
    print(f"Trial Sales   : {trial_sales:.2f}")
    print(f"Control Sales : {control_sales:.2f}")
    print(f"Uplift        : {trial_sales - control_sales:.2f}")

# Driver Analysis
driver_analysis = monthly.groupby('STORE_NBR').agg(
    avg_customers=('customers', 'mean'),
    avg_txn_per_customer=('avg_txn_per_customer', 'mean')
)

print("\nDriver Analysis (Customers vs Transactions per Customer)")
print(driver_analysis.loc[trial_stores])



