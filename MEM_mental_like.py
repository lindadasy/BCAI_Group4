import statsmodels.api as sm
import statsmodels.formula.api as smf
import pandas as pd

user_data_path = './data/MNS_data_full.csv'
article_data_path = './data/articles_filtered.csv'
user_data = pd.read_csv(user_data_path)
article_data = pd.read_csv(article_data_path)
merged_data = pd.merge(user_data, 
                       article_data[['id', 'reading_time','difficulty']],  
                       left_on='article_id', 
                       right_on='id', 
                       how='inner')

features = ["id",'SleepHours', 'Tired', 'Excited', 
            'Motivated', 'Depression','Anxiety','likability',"difficulty", 'reading_time']
merged_data = merged_data[features].dropna() 

print(merged_data.shape)
print(merged_data.head())
print(f"Total rows and columns in merged data: {merged_data.shape}")
print("Columns in merged data:", merged_data.columns)

model = smf.mixedlm(
    "likability ~ SleepHours + Tired + Excited + Motivated + Depression + Anxiety + difficulty + reading_time",
    data=merged_data,
    groups=merged_data["id"], 
    re_formula="~1"
)

result = model.fit()
print(result.summary())

