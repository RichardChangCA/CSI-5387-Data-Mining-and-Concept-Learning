import csv
import numpy as np
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

data_dir = 'Dataset 3/groceries.csv'

data = []
with open(data_dir) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        while('' in row):
            row.remove('')
        data.append(row)

te = TransactionEncoder()
te_ary = te.fit(data).transform(data)
df = pd.DataFrame(te_ary, columns=te.columns_)

def my_apriori(min_support_value,save_file_name):
    # Building the model 
    frq_items = apriori(df, min_support = min_support_value, use_colnames = True)

    # Collecting the inferred rules in a dataframe
    rules = association_rules(frq_items, metric ="confidence", min_threshold = 0.5) 
    rules = rules.sort_values(['lift'], ascending =[False])
    rules["antecedent_len"] = rules["antecedents"].apply(lambda x: len(x))
    filtered_rules = rules[(rules["antecedent_len"] == 1)]
    f = open(save_file_name,'w+')
    f.write(str(filtered_rules.columns) + '\n')
    for item in filtered_rules.values:
        f.write(str(item) + '\n')
    f.close()

my_apriori(0.005,"results_5_new.txt")
my_apriori(0.001,"results_1_new.txt")