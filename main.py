import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from fedot.api.main import Fedot

def read_csv(title):
    return pd.read_csv(title, encoding="WINDOWS-1251")

c1 = read_csv("steel_data_casting_1.csv")
s1 = read_csv("steel_data_shtamp_1.csv")
ls1 = read_csv("leg_steel_data_shtamp_1.csv")
ls2 = read_csv("leg_steel_data_shtamp_2.csv")


casting="casting"
c1[casting] = 1
s1[casting] = 0
ls1[casting] = 0
ls2[casting] = 0

def dif(df, arr_df):
    cols = set()
    for i in arr_df:
        cols.update(set(i)-set(df))
    return cols
def fill(df, arr_df):
    cols = dif(df, arr_df)
    for i in cols:
        df[i] = 0
    print(cols)

fill(c1, [s1, ls1, ls2])
fill(s1, [c1, ls1, ls2])
fill(ls1, [s1, c1, ls2])
fill(ls2, [c1, s1, ls1])
cols_without_fe=sorted(set(c1))
print(set(c1)==set(s1)==set(ls1)==set(ls2), cols_without_fe)

cols_l=list(filter(lambda x: x[0]=="l", list(cols_without_fe)))
cols_x=list(filter(lambda x: x[0]=="x", list(cols_without_fe)))
def add_fe(df):
    df["lFe"]=100
    df["xFe"]=100
    for _, i in enumerate(cols_l):
        df["lFe"]-=df[i]
    for _, i in enumerate(cols_x):
        df["xFe"]-=df[i]
    print(df.at[1, "lFe"], df.at[1, "xFe"])

add_fe(c1)
add_fe(s1)
add_fe(ls1)
add_fe(ls2)


df_full=pd.concat([c1,s1,ls1,ls2])
df=df_full.drop('Label', axis=1)

correlation_matrix = df.drop(columns=cols_l, axis=1).corr()
cm_sorted = correlation_matrix.unstack().sort_values(ascending=True)
print(cm_sorted)

plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', square=True)
plt.title('Correlation Matrix')
plt.show()

df=df.drop(columns=cols_l, axis=1)


df_shuffled=df.sample(frac=1, random_state=42).reset_index(drop=True)
split_point = int(0.6 * len(df_shuffled))
train_df = df_shuffled.iloc[:split_point]
test_df = df_shuffled.iloc[split_point:]
x_train = train_df.drop(casting, axis=1)
y_train = train_df[casting]
x_test = test_df.drop(casting, axis=1)
y_test = test_df[casting]
print(y_train, y_test)

model = Fedot(problem='classification', timeout=5, preset='best_quality', n_jobs=-1)
model.fit(features=x_train, target=y_train)
prediction = model.predict(features=x_test)
metrics = model.get_metrics(target=y_test)
print("metrics:", metrics)
print("the best model is", model.current_pipeline, model.current_pipeline.nodes)
