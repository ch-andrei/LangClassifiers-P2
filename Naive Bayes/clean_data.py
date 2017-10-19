import pandas as pd

dfx = pd.read_csv('C:\Users\guanqing\OneDrive\School\FALL 2017\COMP 551\Assignment2\LangClassifiers-P2\Naive Bayes\data\\train_set_x.csv')
dfy = pd.read_csv('C:\Users\guanqing\OneDrive\School\FALL 2017\COMP 551\Assignment2\LangClassifiers-P2\Naive Bayes\data\\train_set_y.csv')

df=dfx.merge(dfy, how='left',on='Id')

df = df[~df.Text.isnull()]
df = df[~df.Text.str.isnumeric()]

df = df[df.Text.str.len() >= 8]

df = df.reset_index(drop=True)

for i in range(5):
    print len(df[df.Category==i])

df[['Id','Text']].to_csv('train_set_x_clean.csv', index=False)
df[['Id','Category']].to_csv('train_set_y_clean.csv', index=False)