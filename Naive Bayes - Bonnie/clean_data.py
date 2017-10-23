import pandas as pd

x = 'C:\Users\guanqing\OneDrive\School\FALL 2017\COMP 551\Assignment2\LangClassifiers-P2\Naive Bayes\data\\train_set_x.csv'
y = 'C:\Users\guanqing\OneDrive\School\FALL 2017\COMP 551\Assignment2\LangClassifiers-P2\Naive Bayes\data\\train_set_y.csv'

x = 'C:\Users\guanqing\OneDrive\School\FALL 2017\COMP 551\Assignment2\LangClassifiers-P2\Naive Bayes\data\generatedTestSetX-200000.csv'
y = 'C:\Users\guanqing\OneDrive\School\FALL 2017\COMP 551\Assignment2\LangClassifiers-P2\Naive Bayes\data\generatedTestSetY-200000.csv'
dfx = pd.read_csv(x)
dfy = pd.read_csv(y)

df = dfx.merge(dfy, how='left',on='Id')
df['char_count'] = df['Text'].str.len()

df = df[~df.Text.isnull()]
df = df[~df.Text.str.isnumeric()]

#df = df[df.Text.str.len() >= 8]

df = df.reset_index(drop=True)

for i in range(5):
    dfi = df[df.Category==i]
    print '#sample = ',len(dfi)
    print 'char mean = ', dfi.char_count.mean()

df[['Id','Text']].to_csv('train_set_x_clean.csv', index=False)
df[['Id','Category']].to_csv('train_set_y_clean.csv', index=False)