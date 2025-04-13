'''
df = pd.read_csv('laptop_data.csv')
#print(df.head())
#print(df.info())
#print(df.isnull().sum())
df.drop(columns=['Unnamed: 0'] , inplace=True)
#print(df.head())
df['Ram'] = df['Ram'].str.replace('GB' ,'')
df['Weight'] = df['Weight'].str.replace('kg' ,'')
#print(df['Ram'].head())
df['Ram']  =df['Ram'].astype('int32')
df['Weight'] = df['Weight'].astype('float32')
#print(df.info())

import seaborn as sn
import matplotlib.pyplot as plt
sn.displot(df['Price'])
#plt.show()
#print(df['Company'].value_counts())
sn.barplot(x=df['Company'] , y=df['Price'])
plt.xticks(rotation='vertical')
#plt.show()
#print(df['Typename'].head())
#print(df['ScreenResolution'].value_counts())
df['Touchscreen'] = df['ScreenResolution'].apply(lambda x:1 if x in 'Touchscreen' else 0 )
df['IPS'] = df['ScreenResolution'].apply(lambda x:1 if x in 'IPS' else 0 )
#print(df['Touchscreen'].head())
#print(df['Touchscreen'].value_counts().plot(kind='bar'))
#plt.show()
new = df['ScreenResolution'].str.split('x',n=1,expand=True)
#print(new)
df['X_res'] = new[0]
df['Y_res'] = new[1]
df['X_res'] = df['X_res'].str.findall('\d+').apply(lambda x : x[0])
#print(df['X_res'])
df['X_res'] = df['X_res'].astype('int')
df['Y_res'] = df['Y_res'].astype('int')
#print(df.info())
#print(df.corr()['Price'])
df['ppi'] = ((df['X_res']**2) + (df['Y_res']**2))**0.5/(df['Inches']).astype('float')
#print(df['ppi'])
df.drop(columns = ['ScreenResolution' , 'X_res' , 'Y_res' , 'Inches'] , inplace=True)
#print(df['Cpu'].value_counts())

df['Cpu_name'] = df['Cpu'].apply(lambda x : " ".join(x.split()[0:3]))
#print(df['Cpu_name'])

def fetch_data(text):
    if text == 'Intel Core i5' or text =='Intel Core i5' or text =='Intel Core i3':
        return text
    else:
        if text.split()[0] == 'Intel':
            return 'Intel Other Processor'
        else:
            return 'AMD Processor'

df['Cpu_Brand'] = df['Cpu_name'].apply(fetch_data)
#print(df['Cpu_Brand'])
df.drop(columns=['Cpu' , 'Cpu_name'] , inplace=True)
#print(df.head())


#print(df['Memory'].sample(5))


df['Memory'] = df['Memory'].astype(str).replace('\.0', '', regex=True)
df["Memory"] = df["Memory"].str.replace('GB', '')
df["Memory"] = df["Memory"].str.replace('TB', '000')

# Extract numeric values for different memory types using regular expressions
df['HDD'] = df['Memory'].str.extract(r'(\d+)\s*HDD').astype(float).fillna(0)
df['SSD'] = df['Memory'].str.extract(r'(\d+)\s*SSD').astype(float).fillna(0)
df['Hybrid'] = df['Memory'].str.extract(r'(\d+)\s*Hybrid').astype(float).fillna(0)
df['Flash_Storage'] = df['Memory'].str.extract(r'(\d+)\s*Flash Storage').astype(float).fillna(0)

# Drop the original 'Memory' column
df.drop(columns=['Memory'], inplace=True)

#print(df.sample(5))
df.drop(columns=['Hybrid','Flash_Storage'],inplace=True)
df['Gpu brand'] = df['Gpu'].apply(lambda x:x.split()[0])
#print(df['Gpu brand'].value_counts())
df = df[df['Gpu brand']!='ARM']
#print(df['Gpu brand'].value_counts())
df.drop(columns=['Gpu'],inplace=True)
print(df.columns)

def cat_os(inp):
    if inp == 'Windows 10' or inp == 'Windows 7' or inp == 'Windows 10 S':
        return 'Windows'
    elif inp == 'macOS' or inp == 'Mac OS X':
        return 'Mac'
    else:
        return 'Others/No OS/Linux'

df['os'] = df['OpSys'].apply(cat_os)
#print(df.head())
df.drop(columns=['OpSys'],inplace=True)

X= df.drop(columns=['Price'] )
y = np.log(df['Price'])

#print(X)
#print(y)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.15,random_state=2)

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score,mean_absolute_error
step1 = ColumnTransformer(transformers=[
    ('col_tnf',OneHotEncoder(sparse=False,drop='first'),[0,1,7,10,11])
],remainder='passthrough')

#step2 = LinearRegression()
step2 = RandomForestRegressor(n_estimators=100,
                              random_state=3,
                              max_samples=0.5,
                              max_features=0.75,
                              max_depth=15)
pipe = Pipeline([
    ('step1',step1),
    ('step2',step2)
])

pipe.fit(X_train,y_train)

y_pred = pipe.predict(X_test)
#print('R2 score',r2_score(y_test,y_pred))
#print('MAE',mean_absolute_error(y_test,y_pred))

print(X_train['HDD'])
import pickle
pickle.dump(df,open('df.pkl','wb'))
pickle.dump(pipe,open('pipe.pkl','wb'))
'''