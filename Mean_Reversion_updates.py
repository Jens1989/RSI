
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
from random import randint
import matplotlib.pyplot as plt
from datetime import datetime
from datetime import timedelta
import datetime


# In[3]:

def CryptoDataCSV(symbol, frequency):
    #Params: String symbol, int frequency = 300,900,1800,7200,14400,86400
    #Returns: df from first available date
    url ='https://poloniex.com/public?command=returnChartData&currencyPair='+symbol+'&end=9999999999&period='+str(frequency)+'&start=0'
    df1 = pd.read_json(url)
    df1.set_index('date',inplace=True)
    df1.to_csv(symbol + '.csv')
    print('Processed: ' + symbol)
    
def CryptoData(symbol, frequency):
    #Params: String symbol, int frequency = 300,900,1800,7200,14400,86400
    #Returns: df from first available date
    url ='https://poloniex.com/public?command=returnChartData&currencyPair='+symbol+'&end=9999999999&period='+str(frequency)+'&start=0'
    df = pd.read_json(url)
    return df

tickers = ['USDT_BTC','USDT_ETC','USDT_XMR','USDT_ETH','USDT_DASH',
 'USDT_XRP','USDT_LTC','USDT_STR','USDT_REP','USDT_ZEC']


for x in tickers:
    CryptoDataCSV(x,14400)


# In[17]:

BTC=retrieve_symbol_name('USDT_BTC')
BTC.index=pd.to_datetime(BTC.index)
BTC.head()


# In[11]:


ETC=modify_dates(retrieve_symbol_name('USDT_ETC'))
ETC.index[0]


# In[110]:

from datetime import datetime

diction={
       'USDT_BTC':'BTC',
       'USDT_ETC':'ETC',
       'USDT_ETH':'ETH',
       'USDT_LTC':'LTC',
       'USDT_REP':'REP',
       'USDT_STR':'STR',
       'USDT_ZEC':'ZEC',
       'USDT_XRP':'XRP',
       'USDT_XMR':'XMR',
       'USDT_DASH':'DASH' 
  }

def retrieve_symbol_name(coin):
    col=['date','close']
    df=pd.read_csv(coin+'.csv')[col]
    df = pd.DataFrame(df)
    df.columns = ['date',diction[coin]]
    df[diction[coin]+'_change']=df[diction[coin]].pct_change()
    df.set_index('date',inplace=True)
    
    return df



def decline(df,percent):
    
    df_decline=df[df.iloc[:,-1]>percent]
    return df_decline


def std_change(df, window):
    
    df = decline(df,2*df.iloc[:,-1].rolling(window).std())
    
    return df


def modify_dates(df):
    
    new_dates = []
    
    for i in df.index:
    
        new_dates.append(datetime.strptime(i, '%Y-%m-%d %H:%M:%S'))
        
    df.index = new_dates
     
    return df

def mod_dates(df):
    df.index=pd.to_datetime(df.index)
    return df


def clean_df(coin, window):
    
    df = std_change(modify_dates(retrieve_symbol_name(coin)),window)

    return df

def set_period(df,date):
    
    df=df[df.index>date]
    return df


def mean_rev(df, event_dates):
    df=modify_dates(df)
    list00=[]
    list0=[]
    list1=[]
    list2=[]
    list3=[]
    list4=[]
    List=[list00,list0,list1,list2,list3,list4]
    for y in event_dates:
        if y+timedelta(hours=200) < df.index[-1]:
                               list00.append(y)
                               list0.append(df.iloc[:,0].loc[y])
                               list1.append(df.iloc[:,0].loc[y+timedelta(hours=4)])
                               list2.append(df.iloc[:,0].loc[y+timedelta(hours=4*5)])
                               list3.append(df.iloc[:,0].loc[y+timedelta(hours=4*30)])
                               list4.append(df.iloc[:,0].loc[y+timedelta(hours=4*50)])
    return List



# In[12]:

def backtest(coin,window):
    df=mean_rev(retrieve_symbol_name(coin), clean_df(coin,window).index)
    df=pd.DataFrame(df)
    df=df.transpose()
    df.columns=['date','close','T+1','T+5','T+30','T+50']
    df.set_index('date',inplace=True)
    
    for i in range(1,5):
        
        df.iloc[:,i]=df.iloc[:,i]/df.iloc[:,0]
        df.iloc[:,i]=df.iloc[:,i]-1
    return df 


# In[255]:

def average_time(df):
    time=[]
    for i in range (len(df.index)):
        if i+1<len(df.index):
            time.append(df.index[i+1]-df.index[i])
    time=pd.DataFrame(time)        
    return time

average_time(BTC_18).mean()


# In[14]:

def up_days(df):
    T_1=[]
    T_5=[]
    T_30=[]
    T_50=[]
    
    T_1=df.iloc[:,1][df.iloc[:,1]>=0]
    T_5=df.iloc[:,2][df.iloc[:,2]>=0]
    T_30=df.iloc[:,3][df.iloc[:,3]>=0]
    T_50=df.iloc[:,4][df.iloc[:,4]>=0]
    up=[T_1,T_5,T_30,T_50]
    up=pd.DataFrame(up)
    up=up.transpose()
    
    mean=[]
    for x in range(len(up.columns)):
        mean.append(up.iloc[:,x].mean())
        
    diction={}
    keys = up.columns.tolist()
    
    for num, value in enumerate(mean):
            diction[keys[num]] = value
    
    
    return up, diction


def down_days(df):
    T_1=[]
    T_5=[]
    T_30=[]
    T_50=[]
    
    T_1=df.iloc[:,1][df.iloc[:,1]<0]
    T_5=df.iloc[:,2][df.iloc[:,2]<0]
    T_30=df.iloc[:,3][df.iloc[:,3]<0]
    T_50=df.iloc[:,4][df.iloc[:,4]<0]
    down=[T_1,T_5,T_30,T_50]
    down=pd.DataFrame(down)
    down=down.transpose()
    mean=[]
    for x in range(len(down.columns)):
        mean.append(down.iloc[:,x].mean())
        
    
    diction={}
    keys = down.columns.tolist()
    
    for num, value in enumerate(mean):
            diction[keys[num]] = value
    

    return down, diction

def count(df):
    down1=[]
    up=[]
    for i in range (len(df.columns)-1):
        down1.append(len(down_days(df)[0].iloc[:,i].dropna()))
        up.append(len(up_days(df)[0].iloc[:,i].dropna()))
    
    diction = {}
    keys = df.columns[1:].tolist()
    
    for num, value in enumerate(up):
        
        diction[keys[num]] = value/len(df)
    
    return diction



# In[120]:

def stats(df):#requires backtest function
    Stats=pd.DataFrame()
    Stats['T+1']=df['T+1'].map(float).describe()
    Stats['T+5']=df['T+5'].map(float).describe()
    Stats['T+30']=df['T+30'].map(float).describe()
    Stats['T+50']=df['T+50'].map(float).describe()
    Stats=Stats.transpose()
    Stats['Sharpe']=Stats['mean']/Stats['std']
    Stats['Win%']=count(df).values()
    Stats['Loss%']=1-Stats['Win%']
    Stats['Avg_Win']=up_days(df)[1].values()
    Stats['Avg_Loss']=down_days(df)[1].values()
    Stats['Expected_value']=Stats['Win%']*Stats['Avg_Win']+Stats['Loss%']*Stats['Avg_Loss']
    return Stats

XRP=backtest('USDT_XRP',100)
BTC=backtest('USDT_BTC',10)

BTC_18=set_period(BTC,'2018-01-01 00:00:00')
BTC_18.head()

bench=set_period(retrieve_symbol_name('USDT_BTC'),'2018-01-01 00:00:00')

def cumulative_returns(df,period,bench):
    returns = [df.iloc[:,period][0]]
    for num, r in enumerate(df.iloc[:,period][1:]):
        returns.append(r+returns[num])
    b_returns = [bench.iloc[:,1][0]]
    for num, r in enumerate(bench.iloc[:,1][1:]):
        b_returns.append(r+b_returns[num])
    b_returns=pd.DataFrame(b_returns)
    b_returns.columns=['Benchmark']
    col=df.columns.tolist()[period]    
    returns=pd.DataFrame(returns)
    returns.columns=[col]
    returns.plot()
    b_returns.plot()
    plt.show()
    return 

#cumulative_returns(BTC_18,1,bench)


# In[115]:

x=retrieve_symbol_name('USDT_BTC')
x['rolling']=x['BTC_change'].rolling(50).std()
x['BTC_change'].tail(1000).plot()
x['rolling'].tail(1000).plot(figsize=(20,10))
plt.show()


# In[119]:

BTC_18


# In[117]:

stats(BTC_18)


# In[118]:

stats(BTC)


# In[108]:

BTC['T+30'].hist(figsize=(20,10), bins=10)
plt.show()
BTC


# In[109]:

BTC


# In[107]:

def plot(df):
    df['T+1'].plot(label='T+1')
    df['T+1'].plot(label='T+5')
    df['T+30'].plot(label='T+30')
    df['T+50'].plot(label='T+50',figsize=(20,10))
    plt.show()
plot(BTC_18)


# In[134]:

btc=retrieve_symbol_name('USDT_BTC')
mod_dates(btc)
BTC_W=btc.asfreq('W').ffill()
def rolling(df,win):
    
    df['change']=df.iloc[:,0].pct_change()
    df['rolling']=df.iloc[:,-1].rolling(win).std()
    df.iloc[:,-2].plot()
    df.iloc[:,-1]=df.iloc[:,-1]
    df.iloc[:,-1].plot(figsize=(20,10))
    plt.show()
    return df


# In[135]:

rolling(BTC_W,10)


# In[105]:

BTC_W


# In[ ]:



