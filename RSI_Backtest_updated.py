
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from random import randint

from datetime import datetime
from datetime import timedelta
import datetime
import xlsxwriter

import talib
import matplotlib.pyplot as plt
from matplotlib import style
from scipy import stats
ff


# In[164]:


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


# In[120]:


def retrieve_symbol_name(coin,window):
    col=['date','close']
    df=pd.read_csv(coin+'.csv',index_col='date')['close']
    df=pd.DataFrame(df)
    
    if coin in tickers:
        df.columns=[coin[5:]]
        df[coin[5:]+'_change']=df[coin[5:]].pct_change()
    else:
        df.columns=[coin[5:]]
        df[coin[5:]+'_change']=df[coin[5:]].pct_change()
        
    df['RSI']=talib.RSI(df[coin[5:]], timeperiod=window)
    df['RSI_9']=df['RSI'].rolling(9).mean()
    df['RSI_50']=df['RSI'].rolling(50).mean()
    df.index=pd.to_datetime(df.index)
    df=df.dropna()
    return df


def decline(coin, RSI_min, window):
    df=retrieve_symbol_name(coin,window)
    df_decline=df[df['RSI']<RSI_min]
    return df_decline

def rise(coin, RSI_max, window):
    df=retrieve_symbol_name(coin,window)
    df_rise=df[df['RSI']>RSI_max]
    return df_rise

def set_period(df,date): 
    df=df[df.index>date]
    return df


def dates_v2(date,df):
    list0=[]
    for i in df.index:
        list0.append(i-date)
    list0=pd.DataFrame(list0)
    list0=list0[list0>0]
    list0=list0.dropna()
    list0.columns=['dates']
    list0.sort_values(by=['dates'])
   
    return list0.iloc[0,:]


def time_diff(coin,RSI_min,RSI_max,window,direction):
    time=[]
    fall=decline(coin,RSI_min,window)
    ascent=rise(coin,RSI_max,window)
    zero=np.zeros(1,)
    if len(ascent)<1:
        ascent
    if direction=='long':
        if len(ascent)>0:
            fall=fall.ix[fall.index<ascent.index[-1]]
            for x in range (len(fall)):
                time.append(dates_v2(fall.index[x],ascent))
    else:
        if len(fall)>0:
            ascent=ascent.ix[ascent.index<fall.index[-1]]
            for i in range (len(ascent)):
                time.append(dates_v2(ascent.index[i],fall))
    return time

def retrieve_dates(event_dates,time_diff):
    
    dates=[]
    time=[]
 
    List=[dates,time]
    for x,y in zip(event_dates,time_diff):
        dates.append(x)
        time.append(y)
        
    List=pd.DataFrame(List)
    List=List.transpose()
    List.columns=['entry_date','time']
    new_time=[]
    for i in range(len(List)):
        new_time.append(List['time'][i][0])
    List['time_shift']=new_time
    List=List.drop(['time'],axis=1)
    exit=[]
    for x,y in zip(List['time_shift'],List['entry_date']):
        exit.append(x+y)
    List['exit_date']=exit
    List.index=List['entry_date']
    List=List.drop(['entry_date'],axis=1)
    return List





# In[122]:


#df=retrieve_dates(decline('USDT_BTC',5,14).index, time_diff('USDT_BTC',5,70,14,'short'))

#df

fall=decline('USDT_BTC',5,14)
ascent=rise('USDT_BTC',70,14)
fall
time_diff('USDT_BTC',25,70,14,'short')


# In[165]:


def RSI_test(coin,window,RSI_min,RSI_max,period,direction):
    if direction =='long':
        df=retrieve_dates(decline(coin,RSI_min,window).index, time_diff(coin,RSI_min,RSI_max,window,direction))
    else:
        df=retrieve_dates(rise(coin,RSI_max,window).index, time_diff(coin,RSI_min,RSI_max,window,direction))
    
    if len(df) >0:
        df=set_period(df,period)
    
    entry_price=[]
    for i in df.index:
        entry_price.append(retrieve_symbol_name(coin,window).loc[i][coin[5:]])
    exit_price=[]
    for x in df['exit_date']:
        exit_price.append(retrieve_symbol_name(coin,window).loc[x][coin[5:]])
    List=[entry_price,exit_price]
    List=pd.DataFrame(List)
    List=List.transpose()
    List.columns=['entry','exit']
    List.index=df.index
    List['returns']=(List['exit']/List['entry'])-1
    time_shift=[]
    for x in df['time_shift']:
        time_shift.append(x)
    List['duration']=time_shift
    
    if direction=='long':
        List['returns']=List['returns']
    else:
        List['returns']=List['returns']*-1
    
    List['cumulative_returns']=List['returns'].cumsum()
    return List

RSI_test('USDT_BTC',3,5,60,'2018-01-01 00:00:00','long')


# In[187]:


def RSI_test_update(coin,window,RSI_min,RSI_max,capital,n,period,direction):
    
    if direction =='long':
        df=retrieve_dates(decline(coin,RSI_min,window).index, time_diff(coin,RSI_min,RSI_max,window,direction))
    else:
        df=retrieve_dates(rise(coin,RSI_max,window).index, time_diff(coin,RSI_min,RSI_max,window,direction))
    
    
    if len(df) >0:
        df=set_period(df,period)
    pos=1
    entry_date=[]
    exit_date=[]
    time_shift=[]
    
    entry_price=[]
    exit_price=[]
    returns=[]
    cash=[capital]
    PnL=[]
    positions=[pos]
    early_dates=[]
    
    for index,row in df.iterrows():
        if 1==1:#cash[-1]>capital/n:
            entry_date.append(index)
            exit_date.append(row[1])
            time_shift.append(row[0])
            
            
            entry__price=retrieve_symbol_name(coin,window).loc[index][coin[5:]]
            exit__price=retrieve_symbol_name(coin,window).loc[row[1]][coin[5:]]
            ret=(exit__price/entry__price)-1
            
           
            profit=(capital/n)*ret
            if direction=='long':
                profit=profit
            else:
                profit=profit*-1
            
            
            if len(entry_date)>=2:
                if entry_date[-1]<exit_date[-2]:
                    early_dates.append(entry_date[-1])
                
            updated_cash=cash[-1]
                   
            
            if index in early_dates:
                updated_cash=updated_cash-capital/n
            else:
                updated_cash=updated_cash+profit
            
            
            if len(entry_date)>=2:
                if entry_date[-1]>exit_date[-2]:
                    pos=1
                else:
                    pos=pos+1
            
            if len(entry_date)>=2:
                if positions[-1]>=2:
                    if entry_date[-1]>exit_date[-2]:
                        updated_cash=updated_cash+(capital/n)*(positions[-1]-1)+sum(PnL[-positions[-1]+1:])
            
            
            positions.append(pos)
            entry_price.append(entry__price)
            exit_price.append(exit__price)
            returns.append(ret)
            PnL.append(profit)
            
            cash.append(updated_cash)
            
    cash=cash[1:len(cash)]
    positions=positions[1:len(positions)] 
    List=[entry_date,exit_date,entry_price,exit_price,time_shift,returns,cash,PnL]
    col= ['entry_date','exit_date','entry_price','exit_price','duration','returns','capital','PnL']
    List=pd.DataFrame(List)
    List=List.transpose()
    List.columns=col
    List.index=List['entry_date']
    List=List.drop(['entry_date'],axis=1)
    
    List['cumulative_PnL']=List['PnL'].cumsum()
    List['positions']=positions
   # List=List.dropna()
    if direction=='long':
        List['returns']=List['returns']
    else:
        List['returns']=List['returns']*-1
    
    trades=[]
    for x in range(len(List['exit_price'])):
        trades.append(len(List.exit_price[0:x].unique()))
    List['trades']=trades
    zero=np.zeros((1, 10))
    #zero[:10]=1
    #zero=pd.DataFrame(zero)
    #List=List.append(zero)
    if len(List)<1:
        List=pd.DataFrame(zero,columns=List.columns)
        
    return List


df=RSI_test_update('USDT_BTC',14,25,45,1000,10,'2017-01-01 00:00:00','long')

df


# In[184]:


ten_ninety_short=df


# In[183]:


df2=RSI_test_update('USDT_ETH',14,25,55,1000,10,'2018-01-01 00:00:00','long')

df2


# In[18]:


bench_18=set_period(retrieve_symbol_name('USDT_BTC',14),'2018-01-01 00:00:00')
bench=set_period(retrieve_symbol_name('USDT_BTC',14),'2015-01-01 00:00:00')


BTC=RSI_test_update('USDT_BTC',14,35,65,1000,10,'2015-01-01 00:00:00','short')
BTC

BTC_18=RSI_test_update('USDT_BTC',14,35,65,1000,10,'2018-01-01 00:00:00','short')
BTC


# In[180]:


def cumulative_returns(df,bench):
    df['Cumulative_returns']= df['returns'].cumsum()
    bench['Benchmark'] = bench.iloc[:,1].cumsum()
    df['Cumulative_returns'].plot(figsize=(10,5))
    bench['Benchmark'].plot(figsize=(10,5))
    plt.show()
    return 

def stats_v2(coin,window,RSI_min,RSI_max,capital,n,period,direction):#requires backtest function
    df=RSI_test_update(coin,window,RSI_min,RSI_max,capital,n,period,direction)
    Stats=pd.DataFrame()
    Stats[coin[5:]]=df['returns'].map(float).describe()
    Stats=Stats.transpose()
    Time_shift=[]
    for x in df['duration']:
        Time_shift.append(x)
    if Stats['count'][0]>1:
        Stats['Sharpe']=Stats['mean']/Stats['std'] 
        Time=pd.DataFrame(Time_shift)
        x=Time.mean()
        Stats['Average_time']=x[0]
        x=df[df['returns']>0]
        Stats['win%']=len(x)/len(df)
        Stats['trades']=df['trades'][-1]
    
    zero=np.zeros((1, 8))
    if len(Stats)<1:
        Stats['std']=0
        
        #Stats=pd.DataFrame(zero,columns=Stats.columns)
    #Stats['std']=0
    return Stats

coins=['USDT_ETC','USDT_XMR','USDT_ETH','USDT_DASH','USDT_XRP','USDT_LTC','USDT_STR','USDT_REP','USDT_ZEC']

def collective_stats(window,RSI_min,RSI_max,capital,n,period,direction):
    stats=[]
    df=stats_v2('USDT_BTC',window,RSI_min,RSI_max,capital,n,period,direction)
    
    for coin in coins:
        df=df.append(stats_v2(coin,window,RSI_min,RSI_max,capital,n,period,direction))
    df=df.dropna()
    return df

def optimal_window(RSI_min,RSI_max,period,direction):
    av_returns=[]
    btc=[]
    alts=[]
    window=[]
    trades=[]
    count=[]
    for i in range(3,15):
        df=collective_stats(i,RSI_min,RSI_max,1000,10,period,direction)
        av_returns.append(df.iloc[:,1].mean())
        btc.append(df.iloc[0,1])
        alts.append(df.iloc[1:,1].mean())
        window.append(i)
        trades.append(df['trades'].mean())
        count.append(df['count'].mean())
    df=pd.DataFrame(av_returns,index=window,columns=['Average_Return'])
    df['BTC']=btc
    df['Alts']=alts
    df['Count']=count
    df['Trades']=trades    
    return df

def optimal_RSI(window,RSI_min,RSI_max,period,direction,boundd):
    av_returns=[]
    btc=[]
    alts=[]
    bound=[]

    trades=[]
    count=[]
    if boundd=='low':
        for x in range(5,40,5):
            df=collective_stats(window,x,RSI_max,1000,10,period,direction)
            if len(df.columns)>8:
                av_returns.append(df.iloc[:,1].mean())
                btc.append(df.iloc[0,1])
                alts.append(df.iloc[1:,1].mean())
                bound.append(x)
                trades.append(df['trades'].mean())
                count.append(df['count'].mean())
    else:
        for x in range(65,100,5):
            df=collective_stats(window,RSI_min,x,1000,10,period,direction)
            if len(df.columns)>8:
                av_returns.append(df.iloc[:,1].mean())
                btc.append(df.iloc[0,1])
                alts.append(df.iloc[1:,1].mean())
                bound.append(x)
                trades.append(df['trades'].mean())
                count.append(df['count'].mean())
    df=pd.DataFrame(av_returns,index=bound,columns=['Average_Return'])
    df['BTC']=btc
    df['Alts']=alts
    df['Count']=count
    df['Trades']=trades 
    
    return df

def vary_RSI(window,period,direction,boundd):
    RSI=[]
    keys=[]
    if boundd=='up':
        for i in range(5,40,5):
            df=optimal_RSI(window,i,70,period,direction,'up')
            RSI.append(df)
            keys.append(i)
             
    else:
        for i in range(65,100,5):
            df=optimal_RSI(window,30,i,period,direction,'low')
            RSI.append(df)
            keys.append(i)
       
    dicts = {}
    values = RSI
    for x,i in enumerate(keys):
        dicts[i] = values[x]
    return dicts
            

def excel(df):
    writer = pd.ExcelWriter('RSI_3.xlsx',engine='xlsxwriter')   # Creating Excel Writer Object from Pandas  
    workbook=writer.book
    i=0
    for i in df.keys():
        df[i].to_excel(writer,sheet_name='Validation',startrow=i , startcol=0) 
        i=i+12
    writer.save()
    
df2=vary_RSI(5,'2018-01-01 00:00:00','short','low')
df2


# In[177]:


df=vary_RSI(7,'2018-01-01 00:00:00','long','up')
df


# In[163]:


df2


# In[162]:


def excel(df):
    writer = pd.ExcelWriter('RSI_3.xlsx',engine='xlsxwriter')   # Creating Excel Writer Object from Pandas  
    workbook=writer.book
    x=0
    for i in df.keys():
        df[i].to_excel(writer,sheet_name='Validation',startrow=x , startcol=0) 
        x=x+8
    writer.save()
excel(df2)


# In[186]:


p=optimal_RSI(14,20,85,'2018-01-01 00:00:00','long','up')
p


# In[23]:


v


# In[43]:


n


# In[29]:


x=pd.DataFrame(v[65])
2+2


# In[27]:



excel(n)
    


# In[129]:


vv=vary_RSI(3,30,85,'2018-01-01 00:00:00','long','up')


# In[144]:


v


# In[108]:


df1=optimal_RSI(3,30,70,'2018-01-01 00:00:00','short','low')


# In[116]:


df1


# In[110]:


df2=optimal_RSI(3,30,70,'2018-01-01 00:00:00','short','up')
df2


# In[112]:


df3=optimal_RSI(3,20,70,'2018-01-01 00:00:00','short','up')
df3


# In[79]:


optimal_window(30,70,1000,10,'2018-01-01 00:00:00','short')


# In[80]:


optimal_window(30,70,1000,10,'2018-01-01 00:00:00','long')


# In[45]:


df=collective_stats(3,15,75,1000,10,'2018-01-01 00:00:00','long')
df['mean'].mean()


# In[172]:


def visualize(coin,window,RSI_min,RSI_max,period,direction):
    style.use("ggplot")
    price=set_period(retrieve_symbol_name(coin,window),period)
    trades=RSI_test(coin,window,RSI_min,RSI_max,period,direction)
    entry=pd.DataFrame(trades['entry'])
    exit=pd.DataFrame(trades['exit'])
    #price[coin[5:]].plot(figsize=(20,10))
    plt.plot(price[coin[5:]],label=coin[len(coin)-3:])
    plt.plot(entry,'o', color='yellow',label='entry')
    plt.plot(exit,'+',color='black',label='exit')
    plt.legend(loc=1)
    plt.rcParams["figure.figsize"]=[16,9]
    plt.show()
    
    
    


# In[174]:


visualize('USDT_ETH',3,10,90,'2018-01-01 00:00:00','long')


# In[ ]:


stats.normaltest(retrieve_symbol_name('USDT_BTC',14)['BTC'])


# In[101]:


df=retrieve_symbol_name('USDT_BTC',14)['BTC_change']
df


# In[94]:


import numpy as np
import statsmodels.api as sm
import pylab

test = np.random.normal(0,1, 1000)

sm.qqplot(test,line='45')
pylab.show()


# In[81]:


gold=pd.read_csv('GOLD_DAILY.csv')
gold['RSI']=talib.RSI(gold['close'])
gold=gold.dropna()
gold.tail()


# In[138]:


df=retrieve_symbol_name('GOLD_DAILY',14)
df.index


# In[16]:


df=RSI_test('GOLD_DAILY',14,35,65,'2015-01-01 00:00:00','long')


# In[100]:


df_dates=retrieve_dates(retrieve_symbol_name('GOLD_DAILY',14), decline('GOLD_DAILY',35,14).index,time_diff('GOLD_DAILY',35,60,14,'long'))


# In[145]:


time_diff('GOLD_DAILY',30,70,14,'long')


# In[113]:


df['returns']=df['adj close'].pct_change()
df=df.dropna()
sm.qqplot(df['returns'],line='45')


# In[101]:


df_dates


# In[96]:


gld=retrieve_symbol_name('GOLD_DAILY',14)


# In[ ]:


def ff(df):
    


# In[6]:


#standard deviatio with RSI??
df=retrieve_symbol_name('USDT_BTC',14)
df['RSI_change']=df['RSI'].pct_change()
df.dropna()
#df['RSI_change'].tail(100).plot()
df['RSI_change'].tail(100).rolling(10).std().plot()
plt.show()


#RSI MA
df=retrieve_symbol_name('USDT_BTC',14)
df.tail(20)
style.use("ggplot")
#df['RSI'].tail(100).plot(figsize=(20,10))
#df['RSI_9'].tail(100).plot()
#df['RSI_50'].tail(100).plot()
plt.plot(df['RSI'].tail(300),label='RSI')
plt.plot(df['RSI_9'].tail(300), color='green',label='9')
plt.plot(df['RSI_50'].tail(300),color='black',label='50')
plt.legend(loc=1)
plt.rcParams["figure.figsize"]=[16,9]
plt.show()


# In[162]:


def discovery(df):
    df=df[df['returns']<0]
    return df

def vis_errors(df,coin,window,period):
    price=retrieve_symbol_name(coin,window)
    price=set_period(price,period)
    entry=df['entry_price']
    plt.plot(price[coin[5:]],label=coin[5:])
    plt.plot(entry,'o', color='black',label='entry')
    plt.show()
df=discovery(df)
vis_errors(df,'USDT_BTC',14,'2018-01-01 00:00:00')

