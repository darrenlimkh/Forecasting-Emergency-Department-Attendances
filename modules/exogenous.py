import datetime as dt
import pandas as pd
import numpy as np
import holidays


def createExogenous(startYear=2010, endYear=dt.date.today().year):
  """ creates a dataframe of exogenous variables, used to input into the SARIMAX model
  :param startYear: starting year of reference
  :type startYear: int
  :param endYear: ending year of reference
  :type endYear: int
  ...
  :return: exogenous variables (e.g., public holidays, working days, day of the week)
  :rtype: pd.DataFrame
  """
  start = dt.date(startYear, 1, 1)
  end = dt.date(endYear, 12, 31)
  df = pd.DataFrame({"Date":pd.date_range(start=start, end=end+dt.timedelta(days=7), inclusive='both')})
  df.index=df['Date']

  #Initializing public holiday
  holsDf = pd.DataFrame(holidays.SG(years=list(range(startYear,endYear+1))).values(),index=holidays.SG(years=list(range(startYear,endYear+1))).keys(),columns=['Hols'])
  holsDf['Public_Holiday'] = holsDf['Hols'].str.replace(" \(Observed\)","")
  df = df.merge(holsDf['Public_Holiday'] ,how='left',left_index=True, right_index=True)
  df['Public_Holiday_NA'] = df['Public_Holiday']
  df["Public_Holiday_NA"]=df["Public_Holiday_NA"].fillna(method='ffill')
  df['Public_Holiday'] = df['Public_Holiday'].fillna(0)

  df["Holiday"]=np.where(df['Public_Holiday']!=0,1,df["Public_Holiday"])

  #Initializing weekday and work day
  df['Weekday']=df.index.strftime("%A")
  df['Month']=df.index.strftime("%B")
  df['Working_Day']=np.where((df["Holiday"]==0) & (df['Weekday'].isin(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'])),1,0)

  #Getting post holiday
  #Setting date variable
  df['Date'] = df.index
  df['Post_Holiday_1']=df['Holiday'].shift(1,fill_value=0)
  df['Post_Holiday_1']=np.where(df['Working_Day']==0,0,df['Post_Holiday_1'])

  #If PH is on Friday, the following Monday is Post Holiday if it's a working day
  PHFri=df.loc[(df['Holiday']==1)&(df['Weekday']=='Friday'),'Date']
  PHFri=pd.to_datetime(PHFri)+ dt.timedelta(days=3)
  PHFri=PHFri.astype(str).values.tolist()
  df['Post_Holiday_1']=np.where((df['Date'].isin(PHFri))&(df['Working_Day']==1),1,df['Post_Holiday_1'])

  #If PH is on Saturday, the following Monday is Post Holiday if it's a working day
  PHSat=df.loc[(df['Holiday']==1)&(df['Weekday']=='Saturday'),'Date']
  PHSat=pd.to_datetime(PHSat)+ dt.timedelta(days=2)
  PHSat=PHSat.astype(str).values.tolist()
  df['Post_Holiday_1']=np.where((df['Date'].isin(PHSat))&(df['Working_Day']==1),1,df['Post_Holiday_1'])

  df['Post_Holiday_2']=df['Post_Holiday_1'].shift(1,fill_value=0)
  df['Post_Holiday_3']=df['Post_Holiday_1'].shift(2,fill_value=0)

  #Create Post Holiday one hot encoding
  df['Post_Public_Holiday_1']=np.where(df['Post_Holiday_1']==1,"Post_"+df["Public_Holiday_NA"].astype(str),0)
  df['Post_Public_Holiday_2']=df['Post_Public_Holiday_1'].shift(1,fill_value=0)
  df['Post_Public_Holiday_3']=df['Post_Public_Holiday_1'].shift(2,fill_value=0)

  Post_Public_Holiday=pd.get_dummies(df['Post_Public_Holiday_1'],drop_first=True)
  df = df.merge(Post_Public_Holiday,how='left',left_index=True, right_index=True)

  #Create Holiday markers with one hot encoding
  Public_Holiday=pd.get_dummies(df['Public_Holiday'],drop_first=True)
  df = df.merge(Public_Holiday,how='left',left_index=True, right_index=True)

  #Create pre-holiday markers
  holsDf['In lieu']=np.where(holsDf['Hols'].str.contains(" \(Observed\)"),1,0)
  df = df.merge(holsDf['In lieu'] ,how='left',left_index=True, right_index=True)
  df['In lieu_filled']=df['In lieu'].fillna(0)

  #Pre-public Holiday
  df['Pre_Public_Holiday']=df['Public_Holiday'].shift(-1,fill_value=0)
  df['In lieu_2']=df['In lieu'].shift(-1,fill_value=0)
  df['Pre_Public_Holiday']=np.where((df["In lieu_2"]==1) & ((df["Pre_Public_Holiday"]!=0)),0,df["Pre_Public_Holiday"])
  df['Pre_Public_Holiday']=np.where(df['Pre_Public_Holiday']!=0,"Pre_"+df["Pre_Public_Holiday"].astype(str),0)
  Pre_Public_Holiday=pd.get_dummies(df['Pre_Public_Holiday'],drop_first=True)
  df = df.merge(Pre_Public_Holiday,how='left',left_index=True, right_index=True)
  df['Pre_Chinese_New_Year_2']=df['Pre_Chinese New Year'].shift(-1,fill_value=0)
  df['Pre_Chinese New Year']=np.where((df["Pre_Chinese_New_Year_2"]==0) & ((df["Pre_Chinese New Year"]==1)),0,df["Pre_Chinese New Year"])

  #Encoding
  wd=pd.get_dummies(df['Weekday']).drop(['Sunday'], axis=1)
  columnsNames = ['Chinese New Year','Christmas Day','Deepavali','Good Friday',
                  'Hari Raya Haji','Hari Raya Puasa','Labour Day','National Day',"New Year's Day",'Vesak Day',
                  'Pre_Chinese New Year','Pre_Christmas Day','Pre_Deepavali','Pre_Good Friday',
                  'Pre_Hari Raya Haji','Pre_Hari Raya Puasa','Pre_Labour Day','Pre_National Day',"Pre_New Year's Day",'Pre_Vesak Day',
                  'Post_Chinese New Year','Post_Christmas Day','Post_Deepavali','Post_Good Friday',
                  'Post_Hari Raya Haji','Post_Hari Raya Puasa','Post_Labour Day','Post_National Day',"Post_New Year's Day",'Post_Vesak Day',
                  'Working_Day']

  covDf = df[columnsNames]
  covDf = covDf.merge(wd,how='left',left_index=True, right_index=True)
  return(covDf)