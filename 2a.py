# TODO popuniti kodom za problem 2a
import pandas as pd
import tensorflow as tf 
import numpy as np

#ucitavamo prvi fajl
filename = 'jeftini_senzori.TXT'
data = pd.read_csv(filename, sep=";", header=None)
df = pd.DataFrame(data)

#inicijalizujemo kolone i menjamo format datuma i vremena

df.columns = ['node', 'date', 'hour', 'energy','temperature', 'humidity', 'NO2', 'PM1', 'lowcost_pm2.5','PM10', 'Pm1_std','PM2,5 std','PM10 std', 'latituda' ,'longituda']
df['date'] = pd.to_datetime(df['date'])
df['hour'] = pd.to_datetime(df['hour'])
df['date'] = df['date'].dt.strftime("%d/%m/%Y")
df['hour'] = df['hour'].dt.strftime("%H:%M")

#brisemo nepotrebne kolone i spajamo datum i vreme u jednu novu kolonu

del df["node"], df["energy"],df["NO2"], df["PM1"],df["PM10"], df["Pm1_std"],df["PM2,5 std"], df["PM10 std"],df["latituda"], df["longituda"]
df['measurement_datetime'] = df['date'].map(str) + ' ' + df['hour'].map(str)
cols = list(df.columns.values)
cols = cols[-1:] + cols[:-1]
df = df[cols]
del df["date"], df["hour"]

#posto smo zavrsili sa pravljenjem prve tabele, prelzimo na ucitavanje drugog fajla

filename = 'skupi_senzori.XLS'
data = pd.read_csv(filename,skiprows = [0, 1, 2, 3,4, 18056, 18057, 18058, 18059, 18060, 18061],sep="	", header=None)
df1 = pd.DataFrame(data)
df1.columns = ['date beginning', 'time beginning', 'date end', 'time end','relative time [s]', 'PM1_ambient - #11', 'reference_pm2.5', 'PM10_ambient - #11', 'PM1_ambient - #11']
del df1["relative time [s]"], df1["PM1_ambient - #11"],df1["PM10_ambient - #11"]


df1['time beginning'] = pd.to_datetime(df1['time beginning'])
df1['time end'] = pd.to_datetime(df1['time end'])


df1['time end'] = df1['time end'].dt.strftime("%H:%M:%S")
df1['time beginning'] = df1['time beginning'].dt.strftime("%H:%M:%S")

df1['datetime_reference_end'] = df1['date end'].map(str) + ' ' + df1['time end'].map(str)
cols = list(df1.columns.values)
cols = cols[-1:] + cols[:-1]
df1 = df1[cols]

df1['datetime_reference_beginning'] = df1['date beginning'].map(str) + ' ' + df1['time beginning'].map(str)
cols1 = list(df1.columns.values)
cols1 = cols1[-1:] + cols1[:-1]
df1 = df1[cols1]

del df1["date beginning"], df1["time beginning"],df1["date end"],df1["time end"]
#03/10/2019 23:59	

#menjamo vremensku zonu novonastalih kolona i postavljamo odgovarajuci format

df1['datetime_reference_beginning'] = pd.to_datetime(df1['datetime_reference_beginning'], format="%d/%m/%Y %H:%M:%S")
df1['datetime_reference_end'] = pd.to_datetime(df1['datetime_reference_end'], format="%d/%m/%Y %H:%M:%S")

df1['datetime_reference_beginning'] = df1['datetime_reference_beginning'] + pd.DateOffset(hours=-2)
df1['datetime_reference_end'] = df1['datetime_reference_end'] + pd.DateOffset(hours=-2)

df1['datetime_reference_beginning'] = df1['datetime_reference_beginning'].dt.strftime("%d/%m/%Y %H:%M")
df1['datetime_reference_end'] = df1['datetime_reference_end'].dt.strftime("%d/%m/%Y %H:%M")

#pravimo uslovni(pomocni) dataFrame koji cemo kasnije zalepiti na prvu tabelu i dobiti rez

uslovni_df = pd.DataFrame()
uslovni_df = uslovni_df.append(df, ignore_index = True)
uslovni_df.columns = ["datetime_reference_beginning","datetime_reference_end","reference_pm2.5", "d"]
del uslovni_df["d"]

k = 0
for i in range(0,18049):
    
    #proveravamo da li nam se datum i vreme iz prve tabele nalaze u intervalu izmedju pocetka i kraja druge
    while ((pd.to_datetime(df.measurement_datetime[k]) > pd.to_datetime(df1.datetime_reference_beginning[i])) & (pd.to_datetime(df.measurement_datetime[k]) <= pd.to_datetime(df1.datetime_reference_end[i]))) :    
        #u sllucaju da je ispunjen uslov, postavljamo odgovarajuce vrednosti u nas pomocni data frame
        uslovni_df['datetime_reference_beginning'][k] = df1['datetime_reference_beginning'][i]
        uslovni_df['datetime_reference_end'][k]= df1['datetime_reference_end'][i]
        uslovni_df['reference_pm2.5'][k] = df1['reference_pm2.5'][i]
        k = k+1
        if(k == 21594):
            #izvrsavanje poslednje iteracije
            uslovni_df['datetime_reference_beginning'][k] = df1['datetime_reference_beginning'][i]
            uslovni_df['datetime_reference_end'][k]= df1['datetime_reference_end'][i]
            uslovni_df['reference_pm2.5'][k] = df1['reference_pm2.5'][i]
            break

#finalna konkatenacija i dobijanje rezultata
result = pd.concat([df, uslovni_df], axis=1)            
       

