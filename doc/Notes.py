
'ads' + range(1,20))

[f'ads_{x}' for x in range(1,20)]



# https://stackoverflow.com/questions/33907537/groupby-and-lag-all-columns-of-a-dataframe  

IIUC, you can simply use level="grp" and then shift by -1:

>>> shifted = df.groupby(level="grp").shift(-1)
>>> df.join(shifted.rename(columns=lambda x: x+"_lag"))
                col1 col2  col1_lag col2_lag
time       grp                              
2015-11-20 A       1    a         2        b
2015-11-21 A       2    b         3        c
2015-11-22 A       3    c       NaN      NaN
2015-11-23 B       1    a         2        b
2015-11-24 B       2    b         3        c
2015-11-25 B       3    c       NaN      NaN