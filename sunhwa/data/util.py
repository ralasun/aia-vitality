import os
import pandas as pd
from collections import namedtuple
import boto3

#bucket
data_dir = 'aiavitality'

#sub_data_dir
applog_dir = 'data/applog/'
mbr_dir = 'data/member/'
mss_dir = 'data/mission/'

#fname 
def appfname(year, month):
    if month < 10:
        month = f'0{month}'
    return os.path.join(applog_dir, f'applog_{year}{month}.csv')

def mbrfname(year, month):
    if month < 10:
        month = f'0{month}'
    return os.path.join(mbr_dir, f'mbr_{year}{month}.csv')

def gmfname(year, month):
    if month < 10:
        month = f'0{month}'
    return os.path.join(mss_dir, f'goal_misn_{year}{month}.csv')

def load_file(year, month, bucket=data_dir):
    s3 = boto3.client('s3')
    
    #applog
    applog = appfname(year, month)
    #member
    mbr = mbrfname(year, month)
    #goal_mission
    gm = gmfname(year, month)
    
    data = []
    for fname in [applog, mbr, gm]:
        print(fname)
        obj = s3.get_object(Bucket=data_dir, Key=fname)
        data.append(pd.read_csv(obj['Body']))  
        print(f'{fname} is read')
    return data



if __name__ == '__main__':
    import time
    start= time.time()
    data = load_file(cfg.gm3)
