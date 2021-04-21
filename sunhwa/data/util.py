import os
import pandas as pd
from collections import namedtuple
import boto3
from easydict import EasyDict as edict

cfg = edict()

#bucket
cfg.data_dir = 'aiavitality'

#sub_data_dir
cfg.applog_dir = 'data/applog/'
cfg.mbr_dir = 'data/member/'
cfg.mss_dir = 'data/mission/'

#page_code
cfg.pgcd = 'data/applog/page_code_info_210421.csv'

#party_id label
cfg.label = 'data/partyid_monthly_class.csv'

#fname 
def appfname(year, month):
    if month < 10:
        month = f'0{month}'
    return os.path.join(cfg.applog_dir, f'applog_{year}{month}.csv')

def mbrfname(year, month):
    if month < 10:
        month = f'0{month}'
    return os.path.join(cfg.mbr_dir, f'mbr_{year}{month}.csv')

def gmfname(year, month):
    if month < 10:
        month = f'0{month}'
    return os.path.join(cfg.mss_dir, f'goal_misn_{year}{month}.csv')

def load_file(year, month):
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
        obj = s3.get_object(Bucket=cfg.data_dir, Key=fname)
        data.append(pd.read_csv(obj['Body']))  
        print(f'{fname} is read')
    return data

def read_csv(key):
    s3 = boto3.client('s3')
    obj = s3.get_object(Bucket=cfg.data_dir, Key=key)
    return pd.read_csv(obj['Body'])



if __name__ == '__main__':
    
    import time
    start= time.time()
    data = load_file(cfg.gm3)

