import pandas as pd
from datetime import datetime
from collections import namedtuple
from util import cfg, load_file, read_csv

import time
import numpy as np
import pickle
import pdb

class SeqDataset():
    
    def __init__(self, year, month, maxlen=4):
        self.year = year
        self.month = month
        self.maxlen = maxlen
        
        print('데이터 셋 로딩중..')
        datelist = self.create_seq_dataset()
        print(datelist)
        print('데이터 셋 로딩 완료')
        print()
        
    def preprocess(self):

        appds = [data[0] for data in self.dataset]
        mbrds = [data[1] for data in self.dataset]
        gmds = [data[2] for data in self.dataset]
        pcd = read_csv(cfg.pgcd)
        
        print('applog 전처리중')
        appdf = prep_applogs(appds)
        pcd, code2name = prep_pagecd(pcd)
        appdf = merge_app_and_pcd(appdf, pcd)
        print('applog 전처리 완료')
        print()
        
        print('member 전처리중')
        mbrdf = prep_mbrlogs(mbrds)
        print('member 전처리 완료')
        print()
        
        return appdf, mbrdf
        
    def create_seq_dataset(self):
        """
        year:현재년도
        month:현재월
        maxlen:시계열에서 고려하는 개월 수 
        ex) 현재 개월수가 11월이면, 11월/10월/9월/8월을 고려함
        dataset: [[app, mbr, gm], [app, mbr, gm],...]
        """
        datelist = self.list_dates()
        dataset = self.load_dataset(datelist)
        self.dataset = dataset
        return datelist
    
    def load_dataset(self, datelist):
        data = []
        for date in datelist:
            data.append(load_file(date.year, date.month))

#         #임시
#         data = []
#         with open('sample_dataset0.pickle', 'rb') as f:
#             data.append(pickle.load(f))
#         with open('sample_dataset1.pickle', 'rb') as f:
#             data.append(pickle.load(f))
        return data
        
    def list_dates(self):
        mydate = namedtuple('mydate', ['year', 'month'])
        curdate = mydate(self.year, self.month)
        datelist = [curdate]
        for i in range(1, self.maxlen):
            year = curdate.year
            month = curdate.month - i
            if year == 2021 and month <= 0:
                year = 2020
                month += 12
            elif year == 2020 and month <= 0:
                return datelist       
            datelist.append(mydate(year, month))
        return datelist
    
def prep_applogs(applogs):
    df = pd.concat(applogs, axis=0)
    
    #null제거
    print('Orig. data len:', len(df))
    df = df.dropna()
    print('Aft. drop-nan:', len(df))
    
    #방문일시 변경
    vst_dtm = df['vst_dtm'].astype('str')
    f = lambda x: x[:-3]
    vst_dtm = vst_dtm.apply(f)
    vst_dtm = pd.to_datetime(vst_dtm, format='%Y%m%d%H%M%S')
    df['vst_dtm'] = vst_dtm
    
    #필요 없는 칼럼 drop
    df = df.drop(['login_yn', 'new_vst_yn', 'tlcom_co_cd'], axis=1)
    
    #1970년대 데이터 제외
    df = df[df['vst_dtm'].dt.year != 1970]
    df = df.reset_index(drop=True)

    #session_id '#' 제거
    inds = np.where(df['sesn_id'] == '#')[0]
    df = df.drop(inds)
    
    #sorting
    df = df.sort_values(['party_id', 'vst_dtm', 'sesn_id'])
    return df

def prep_pagecd(pcd):
    pcd = pcd.reset_index(drop=True)
    pcd = pcd.drop(columns=['No'])
    
    code2name = {}
    for k, v in zip(pcd['page_cd'].values,  pcd['page_nm'].values):
        if pd.isnull(v):
            code2name[k]=k
        else:
            code2name[k]=v
    return pcd, code2name

def prep_mbrlogs(mbrlogs):
    
    def count_vtlt_age_eff_dt(x):
        count_vtlt_age = np.zeros(len(x['vtlt_age_eff_dt']), dtype=np.float32)
        vtlt_effs = np.unique(x['vtlt_age_eff_dt'])
        for eff in vtlt_effs:
            if eff == 99991231:
                continue
            else:
                ind = np.where(x['vtlt_age_eff_dt'] == eff)[0][0]
                count_vtlt_age[ind:] += 1
        return pd.Series(count_vtlt_age, name='count_vtlt_age')
    
    #concat
    mbrdf = pd.concat(mbrlogs, axis=0)
    
    #dt -> datetime 으로 변경
    mbrdf['dt'] = pd.to_datetime(mbrdf['dt'], format='%Y%m%d')
    
    #party_id당 dt순으로 sorting
    mbrdf = mbrdf.sort_values(['party_id', 'dt'])
    
    #사용안하는 컬럼 drop
    mbrdf = mbrdf.drop(columns=cfg.unused_mbrcol)
    
    #null제거
    print('Orig. data len:', len(mbrdf))
    mbrdf = mbrdf.dropna()
    print('Aft. drop-nan:', len(mbrdf))
    
    #party_id -> int형으로 변환
    mbrdf['party_id'] = mbrdf['party_id'].astype('int32')
    
    #바이탈리티 나이 측정 횟수 관련 전처리
    newcol = mbrdf.groupby(['party_id']).apply(count_vtlt_age_eff_dt)
    mbrdf['count_vtlt_age_dt'] = newcol.values
    
    #바이탈리티 나이 차이 관련 전처리
    mbrdf = mbrdf.reset_index(drop=True)
    inds = np.where(mbrdf['vtlt_age'] == 'NOT_ENOUGH_DATA')[0]
    mbrdf.loc[inds, 'vtlt_age'] = '0'
    mbrdf['vtlt_age'] = mbrdf['vtlt_age'].astype('int32')
    mbrdf['diff_age'] = mbrdf['vtlt_age'] - mbrdf['age']
    
    #주간미션달성률 관련 전처리
    mbrdf['achv_rat'] = mbrdf['cur_mbrsh_pd_goal_achv_cnt'] / mbrdf['cur_mbrsh_pd_goal_alct_cnt']
    
    #회원가입이후 경과일
    pids = np.unique(mbrdf.loc[mbrdf['mbr_scrb_dt'] == 99991231]['party_id'].values)
    newval = []
    for pid in pids:
        vals = np.unique(mbrdf.loc[mbrdf['party_id']==pid]['mbr_scrb_dt'].values)
        newval.append(vals[np.where(vals != 9991231)[0][0]])
    
    for val, pid in zip(newval, pids):
        inds = np.where(mbrdf['party_id'] == pid)[0]
        mbrdf.loc[inds, 'mbr_scrb_dt'] = val 
    
    mbrdf['mbr_scrb_dt'] = pd.to_datetime(mbrdf['mbr_scrb_dt'], format='%Y%m%d')
    mbrdf['active_dur'] = mbrdf['dt'] - mbrdf['mbr_scrb_dt']
    
    #멤버십 등급 -> 1,2,3,4로 변경
    mbrsh_dic = {'Bronze': 1, 'Silver': 2, 'Gold': 3, 'Platinum': 4}
    f = lambda x : mbrsh_dic[x]
    newmbrsh = mbrdf['cur_mbrsh_rwrd_st_cd'].transform(f)
    mbrdf['cur_mbrsh_rwrd_st_cd'] = newmbrsh
    
    #필요없는 칼럼 drop
    mbrdf = mbrdf.drop(columns=['vtlt_age_eff_dt', 'mbr_scrb_dt', 'cur_mbrsh_pd_goal_alct_cnt','cur_mbrsh_pd_goal_achv_cnt'])
    
    return mbrdf

def prep_gmlogs(gmlogs):
    
    gmdf = pd.concat(gmlogs, axis=0)
    gmdf = gmdf[['party_id', 'p_event_apl_dte','points_value','points_effective_dte']]
    #gmdf = gmdf[cfg.used_gmcol]
    gmdf = gmdf.replace('#', np.nan)
    
    print('Orig. data len:', len(gmdf))
    gmdf = gmdf.dropna()
    print('Aft. drop-nan:', len(gmdf))
    
    #party_id -> int
    gmdf['party_id'] = gmdf['party_id'].astype('int32')
    
    #datetime형으로 변환
    gmdf['p_event_apl_dte'] = pd.to_datetime(gmdf['p_event_apl_dte'], format='%Y%m%d')
    gmdf['points_effective_dte'] = pd.to_datetime(gmdf['points_effective_dte'], format='%Y%m%d')
    
    #sorting
    gmdf = gmdf[['party_id', 'p_event_apl_dte']]
    return gmdf

def merge_app_and_pcd(df, pcd):
    return pd.merge(left=df, right=pcd[['page_cd','menu_nm_1','menu_nm_2']], on=['page_cd'], how='left', sort=False)

def count_vtlt_age_eff_dt(x):
    count_vtlt_age = np.zeros(len(x['vtlt_age_eff_dt']), dtype=np.float32)
    vtlt_effs = np.unique(x['vtlt_age_eff_dt'])
    for eff in vtlt_effs:
        if eff == 99991231:
            continue
        else:
            ind = np.where(x['vtlt_age_eff_dt'] == eff)[0][0]
            count_vtlt_age[ind:] += 1
    return pd.Series(count_vtlt_age, name='count_vtlt_age')

if __name__ == '__main__':
    
    import pickle        
    seqds = SeqDataset(2021, 2, maxlen=4)
    df = seqds.preprocess()
    df[0].to_csv('s3://aiavitality/sunhwa/appdf212.csv')
    df[1].to_csv('s3://aiavitality/sunhwa/mbrdf212.csv')
    df[2].to_csv('s3://aiavitality/sunhwa/gmdf212.csv')