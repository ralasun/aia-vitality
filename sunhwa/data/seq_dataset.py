import pandas as pd
from datetime import datetime
from collections import namedtuple
from util import cfg, load_file, read_csv, gmfname, seqfname

import time
import numpy as np
import pickle
import os

import pdb

#second version -> 월마다 preprocess해서 합치기
class SeqPreProcess():
    
    def __init__(self, start, end=None):
        
        self.pcd = None
        self.lbl = None
        
        startmth = pd.to_datetime(start, format='%Y%m')
        self.date_ranges = None
        if end is not None:
            endmth = pd.to_datetime(end, format='%Y%m')
            self.date_ranges = pd.date_range(start=startmth, end=endmth, freq='M')
        else:
            self.date_ranges = [startmth]
        
    def preprocess(self):
        
        #page_code와 label 불러오기
        self.pcd, self.code2name = prep_pagecd(read_csv(cfg.pgcd))
        self.lbl = read_csv(cfg.label)   
        
        for date in self.date_ranges:
            print(f"### {date} 데이터 전처리 시작")
            merged_df = self._preprocess_per_mth(date)
            
            csvpath = os.path.join('s3://', cfg.data_dir, seqfname(date.year, date.month))
            merged_df.to_csv(csvpath)
            print(f'### {date} 데이터 전처리 완료')
            print()
            
        return merged_df
    
    def load_dataset(self, date):
        return load_file(date.year, date.month)
    
    def _preprocess_per_mth(self, date):
        
        appdf, mbrdf, gmdf = self.load_dataset(date)
        #goal_mission은 이전 달까지 추가로 불러야 함
        prev_date = date - pd.DateOffset(months=1)
        if prev_date == pd.to_datetime('20191231', format='%Y%m%d'):
            gmdf_prev = None
        else:
            gmdf_prev = read_csv(gmfname(prev_date.year, prev_date.month))
        
        print('label 전처리 중')
        lblmth = prep_lbl_per_mth(self.lbl, date)
        print('label 전처리 완료')
        print()
        
        print('applog 전처리 중')
        appdf = prep_applog_per_mth(appdf, self.pcd, lblmth)
        print('applog 전처리 완료')
        print()
        
        print('member 전처리 중')
        mbrdf = prep_mbrlog_per_mth(mbrdf)
        print('member 전처리 완료')
        print()
        
        print('goal_mission 전처리 중')
        if gmdf_prev is None:
            gmdflist = [gmdf]
        else:
            gmdflist = [gmdf_prev, gmdf]
        gmdf = pd.concat(gmdflist, axis=0)
        gmdf = gmdf.sort_values(['party_id','p_event_apl_dte'])
        gmdf = prep_gmlog_per_mth(gmdf)
        print('goal_mission 전처리 완료')
        print()
        
        print('세 데이터 합치는 중')
        merged_df = merge_app_and_mbr(appdf, mbrdf)
        merged_df = merge_app_and_gm(merged_df, gmdf)
        
        merged_df['gender_cd'] = merged_df['gender_cd'].astype('category').cat.codes
        merged_df['push_alarm_yn'] = merged_df['push_alarm_yn'].astype("category").cat.codes
        
        return merged_df
    
    def dates(self):
        return self.date_ranges    
        
class SeqDataSet():
    
    """create seq dataset after preprocess"""
    
    def __init__(year, month, maxlen=4):
        
        self.year = year
        self.month = month
        self.maxlen = maxlen
        
    def create_seq_dataset():
        return
    
    
def basic_prep_applog_per_mth(df, pcd, lblmth):
    
    #null제거
    print('Orig. data len:', len(df))
    df = df.dropna()
    print('Aft. drop-nan:', len(df))
    
    #방문일시 변경
    vst_dtm = df['vst_dtm'].astype('str')
    f = lambda x: x[:-3]
    vst_dtm = vst_dtm.apply(f)
    vst_dtm = pd.to_datetime(df['vst_dtm'], format='%Y-%m-%d %H:%M:%S')
    df['vst_dtm'] = vst_dtm
    
    #sesn_id, sty_tms drop -> new_sesn_id를 sesn_id로, new_sty_tms를 sty_tms로
    df['sty_tms'] = df['new_sty_tms']
    df['sesn_id'] = df['new_sesn_id']
    df = df.drop(columns=['new_sesn_id','new_sty_tms'])

    #1970년대 데이터 제외
    df = df[df['vst_dtm'].dt.year != 1970]
    df = df.reset_index(drop=True)

    #session_id '#' 제거
    inds = np.where(df['sesn_id'] == '#')[0]
    df = df.drop(inds)
    
    #'month'칼럼 & 'dt'칼럼 추가
    df['month'] = df['vst_dtm'].dt.to_period('M')
    df['dt'] = df['vst_dtm'].dt.to_period('D')
    
    #sorting
    df = df.sort_values(['party_id', 'vst_dtm', 'sesn_id'])
    
    df = merge_app_and_pcd(df, pcd)
    pdb.set_trace()
    df = merge_app_and_lbl(df, lblmth)
    
    return df

    
def prep_applog_per_mth(appdf, pcd, lblmth):
    
    appdf = basic_prep_applog_per_mth(appdf, pcd, lblmth)
    print(appdf.columns)
    
    print('Before appdf len', len(appdf))
    #1.menu _nm_1 == Nan or menu_nm_2 == Nan인 경우로만 이뤄진 session_id 제거하기
    ##nan이 포함된 전체 고유 party_id와 session_id갯수
    menusess1 = appdf[['party_id','sesn_id','page_cd']].groupby(['party_id','sesn_id']).first().reset_index()
    menusess1 = menusess1[['party_id','sesn_id']]
    
    ## nan이 제거된 전체 고유 party_id와 session_id갯수
    menusess2 = appdf[['party_id','sesn_id','menu_nm_1','page_cd']].groupby(['party_id','sesn_id','menu_nm_1']).count().reset_index()
    menusess2 = menusess2[['party_id','sesn_id']].drop_duplicates()
    menusess3 = appdf[['party_id','sesn_id','menu_nm_2','page_cd']].groupby(['party_id','sesn_id','menu_nm_2']).count().reset_index()
    menusess3 = menusess3[['party_id','sesn_id']].drop_duplicates()
    
    menusess = pd.concat([menusess1, menusess2], axis=0)
    menusess = menusess.loc[~menusess.duplicated(keep=False)]
    
    pids_isin = np.isin(appdf['party_id'], menusess['party_id'])
    sess_isin = np.isin(appdf['sesn_id'], menusess['sesn_id'])
    
    appdf = appdf.loc[~np.all([pids_isin, sess_isin], axis=0)]
    print('after removing nan in category1', len(appdf))
    
    menusess = pd.concat([menusess1, menusess3], axis=0)
    menusess = menusess.loc[~menusess.duplicated(keep=False)]

    pids_isin = np.isin(appdf['party_id'], menusess['party_id'])
    sess_isin = np.isin(appdf['sesn_id'], menusess['sesn_id'])
    appdf = appdf.loc[~np.all([pids_isin, sess_isin], axis=0)]
    print('after removing nan in category2', len(appdf))
    
    #session간의 방문일자 차이
    seqdf = appdf[['party_id','page_cd','sesn_id','dt']].groupby(['party_id','sesn_id']).last()['dt']
    seqdf = seqdf.reset_index()
    seqdf = seqdf.sort_values(['party_id','dt'])
    seqdf = seqdf.reset_index(drop=True)
    
    def diff_vstdate(x):
        b = pd.concat([pd.Series(x['dt'].iloc[0]), x['dt'].iloc[:-1]]).reset_index(drop=True)
        seqdiff = x['dt'].reset_index(drop=True).dt.to_timestamp() - b.dt.to_timestamp()
        seqdiff.name = "diff_dt"
        return seqdiff
    
    diffdf = seqdf.groupby(['party_id']).apply(diff_vstdate)
    diffdf = diffdf.reset_index()
    seqdf = pd.concat([seqdf, diffdf['diff_dt']], axis=1)
    
    #session별 페이지 길이
    uni_pcd_depth1 = pcd['menu_nm_1'].unique()
    uni_pcd_depth2 = pcd['menu_nm_2'].unique()
    pglen_perse = appdf.groupby(['party_id','sesn_id']).count().reset_index()[['party_id','sesn_id','page_cd']]
    
    #카테고리별 방문횟수
    uv_per_d1 = appdf.groupby(['party_id','sesn_id','menu_nm_1']).count().reset_index()[['party_id','sesn_id','menu_nm_1','page_cd']]
    uv_per_d2 = appdf.groupby(['party_id','sesn_id','menu_nm_2']).count().reset_index()[['party_id','sesn_id','menu_nm_2','page_cd']]

    uv_per_d1 = uv_per_d1.pivot(index=['party_id','sesn_id'], columns='menu_nm_1', values='page_cd')
    uv_per_d1 = uv_per_d1.fillna(0).reset_index()

    uv_per_d2 = uv_per_d2.pivot(index=['party_id','sesn_id'], columns='menu_nm_2', values='page_cd')
    uv_per_d2 = uv_per_d2.fillna(0).reset_index()
    
    #카테고리별 체류시간
    stydepth1 = appdf[['party_id','sesn_id','menu_nm_1','sty_tms']].groupby(['party_id','sesn_id','menu_nm_1']).mean()
    stydepth1 = stydepth1.reset_index()
    stydepth1 = stydepth1.pivot(index=['party_id','sesn_id'], columns=['menu_nm_1'], values=['sty_tms'])
    stydepth1 = stydepth1.fillna(0).reset_index()
    
    stydepth2 = appdf[['party_id','sesn_id','menu_nm_2','sty_tms']].groupby(['party_id','sesn_id','menu_nm_2']).mean()
    stydepth2 = stydepth2.reset_index()
    stydepth2 = stydepth2.pivot(index=['party_id','sesn_id'], columns=['menu_nm_2'],values=['sty_tms'])
    stydepth2 = stydepth2.fillna(0).reset_index()
    
    #종료율 관련
    endmenu = appdf[['party_id','page_cd','sesn_id','menu_nm_1']].groupby(['party_id','sesn_id']).last().reset_index()
    endmenu['value'] = 1
    endmenu = endmenu.pivot(index=['party_id','sesn_id'], columns=['menu_nm_1'], values=['value'])
    endmenu = endmenu.fillna(0).reset_index()
    
    assert len(seqdf) == len(pglen_perse) == len(uv_per_d1) ==len(uv_per_d2) == len(stydepth1) == len(stydepth2) == len(endmenu), 'All of them should have same length'
    
    #merge
    cand_df = [pglen_perse, uv_per_d1, uv_per_d2, stydepth1, stydepth2, endmenu]
    for cand in cand_df:
        beflen = len(seqdf)
        seqdf = pd.merge(seqdf, cand, on=['party_id','sesn_id'])
        assert beflen == len(seqdf), 'they should have same length'

    return seqdf 

def prep_lbl_per_mth(lbl, date):
    lbl['party_id'] = lbl['PartyId']
    lbl = lbl.drop(columns=['Unnamed: 0', 'PartyId'])
    lbl['month'] = pd.to_datetime(lbl['month'], format='%Y-%m')
    lblmth = lbl.loc[np.all([lbl['month'].dt.year==date.year, lbl['month'].dt.month==date.month], axis=0)]
    lblmth = lblmth.drop(columns=['month'])
    return lblmth

def prep_mbrlog_per_mth(mbrdf):
    
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
    
    #dt -> datetime 으로 변경
    mbrdf['dt'] = pd.to_datetime(mbrdf['dt'], format='%Y%m%d')
#     mbrdf['dt'] = pd.to_datetime(mbrdf['dt'], format='%Y-%m-%d')
    
    #party_id당 dt순으로 sorting
    mbrdf = mbrdf.sort_values(['party_id', 'dt'])
    
    #사용안하는 컬럼 drop
    mbrdf = mbrdf.drop(columns=cfg.unused_mbrcol)
    
    #null제거
    print('Orig. data len:', len(mbrdf))
    mbrdf = mbrdf.dropna()
    print('Aft. drop-nan:', len(mbrdf), '\n')
    
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
    passpids = []
    for pid in pids:
        pidmbrdf = mbrdf.loc[mbrdf['party_id'] == pid]
        vals = np.unique(pidmbrdf['mbr_scrb_dt'].values)
        inds = np.where(vals != 99991231)[0]
        if len(inds) > 1:
            newval.append(vals[np.where(vals != 99991231)[0][0]])
        else:
            mbrdf = mbrdf.drop(pidmbrdf.index)
            passpids.append(pid)
            
    for val, pid in zip(newval, pids):
        if pid in passpids:
            pass
        inds = np.where(mbrdf['party_id'] == pid)[0]
        mbrdf.loc[inds, 'mbr_scrb_dt'] = val 
    
    mbrdf['mbr_scrb_dt'] = pd.to_datetime(mbrdf['mbr_scrb_dt'], format='%Y%m%d')
    mbrdf['active_dur'] = mbrdf['dt'] - mbrdf['mbr_scrb_dt']
    
    #멤버십 등급 -> 1,2,3,4로 변경
    mbrsh_dic = {'Bronze': 1, 'Silver': 2, 'Gold': 3, 'Platinum': 4, '#':1}
    f = lambda x : mbrsh_dic[x]
    newmbrsh = mbrdf['cur_mbrsh_rwrd_st_cd'].transform(f)
    mbrdf['cur_mbrsh_rwrd_st_cd'] = newmbrsh
    
    #필요없는 칼럼 drop
    mbrdf = mbrdf.drop(columns=['vtlt_age_eff_dt', 'mbr_scrb_dt', 'cur_mbrsh_pd_goal_alct_cnt','cur_mbrsh_pd_goal_achv_cnt'])
    
    return mbrdf

def prep_gmlog_per_mth(gmdf):
    
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
    gmdf = gmdf.sort_values(['party_id', 'p_event_apl_dte'])
    
    #(포인트 반영일 - 획득일) <= 10
    gmdf = gmdf.loc[(gmdf['p_event_apl_dte'] - gmdf['points_effective_dte']).dt.days <= 10]
    return gmdf

def prep_pagecd(pcd):
    
    pcd = pcd.reset_index(drop=True)
    pcd = pcd.drop(columns=['No'])
    pcd = pcd.loc[pcd['menu_nm_1'] != '위젯']
    pcd = pcd.loc[pcd['menu_nm_2'] != '위젯']
    
    code2name = {}
    for k, v in zip(pcd['page_cd'].values,  pcd['page_nm'].values):
        if pd.isnull(v):
            code2name[k]=k
        else:
            code2name[k]=v
            
    return pcd, code2name

def merge_app_and_pcd(df, pcd):
    return pd.merge(left=df, right=pcd[['page_cd','menu_nm_1','menu_nm_2']], on=['page_cd'], how='left', sort=False)

def merge_app_and_lbl(df, lbl):   
    return pd.merge(df, lbl, on=['party_id'], how='inner', sort=False)

def merge_app_and_mbr(seqdf, mbrdf):
    seqdf['dt'] = seqdf['dt'].dt.to_timestamp()
    return pd.merge(seqdf, mbrdf, on=['party_id','dt'], how='inner')

def merge_app_and_gm(seqdf, gmdf):
    
    pointsdf = gmdf[['party_id','p_event_apl_dte','points_value']]
    pointsdf['points_value'] = pointsdf['points_value'].astype('float32')
    pointsdf = pointsdf.groupby(['party_id','p_event_apl_dte']).sum()
    pointsdf = pointsdf.reset_index()
    pointsdf['dt'] = pointsdf['p_event_apl_dte']
    pointsdf = pointsdf.drop(columns=['p_event_apl_dte'])
    
    mergeddf = pd.merge(seqdf, pointsdf, on=['party_id','dt'], how='left')
    mergeddf[['achv_rat','points_value']] = mergeddf[['achv_rat','points_value']].fillna(value=0)
    return mergeddf
        
def create_appcolnms(pcd):
    
    def appcol_names_pg(appcolnms):
        for cat in cat1:
            if cat in samecat:
                cat = cat + '_x'
            appcolnms.append(cat)
        for cat in cat2:
            if cat in samecat:
                cat = cat + '_y'
            appcolnms.append(cat)
        return appcolnms

<<<<<<< HEAD
def find_sty_ind(columns):
    for ind, col in enumerate(columns):
        if 'sty_tms' in col:
            return ind
=======
    def appcol_names_sty(appcolnms):
        for cat in cat1:
            if cat in samecat:
                cat = f"('sty_tms', '{cat}')_x"
            else:
                cat = f"('sty_tms', '{cat}')"
            appcolnms.append(cat)
        for cat in cat2:
            cat = f"('sty_tms_y', '{cat}')"
            appcolnms.append(cat)
        return appcolnms

    def appcol_names_end(appcolnms):
        for cat in cat1:
            cat=f"('value', '{cat}')"
            appcolnms.append(cat)
        return appcolnms
>>>>>>> 0dccc50425c38d9b9ad4bfd5f5f1732e674e97f0

def find_end_ind(columns):
    for ind, col in enumerate(columns):
        if 'value' in col:
            return ind
        
def appcol_names_pg(appcolnms):
    for cat in cat1:
        if cat in samecat:
            cat = cat + '_x'
        appcolnms.append(cat)
    for cat in cat2:
        if cat in samecat:
            cat = cat + '_y'
        appcolnms.append(cat)
    return appcolnms

def appcol_names_sty(appcolnms):
    for cat in cat1:
        if cat in samecat:
            cat = f"('sty_tms', '{cat}')_x"
        else:
            cat = f"('sty_tms', '{cat}')"
        appcolnms.append(cat)
    for cat in cat2:
        cat = f"('sty_tms_y', '{cat}')"
        appcolnms.append(cat)
    return appcolnms

def appcol_names_end(appcolnms):
    for cat in cat1:
        cat=f"('value', '{cat}')"
        appcolnms.append(cat)
    return appcolnms

def add_extra_appcols(seqdf, appcolnms):
    
    masks = np.ones(len(appcolnms), dtype=np.bool)
    pcd_ind = np.where(seqdf.columns == 'page_cd')[0][0]
    sty_ind = find_sty_ind(seqdf.columns)
    end_ind = find_end_ind(seqdf.columns)
    gen_ind = np.where(seqdf.columns == 'gender_cd')[0][0]
    
    pgcols = seqdf.columns[pcd_ind+1:sty_ind]
    stycols = seqdf.columns[sty_ind:end_ind]
    endcols = seqdf.columns[end_ind:gen_ind]
    
    for colset in [pgcols, stycols, endcols]:
        inds = np.where(np.isin(appcolnms, colset))
        masks[inds] = False
    
    cols_notexist = appcolnms[masks]
    seqdf[cols_notexist] = np.zeros(shape=(len(seqdf), len(cols_notexist)), dtype=np.float32)
    seqdfcol_woapp = seqdf.columns[~np.isin(seqdf.columns, appcolnms)]
    seqdf = seqdf[np.concatenate([seqdfcol_woapp, appcolnms])]
    return seqdf

    cat1 = pcd['menu_nm_1'].unique()[1:]
    cat2 = pcd['menu_nm_2'].unique()[1:]
    samecat = []
    for cat in cat2:
        if cat in cat1:
            samecat.append(cat)
    
    appcolnms = []
    appcolnms = appcol_names_pg(appcolnms)
    appcolnms = appcol_names_sty(appcolnms)
    appcolnms = appcol_names_end(appcolnms)
    
    return np.asarray(appcolnms)
    
def add_extra_appcols(seqdf, appcolnms):
    
    def find_sty_ind(columns):
        for ind, col in enumerate(columns):
            if 'sty_tms' in col:
                return ind

    def find_end_ind(columns):
        for ind, col in enumerate(columns):
            if 'value' in col:
                return ind
    
    masks = np.ones(len(appcolnms), dtype=np.bool)
    pcd_ind = np.where(seqdf.columns == 'page_cd')[0][0]
    sty_ind = find_sty_ind(seqdf.columns)
    end_ind = find_end_ind(seqdf.columns)
    gen_ind = np.where(seqdf.columns == 'gender_cd')[0][0]
    
    pgcols = seqdf.columns[pcd_ind+1:sty_ind]
    stycols = seqdf.columns[sty_ind:end_ind]
    endcols = seqdf.columns[end_ind:gen_ind]
    
    for colset in [pgcols, stycols, endcols]:
        inds = np.where(np.isin(appcolnms, colset))
        masks[inds] = False
    
    cols_notexist = appcolnms[masks]
    seqdf[cols_notexist] = np.zeros(shape=(len(seqdf), len(cols_notexist)), dtype=np.float32)
    seqdfcol_woapp = seqdf.columns[~np.isin(seqdf.columns, appcolnms)]
    seqdf = seqdf[np.concatenate([seqdfcol_woapp, appcolnms])]
    
    for col in seqdf.columns:
        if 'Unnamed' in col:
            seqdf = seqdf.drop(columns=[col])
            
    return seqdf

if __name__ == '__main__':
    
    from util import *
<<<<<<< HEAD
    seqds = SeqPreProcess(start='202002', end='202103')
    lbl = read_csv(cfg.label)
    for date in seqds.dates():
        print(date)
        csvpath = os.path.join('s3://', cfg.data_dir, seqfname(date.year, date.month))
        seqdf = read_csv(seqfname(date.year, date.month))
        lblmth = prep_lbl_per_mth(lbl, date)
        seqdf = merge_app_and_lbl(seqdf, lblmth)
=======
    seqds = SeqPreProcess(start='202001', end='202103')
    pcd, _ = prep_pagecd(read_csv(cfg.pgcd))
    appcolnms = create_appcolnms(pcd)
    for date in seqds.dates():
        print(date)
        seqdf = read_csv(seqfname(date.year, date.month))
        seqdf = add_extra_appcols(seqdf, appcolnms)
        
        csvpath = os.path.join('s3://', cfg.data_dir, seqfname(date.year, date.month))
>>>>>>> 0dccc50425c38d9b9ad4bfd5f5f1732e674e97f0
        seqdf.to_csv(csvpath)
        