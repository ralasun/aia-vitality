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
cfg.seq_dir = 'sunhwa/'

#page_code
cfg.pgcd = 'data/applog/page_code_info_210421.csv'

#party_id label
cfg.label = 'data/partyid_monthly_class_210428.csv'

#member columns not used
cfg.unused_mbrcol = ['cur_mbrsh_pd_sta_dt', 'cur_mbrsh_pd_end_dt', 'bf_mbrsh_pd_sta_dt', 'bf_mbrsh_pd_end_dt', 
                                'bf_mbrsh_pd_acqr_pt', 'cur_mbrsh_pd_goal_not_achv_cnt', 'geographical_area', 'cancelled_dt',
                                'trdprt_psnl_info_ofr_agre_yn', 'trdprt_psnl_info_ofr_agre_dt','mkt_psnl_info_clct_agre_yn',
                                'mkt_psnl_info_clct_agre_dt','mkt_psnl_info_ofr_agre_yn', 'mkt_psnl_info_ofr_agre_dt','trdprt_sens_info_ofr_agre_yn', 
                                'trdprt_sens_info_ofr_agre_dt', 'skt_data_share_agre_yn', 'skt_data_share_agre_dt','info_share_agre_yn', 
                                'info_share_agre_dt', 'info_get_agre_yn','info_get_agre_dt', 'other_info_agre_yn', 'other_info_agre_dt','receive_ad_yn', 
                                'receive_ad_dt', 'hh_data_share_yn','hh_data_share_dt', 'mbrsh_st_ty_id', 'mbrsh_st_eff_dt', 'ip_insr_cd', 'gnrl_insr_cd', 'wk_misn_sta_dt',
                                'fee_yn','fcip_yn', 'lst_vst_dt']

#goal mission columns used
cfg.used_gmcol = ['p_event_apl_dte','points_value','points_effective_dte']



#fname 
def appfname(year, month):
    if month < 10:
        month = f'0{month}'
    return os.path.join(cfg.applog_dir, f'applog_{year}{month}_new_sesn_id.csv')

def mbrfname(year, month):
    if month < 10:
        month = f'0{month}'
    return os.path.join(cfg.mbr_dir, f'mbr_{year}{month}.csv')

def gmfname(year, month):
    if month < 10:
        month = f'0{month}'
    return os.path.join(cfg.mss_dir, f'goal_misn_{year}{month}.csv')

def seqfname(year, month):
    if month < 10:
        month = f'0{month}'
    return os.path.join(cfg.seq_dir, f'seq_{year}{month}.csv')

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
        print(f'{fname} 로딩중')
        obj = s3.get_object(Bucket=cfg.data_dir, Key=fname)
        data.append(pd.read_csv(obj['Body']))  
        print(f'{fname} 로딩완료')
        print()
    return data

def read_csv(key):
    s3 = boto3.client('s3')
    obj = s3.get_object(Bucket=cfg.data_dir, Key=key)
    return pd.read_csv(obj['Body'])



if __name__ == '__main__':
    
    import time
    start= time.time()
    data = load_file(cfg.gm3)