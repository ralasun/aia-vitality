import pandas as pd
import numpy as np


def load_applog(start, periods):
    # Load applog data
    monthly_df = []
    filename = pd.period_range(start=start, periods=periods, freq='M').strftime('%Y%m')
    for fn in filename:
        monthly_df.append(pd.read_csv('./data/applog/applog_' + fn + '.csv'))
    df = pd.concat(monthly_df)
    return df

def basic_preprocess_applog(df):
    # 칼럼명 변경
    df.columns = ['PartyId', '방문일시', '페이지코드', '체류시간', '세션ID', '로그인여부', '신규방문여부', '통신회사코드', '이탈여부', '연월일']

    # null 제거
    print('Orig. data len:', len(df))
    df = df.dropna()
    print('Aft. drop-nan:', len(df), '\n')

    # party_id int형으로 변경
    df['PartyId'] = df['PartyId'].astype('Int64')

    # 방문일시 datetime으로 변환
    vst_dtm = df['방문일시'].astype('str')
    f = lambda x: x[:-3]
    vst_dtm = vst_dtm.apply(f)

    vst_dtm = pd.to_datetime(vst_dtm, format='%Y%m%d%H%M%S')
    df['방문일시'] = vst_dtm

    # 연월일 칼럼 제거
    df = df.drop(columns='연월일')

    # 신규방문여부 Y->1, N->0
    df['신규방문여부'] = df['신규방문여부'].replace({'Y': 1, 'N': 0})

    # 1970년 데이터 제외
    df = df[df['방문일시'].dt.year != 1970]
    df = df.reset_index(drop=True)
    return df

def load_member(date):
    # Load members data
    members_df = pd.read_csv('data/member/mbr_' + date + '.csv',
                             usecols=['party_id', 'mbr_scrb_dt', 'fee_yn', 'fcip_yn', 'ip_insr_cd', 'lst_vst_dt'])
    members_df.columns = ['PartyId', '가입일자', '멤버십비납부여부', '유료멤버십보험가입여부', '11/4이전유료멤버십', '마지막로그인일자']

    # null 제거
    print('Orig. data len:', len(members_df))
    members_df = members_df.dropna()
    print('Aft. drop-nan:', len(members_df), '\n')

    # party_id int형으로 변경
    members_df['PartyId'] = members_df['PartyId'].astype('Int64')

    # 가입일자 out of bound 제거
    members_df = members_df[members_df['가입일자'] != 99991231]

    # 가입일자 datetime으로 변환
    join_dtm = members_df['가입일자'].astype('str')
    join_dtm = pd.to_datetime(join_dtm, format='%Y%m%d')
    members_df['가입일자'] = join_dtm

    last_dtm = members_df['마지막로그인일자'].astype('str')
    last_dtm = pd.to_datetime(last_dtm, format='%Y%m%d')
    members_df['마지막로그인일자'] = last_dtm

    # PartyId index 설정
    members_df = members_df.set_index('PartyId')

    # PartyId 중복 제거
    members_df = members_df[~members_df.index.duplicated(keep='last')]
    return members_df