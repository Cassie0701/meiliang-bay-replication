"""重跑梅梁湾透明度探索模型：python 梅梁湾透明度模型_重跑.py
依赖：pandas, numpy, scikit-learn。CSV 与此脚本放同一目录。
配对表只含月份，不能视作同日验证。影像提取使用 Landsat 8 Collection 2 L2
绿色和红色波段：SR = DN * 0.0000275 - 0.2，QA_PIXEL 排云/云影并保留水体。
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

path=Path(__file__).with_name('梅梁湾Landsat8_月度点位配对59条.csv')
data=pd.read_csv(path)
assert len(data)==59 and data[['year','month','site']].drop_duplicates().shape[0]==59
features=data[['green_SR','red_SR']].copy()
features['red_green_ratio']=data['red_SR']/data['green_SR']
train=data.year<2015
test=data.year==2015
assert train.sum()==42 and test.sum()==17
for target in ['secchi_m','suspended_solid_mg_L']:
    y=data[target]
    model=make_pipeline(StandardScaler(),Ridge(alpha=10))
    model.fit(features.loc[train],y.loc[train])
    pred=model.predict(features.loc[test])
    baseline=np.repeat(y.loc[train].mean(),test.sum())
    print(target,'test_n',test.sum(),
          'MAE',round(mean_absolute_error(y.loc[test],pred),3),
          'R2',round(r2_score(y.loc[test],pred),3),
          'baseline_MAE',round(mean_absolute_error(y.loc[test],baseline),3))
