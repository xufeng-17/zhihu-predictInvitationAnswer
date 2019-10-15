import os,sys
import pandas as pd
import numpy as np

tool_path = '/Users/xufeng/learning/public_tool'
sys.path.append(tool_path)

from transform import timestampTransformToDatetime

data_path = 'data/'

__depend__ = {
    "answer_info": "data/data_set_0926/answer_info_0926.txt",
    "invite_info": "data/data_set_0926/invite_info_0926.txt",
    "invite_info_evaluate_1": "data/data_set_0926/invite_info_evaluate_1_0926.txt",
    "member_info": "data/data_set_0926/member_info_0926.txt",
    "question_info": "data/data_set_0926/question_info_0926.txt",
    "single_word_vectors_64d": "data/data_set_0926/single_word_vectors_64d.txt",
    "topic_vectors_64d": "data/data_set_0926/topic_vectors_64d.txt",
    "word_vectors_64d": "data/data_set_0926/word_vectors_64d.txt"
}

## 1. load data

### 1.1 目标数据集
invite_info = pd.read_csv(__depend__['invite_info'], header=None,  sep='\t')
invite_info.columns = ['ques_id', 'user_id', 'create_time', 'is_ans']
print(invite_info.shape) # (9489162, 4)
target = 'is_ans'
create_time = 'create_time'
new_variable = ['day', 'hour', 'group']
key_variable = ['ques_id','user_id']

invite_info['day'] = invite_info[create_time].apply(lambda x: int(x.split('-')[0][1:])).astype(np.int16)
invite_info['hour'] = invite_info[create_time].apply(lambda x: int(x.split('-')[1][1:])).astype(np.int16)


#### 目标占比
invite_info[target].isnull().value_counts()
train_is_ans = invite_info[target].value_counts()
train_is_ans/train_is_ans.sum()

#### 时间区间
max(invite_info['day']) - min(invite_info['day'])

#### 查看是否重复
invite_info_counter = invite_info.groupby(key_variable)[target].count()
invite_info_counter[invite_info_counter > 1]

#### 划分训练测试集
invite_info['group'] = ((invite_info['day'] - min(invite_info['day'])) / 7).astype(int)
invite_info['group'].value_counts().sort_index()

invite_info['is_ans'].groupby(invite_info['group']).agg({'acc':'count' , 'target':'sum','rate':'mean'})

train = invite_info[invite_info['group'] != 4]
test =  invite_info[invite_info['group'] == 4]

train.to_hdf('data/processed/invite_train.h5', key='data')
test.to_hdf('data/processed/invite_test.h5', key='data')


## 用户


## 问题信息



## 用户和问题的交叉信息



