import pandas as pd
from qis import get_qis, extract_qis
from query_gpt_categorizer.fasttext_hierarchical_model import FastTextHierarchicalModel
import os

def get_qis_info(query):
    qis = get_qis(query)
    if qis is None:
        return None
    else:
        return extract_qis(qis)
    
def explode_qis(df):
    dict_columns_df = pd.DataFrame(df['qis'].tolist())
    return pd.concat([df.drop('qis', axis=1), dict_columns_df], axis=1)

def is_product_query(row):
    if not row['shopping_intent'] and not row['product_intent']:
        return False
    else:
        return not all([row['product'] is None, row['product_family'] is None, row['model'] is None, row['material'] is None])
        
def get_data(source, qis_size):
    if source == 'ysearch':
        data = pd.read_parquet("data/ysearch_202205_202304_top_10000")
    elif os.path.exists(f'data/{source}_qis_{qis_size}.csv'):
        data = pd.read_csv(f'data/{source}_qis_{qis_size}.csv')
    else:
        data = pd.read_parquet(f"data/{source}_202205_202304_top_10000").head(qis_size)
        data['qis'] = data.apply(lambda x: get_qis_info(x['query']), axis=1)
        data = explode_qis(data)
        data['is_product_query'] = data.apply(lambda x: is_product_query(x), axis=1)
        data.to_csv(f'data/{source}_qis_{qis_size}.csv', index=False)
    return data

fasttext_models = FastTextHierarchicalModel.get_models('model_files/query_categorizer/ysearch', 'query_depth_quantized', 4)
query_categorizer = FastTextHierarchicalModel(fasttext_models, 4, 2, 0.2)

ysearch = get_data('ysearch', 2000)
b2b = get_data('b2b', 2000)
shopping_hub = get_data('shopping_hub', 2000)

ysearch = ysearch[~ysearch['query'].str.contains('"""')]
b2b = b2b[b2b['is_product_query']]
shopping_hub = shopping_hub[shopping_hub['is_product_query']]
shopping_hub = shopping_hub[~shopping_hub['query'].str.startswith('GE ')]

ysearch['source'] = 'ysearch'
b2b['source'] = 'b2b'
shopping_hub['source'] = 'shopping_hub'

ysearch = ysearch.reset_index()
b2b = b2b.reset_index()
shopping_hub = shopping_hub.reset_index()

sample_size = 600
ysearch = ysearch.head(sample_size)
b2b = b2b.head(sample_size)
shopping_hub = shopping_hub.head(sample_size)


data = pd.concat([ysearch, b2b, shopping_hub])
print(data)
def agg(group):
    sources_count = {}
    for source, count in zip(group['source'], group['count']):
        sources_count[source] = count
    score = sum([(1/(i+1)) for i, j in zip(group['index'], group['count'])])
    return pd.Series({'sources_count': sources_count, 'score': score, 'query': group.name})

data = data.groupby('query').apply(agg)
data['query_gpt_category'] = data.apply(lambda x: query_categorizer.predict(x['query'])[0][0], axis=1)
data = data.sort_values('score', ascending=False)
data[['query', 'score', 'sources_count', 'query_gpt_category']].to_csv(f'data/combined_queries_{sample_size}.csv', index=False)
print(data)
ysearch_count = data['sources_count'].apply(lambda x: 'ysearch' in x.keys()).sum()
b2b_count = data['sources_count'].apply(lambda x: 'b2b' in x.keys()).sum()
shopping_hub_count = data['sources_count'].apply(lambda x: 'shopping_hub' in x.keys()).sum()
print('ysearch_count', ysearch_count)
print('b2b_count', b2b_count)
print('shopping_hub_count', shopping_hub_count)
print(data['query_gpt_category'].value_counts())
