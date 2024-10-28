import pandas as pd
from datasets import load_dataset


def prep_sst2(col):

    ds = load_dataset("maritaca-ai/sst2_pt")
    
    data_val = ds[col].rename_column('label', 'true_label')
    df_sst2 = pd.DataFrame(data=data_val)
    df_sst2['text_fillmask'] = df_sst2['text'] + '. Sentimento: [MASK]'
    df_sst2['sentiment'] = df_sst2['true_label']\
        .replace(1, 'positivo').replace(0, 'negativo').replace(-1, 'neutro')
    df_sst2.to_csv('data/maritaca-ai_sst2_pt.csv', index=False)


def prep_imdb(col):

    ds = load_dataset("maritaca-ai/imdb_pt")
    imdb_val = ds[col].rename_column('label', 'true_label')
    df_imdb = pd.DataFrame(data=imdb_val)
    df_imdb['text_fillmask'] = df_imdb['text'] + '. Sentimento: [MASK]'
    df_imdb['sentiment'] = df_imdb['true_label']\
        .replace(1, 'positivo').replace(0, 'negativo').replace(-1, 'neutro')
    df_imdb.to_csv('data/maritaca-ai_imdb_pt.csv', index=False)


def prep_sentiment_base(col):

    ds = load_dataset("maritaca-ai/sst2_pt")    
    data_val = ds[col].rename_column('label', 'true_label')
    df_sst2 = pd.DataFrame(data=data_val)
    df_sst2['text_fillmask'] = df_sst2['text'] + '. Sentimento: [MASK]'
    df_sst2['sentiment'] = df_sst2['true_label']\
        .replace(1, 'positivo').replace(0, 'negativo').replace(-1, 'neutro')
    
    ds = load_dataset("maritaca-ai/imdb_pt")
    imdb_val = ds[col].rename_column('label', 'true_label')
    df_imdb = pd.DataFrame(data=imdb_val)
    df_imdb['text_fillmask'] = df_imdb['text'] + '. Sentimento: [MASK]'
    df_imdb['sentiment'] = df_imdb['true_label']\
        .replace(1, 'positivo').replace(0, 'negativo').replace(-1, 'neutro')    

    df = pd.concat([df_sst2, df_imdb], axis=0)

    df.to_csv('data/maritaca-ai_sst2_imdb_pt.csv', index=False)



def main():
    col='train'

    prep_sentiment_base(col)

main()
