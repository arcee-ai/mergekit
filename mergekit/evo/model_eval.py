import pandas as pd
import numpy as np

from sklearn.metrics import f1_score, accuracy_score
from transformers import pipeline
from datasets import load_dataset


def eval_task(pipe, task):

    if task == 'sentiment_pt':
        targets=['positivo', 'negativo']
        data_val = load_dataset('csv', data_files='data/maritaca-ai_sst2_imdb_pt.csv')
        indices = np.random.randint(0, data_val.shape[0], 500)
        data_val = data_val.select(indices)
        vals = data_val['train'].map(
            lambda x: pipe(
                x['text_fillmask'],
                top_k=1,
                targets=targets
            )[0]
        )
        df = pd.DataFrame(vals)

        score = df['score'].mean()*1000
        f1 = f1_score(
            df['sentiment'].replace('positivo', 1).replace('negativo', 0), 
            df['token_str'].replace('positivo', 1).replace('negativo', 0), 
            average='binary'
        )

        acc = accuracy_score(
            df['sentiment'].replace('positivo', 1).replace('negativo', 0),
            df['token_str'].replace('positivo', 1).replace('negativo', 0),
        )

        results = {
            'sst2_pt': {
                'score': score, 
                'f1-score': f1,
                'accuracy': acc,
                'alias': 'sst2_pt'
            }
        }

        return results


def fillmask_evaluator(
        merged_path,
        task
):

    fill_mask = pipeline(
        task="fill-mask",
        model=merged_path,
        tokenizer=merged_path,
        device='cuda'
    )

    return eval_task(fill_mask, task)
