import pandas as pd

from sklearn.metrics import f1_score
from transformers import pipeline
from datasets import load_dataset


def eval_task(pipe, task):

    if task == 'sst2_pt':
        targets=['positivo', 'negativo']
        data_val = load_dataset('csv', data_files='data/maritaca-ai_sst2_pt.csv')
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

        results = {
            'sst2_pt': {
                'score': score, 
                'f1-score': f1,
                'alias': 'sst2_pt'
            }
        }

        return results



def fillmask_evaluator(
        merged_path,
        task
):

    tokenizer_kwargs = {
        'truncation':True,
        'max_length':512
    }

    fill_mask = pipeline(
        task="fill-mask",
        model=merged_path,
        tokenizer=merged_path,
        device='cuda'
    )

    return eval_task(fill_mask, task)