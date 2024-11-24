import os
import pandas as pd
import json
import re
from collections import defaultdict

tweets_by_label = defaultdict(list)

cwd = os.getcwd()
with open(f'{cwd}/data/data/public_datasets/3/dataset.json', 'r', encoding='utf-8') as f:
    for line in f:
        tweet = json.loads(line)
        hashtags = re.findall(r'#(\w+)', tweet['text'])
        
        if hashtags:
            label = tweet['label_name'][0]
            tweets_by_label[label].append({
                'text': tweet['text'],
                'tags': ' '.join(hashtags)
            })

for label, tweets in tweets_by_label.items():
    filename = label.lower().replace(' ', '_').replace('&', 'and')
    output_path = f'{cwd}/data/data/{filename}.csv'
    
    new_df = pd.DataFrame(tweets)
    
    if os.path.exists(output_path):
        existing_df = pd.read_csv(output_path)
        df = pd.concat([existing_df, new_df]).drop_duplicates(subset=['text'])
    else:
        df = new_df
    
    df.to_csv(output_path, index=False)

    print(f"{filename}.csv: {len(df)} rows")