import math
from sklearn.preprocessing import OneHotEncoder
import pandas as pd


def count_capital_letters_part(text):
    capital_letters = 0
    c: str
    for c in text:
        if c.isupper():
            capital_letters += 1

    return capital_letters / len(text) if len(text) > 0 else 0


def count_lowercase_letters_part(row):
    if len(row['text']) == 0:
        return 0
    else:
        return 1 - row['capital_letters_part']

def extract_data(df): 
    print(len(df['location']))

    encoder = OneHotEncoder()
    # fir_transform expects 2-D array hence we need to reshape the data from 1-D to 2-D.
    encoder_df = pd.DataFrame(encoder.fit_transform(df[['location']]).toarray())
    print(encoder_df.head())


    df['text_length'] = df['text'].str.len()
    df['capital_letters_part'] = df['text'].apply(count_capital_letters_part)
    df['lowercase_letters_part'] = df.apply(lambda row: count_lowercase_letters_part(row), axis=1)
    df['has_url'] = df['text'].str.contains('http')

    return df[['text_length', 'capital_letters_part', 'lowercase_letters_part', 'has_url', 'target']]