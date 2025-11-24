# 0_preprocess.py
import re, argparse, math
import pandas as pd
import numpy as np

def extract_value(text):
    if pd.isna(text): return np.nan
    m = re.search(r'Value:\s*([0-9]*\.?[0-9]+)', text, flags=re.I)
    if m: return float(m.group(1))
    # fallback find first number that looks like quantity / volume
    m2 = re.search(r'([0-9]*\.?[0-9]+)\s*(oz|ounce|fl oz|ml|l|g|kg)', text, flags=re.I)
    if m2:
        return float(m2.group(1))
    return np.nan

def extract_unit(text):
    if pd.isna(text): return None
    m = re.search(r'Unit:\s*([A-Za-z0-9\s\./%]+)', text, flags=re.I)
    if m: return m.group(1).strip().lower()
    m2 = re.search(r'([0-9]*\.?[0-9]+)\s*(oz|ounce|fl oz|ml|l|g|kg)', text, flags=re.I)
    if m2:
        return m2.group(2).lower()
    return None

def extract_ipq(text):
    if pd.isna(text): return 1
    # Pack of N, N per case, (Pack of N), Pack N, Pack: N
    m = re.search(r'pack\s*(?:of)?\s*[:\(\s]*\s*(\d{1,3})', text, flags=re.I)
    if m: return int(m.group(1))
    m2 = re.search(r'(\d{1,3})\s*per\s*case', text, flags=re.I)
    if m2: return int(m2.group(1))
    # looks for detailed packs "32 cookies total" sometimes indicates total units but we treat as ipq if explicit
    return 1

# unit normalization (convert ounces to ml if needed) - only if you want a single unit baseline
unit_to_ml = {
    'oz': 29.5735, 'ounce': 29.5735, 'fl oz': 29.5735,
    'ml': 1.0, 'l': 1000.0, 'g': 1.0, 'kg': 1000.0
}

def normalize_value_unit(value, unit):
    if pd.isna(value) or not unit: return value, unit
    u = unit.lower().strip()
    if u in unit_to_ml:
        # convert numeric value to ml units and mark unit='ml'
        return float(value) * unit_to_ml[u], 'ml'
    return value, unit

def build_features(df):
    df['catalog_content'] = df['catalog_content'].fillna('').astype(str)
    df['content_clean'] = df['catalog_content'].str.replace(r'\s+', ' ', regex=True).str.strip()
    df['value_raw'] = df['catalog_content'].apply(extract_value)
    df['unit_raw'] = df['catalog_content'].apply(extract_unit)
    df['ipq'] = df['catalog_content'].apply(extract_ipq).fillna(1).astype(int)
    # normalize
    norm = df.apply(lambda r: normalize_value_unit(r['value_raw'], r['unit_raw']), axis=1, result_type='expand')
    df['value_norm'] = norm[0]
    df['unit_norm'] = norm[1]
    # some simple heuristics
    df['content_len'] = df['content_clean'].str.len()
    df['word_count'] = df['content_clean'].str.split().str.len().fillna(0)
    # brand heuristics - first token with capital letter in first 6 tokens
    def extract_brand(s):
        toks = re.split(r'[\:\-\|]', s)  # split on separators
        pref = toks[0]
        for tok in pref.split()[:6]:
            if tok and tok[0].isupper():
                return tok.lower()
        return 'unknown'
    df['brand'] = df['catalog_content'].apply(extract_brand)
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', default='dataset/train.csv')
    parser.add_argument('--test', default='dataset/test.csv')
    parser.add_argument('--out', default='preprocessed.parquet')
    args = parser.parse_args()
    train = pd.read_csv(args.train)
    test = pd.read_csv(args.test)
    train = build_features(train)
    test = build_features(test)
    # keep useful columns; preserve original price in train
    train['is_test'] = 0
    test['is_test'] = 1
    combined = pd.concat([train, test], sort=False).reset_index(drop=True)
    combined.to_parquet(args.out, index=False)
    print("Saved", args.out)
