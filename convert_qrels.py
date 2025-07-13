import pandas as pd
import json
import os

def convert_qrels_tsv_to_jsonl(tsv_path, jsonl_path):
    try:
        # قراءة الملف مع اعتبار السطر الأول هو رؤوس الأعمدة
        df = pd.read_csv(tsv_path, sep='\t', header=0)

        # التحقق من أسماء الأعمدة تلقائيًا (قد تكون مختلفة حسب dataset)
        if 'score' in df.columns:
            df = df.rename(columns={'score': 'relevance'})
        if 'corpus-id' in df.columns:
            df = df.rename(columns={'corpus-id': 'doc_id'})
        if 'query-id' in df.columns:
            df = df.rename(columns={'query-id': 'query_id'})

        # تجاهل الصفوف التي تحتوي على قيم غير رقمية في "relevance"
        df = df[pd.to_numeric(df['relevance'], errors='coerce').notna()]
        df['relevance'] = df['relevance'].astype(int)

        # كتابة إلى jsonl
        with open(jsonl_path, 'w', encoding='utf-8') as f:
            for _, row in df.iterrows():
                json.dump({
                    "query_id": str(row['query_id']),
                    "doc_id": str(row['doc_id']),
                    "relevance": int(row['relevance'])
                }, f) 
                f.write('\n')

        print(f"✅ تم تحويل qrels بنجاح إلى {jsonl_path}")
    
    except Exception as e:
        print(f"❌ حدث خطأ أثناء التحويل: {e}")

if __name__ == "__main__":
    convert_qrels_tsv_to_jsonl(
        ".data/quora/test.tsv",
        ".data/quora/qrels.jsonl"
    )
