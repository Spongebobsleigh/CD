import openai
import pandas as pd
from datetime import datetime

# クライアント初期化
client = openai.OpenAI()

# モデル一覧を取得
models = client.models.list()

# モデル情報を整理
model_data = []
for model in models.data:
    model_data.append({
        "ID": model.id,
        "Created": datetime.fromtimestamp(model.created).strftime("%Y-%m-%d %H:%M:%S"),
        "Object": model.object,
        "Owned by": model.owned_by
    })

# DataFrameを作成
df = pd.DataFrame(model_data)

# 「Created」を降順（新しい順）でソート
df_sorted = df.sort_values(by="Created", ascending=False)

# Markdownで出力
print(df_sorted.to_markdown(index=False))

# APIkeyをexportする必要がある