import pandas as pd
import numpy as np

dataMahasiswa= {
    'nim': [1, 2],
    'nama': ['Ega', 'Annora']
}

df = pd.DataFrame(dataMahasiswa)
print(df)
df.isnull().sum()