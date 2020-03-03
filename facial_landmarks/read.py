import pandas as pd

# only select split
# df = pd.read_csv("omniart_v3_datadump.csv")
# omniart_paintings_df = df[df.general_type == "painting"]
# omniart_paintings_df.to_csv("omniart_v3_paintings.csv")
omniart_df = pd.read_csv("omniart_v3_datadump.csv", encoding = 'utf8')
portrait_df =  omniart_df[omniart_df.artwork_type == "portrait"]
portrait_df.to_csv("omniart_v3_portrait.csv", encoding = 'utf8')
