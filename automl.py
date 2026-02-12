import pandas as pd
from pycaret.regression import *

df = pd.read_csv("level1.csv", encoding="cp1252")

# Auto detect regression setup
s = setup(df, target=df.columns[-1], session_id=123)

best_model = compare_models()

final_model = finalize_model(best_model)

save_model(final_model, "best_level1_model")
