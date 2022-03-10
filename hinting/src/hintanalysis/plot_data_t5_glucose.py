import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="darkgrid")

def fix_run_name(entry):
    entry = entry.replace("t5","")
    entry = entry.replace(" - score","")
    if not entry.strip()=="hint":
        entry = entry.replace("hint"," hint")
    return entry
data = pd.read_excel("t5-glucose-hinting.xlsx")
data["Run"] = data["Run"].apply(fix_run_name)
data = data.sort_values("Run")
fig, axs = plt.subplots(1,2,figsize=(18, 10))
sns.lineplot(x="Step", y="BLEU", hue="Run",  data=data,ax=axs[0]).set_title("Average BLEU")
sns.lineplot(x="Step", y="BLEU", hue="Run",  data=data,estimator=np.median,ax=axs[1]).set_title("Median BLEU")

fig.show()
