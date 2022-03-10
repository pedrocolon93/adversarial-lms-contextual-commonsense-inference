import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="darkgrid")

data = pd.read_csv("bart_hinting.csv")

fig, axs = plt.subplots(1,2,figsize=(18, 10))
g = sns.lineplot(x="ITERATION", y="BLEU", hue="Hinting", data=data,ax=axs[0]).set_title("Average BLEU")
g2 = sns.lineplot(x="ITERATION", y="BLEU", hue="Hinting", data=data,estimator=np.median,ax=axs[1]).set_title("Median BLEU")
plt.show()
