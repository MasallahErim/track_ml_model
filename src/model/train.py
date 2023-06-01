import pandas as pd 
import numpy as np
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn import preprocessing
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split
import json
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer

from pathlib import Path
from typing import Optional
def find_project_root() -> Optional[Path]:
    current = Path(".").resolve()
    while True:
        if (current / ".git").exists():
            return current
        if current.parent == current:
            print("WARNING: No .git dir found")
            return current
        current = current.parent

rootPath = find_project_root()








df = pd.read_csv(rootPath/"src/data/processed/processed_farms_data.csv")

if "Unnamed: 0" in df.columns:
    del df["Unnamed: 0"]



y = df.pop("cons_general").to_numpy()
y[y< 4] = 0
y[y>= 4] = 1

X = df.to_numpy()
X = preprocessing.scale(X)

imp = SimpleImputer(missing_values=np.nan, strategy='mean')
imp.fit(X)
X = imp.transform(X)


clf = QuadraticDiscriminantAnalysis()
yhat = cross_val_predict(clf, X, y, cv=5)

acc = np.mean(yhat==y)
tn, fp, fn, tp = confusion_matrix(y, yhat).ravel()
specificity = tn / (tn+fp)
sensitivity = tp / (tp + fn)


with open(rootPath/"outputs/metrics.json", 'w') as outfile:
        json.dump({ "accuracy": acc, "specificity": specificity, "sensitivity":sensitivity}, outfile)

score = yhat == y
score_int = [int(s) for s in score]
df['pred_accuracy'] = score_int

sns.set_color_codes("dark")
ax = sns.barplot(x="region", y="pred_accuracy", data=df, palette = "Greens_d")
ax.set(xlabel="Region", ylabel = "Model accuracy")
plt.savefig(rootPath/"outputs/by_region.png",dpi=80)