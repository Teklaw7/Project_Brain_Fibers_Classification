import torch
from sklearn.metrics import classification_report, confusion_matrix
import plotly.express as px
import numpy as np
pred = []
true = []

for i in range(514):
    label_true = torch.load(f"/home/timtey/Documents/Projet/label_true/label_true_{i}.pt")
    labell = label_true.tolist()
    true.append(labell)

    label_pred = torch.load(f"/home/timtey/Documents/Projet/label_pred/label_pred_{i}.pt")
    labelp = label_pred.tolist()
    pred.append(labelp)

true = [item for sublist in true for item in sublist]
pred = [item for sublist in pred for item in sublist]
list_class = list(range(1, 58))
conf_matrix = confusion_matrix(true, pred)
cmn = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
fig = px.imshow(cmn,labels=dict(x="Predicted condition", y="Actual condition"),x=list_class,y=list_class)
fig.update_xaxes(side="top")
fig.write_image("/home/timtey/Documents/Projet/confusion_matrix/confusion_matrix_px.png")
fig.show()

print(classification_report(true, pred))