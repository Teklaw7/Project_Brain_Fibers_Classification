import torch
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
pred = []
true = []

for i in range(514):
    label_true = torch.load(f"/home/timtey/Documents/Projet/label_true/label_true_{i}.pt")
    # print(label_true)
    # print(label_true.shape)
    labell = label_true.tolist()
    # print(labell)
    # print(labell.shape)
    true.append(labell)

    label_pred = torch.load(f"/home/timtey/Documents/Projet/label_pred/label_pred_{i}.pt")
    # print(label_pred)
    # print(label_pred.shape)
    labelp = label_pred.tolist()
    # print(labelp)
    # print(labelp.shape)
    pred.append(labelp)

true = [item for sublist in true for item in sublist]
pred = [item for sublist in pred for item in sublist]
# print(len(true))
# print(len(pred))

list_class = list(range(1, 58))
# ex_classes = {'Classes':}
# print(confusion_matrix(true, pred))
conf_matrix = confusion_matrix(true, pred)
# print(conf_matrix)
cmn = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
fig = px.imshow(cmn,labels=dict(x="Predicted condition", y="Actual condition"),x=list_class,y=list_class)
fig.update_xaxes(side="top")
fig.write_image("/home/timtey/Documents/Projet/confusion_matrix/confusion_matrix_px.png")
fig.show()

# cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
# fig, ax = plt.subplots(figsize=(10,10))
# sns.heatmap(cmn, annot=True, fmt='.2f', xticklabels=target_names, yticklabels=target_names)
# plt.ylabel('Actual')
# plt.xlabel('Predicted')
# plt.show(block=False)

# conf_matrix_inv = confusion_matrix(pred, true)
# # print(conf_matrix_inv)
# fig_inv = px.imshow(conf_matrix_inv,labels=dict(x="Actual condition", y="Predicted condition"),x=list_class,y=list_class)
# fig_inv.update_xaxes(side="top")
# fig_inv.write_image("/home/timtey/Documents/Projet/confusion_matrix/confusion_matrix_px_inv.png")
# fig_inv.show()

print(classification_report(true, pred))
# print(confusion_matrix(true, pred))
# cm_display = ConfusionMatrixDisplay(confusion_matrix(true, pred), display_labels = list_class).plot()
# plt.savefig("/home/timtey/Documents/Projet/confusion_matrix/confusion_matrix.png")
# plt.show()
