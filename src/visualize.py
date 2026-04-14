import os
import matplotlib.pyplot as plt
import seaborn as sns

def save_confusion_matrix(cm):
    os.makedirs("images", exist_ok=True)

    plt.figure()
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title("Confusion Matrix")
    plt.savefig("images/confusion_matrix.png")
    plt.close()