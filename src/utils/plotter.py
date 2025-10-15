import matplotlib.pyplot as plt

def plot_results(x, y_true, y_pred):
    plt.figure(figsize=(6, 4))
    plt.plot(x, y_true, label='True', color='blue')
    plt.plot(x, y_pred, label='Predicted', color='red', linestyle='--')
    plt.legend()
    plt.title('Model Prediction vs Ground Truth')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
