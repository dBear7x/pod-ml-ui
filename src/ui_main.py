from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLabel,
    QPushButton, QLineEdit, QMessageBox
)
import torch
import numpy as np
from src.train_model import SimpleNet
from src.utils.plotter import plot_results

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PyTorch + PySide6 Demo")
        self.setGeometry(200, 200, 400, 200)

        layout = QVBoxLayout(self)
        self.input_field = QLineEdit(self)
        self.input_field.setPlaceholderText("输入一个x值，例如 1.57")
        self.predict_button = QPushButton("预测 sin(x)")
        self.result_label = QLabel("结果：")
        self.plot_button = QPushButton("绘制曲线")

        layout.addWidget(self.input_field)
        layout.addWidget(self.predict_button)
        layout.addWidget(self.result_label)
        layout.addWidget(self.plot_button)

        self.model = SimpleNet()
        try:
            self.model.load_state_dict(torch.load("model.pt"))
        except FileNotFoundError:
            QMessageBox.warning(self, "警告", "未找到模型文件，请先运行 train_model.py 训练模型。")

        self.predict_button.clicked.connect(self.on_predict)
        self.plot_button.clicked.connect(self.on_plot)

    def on_predict(self):
        try:
            x_val = float(self.input_field.text())
            x_t = torch.tensor([[x_val]], dtype=torch.float32)
            y_pred = self.model(x_t).item()
            self.result_label.setText(f"结果：sin({x_val:.3f}) ≈ {y_pred:.4f}")
        except ValueError:
            QMessageBox.warning(self, "输入错误", "请输入有效数字。")

    def on_plot(self):
        x = np.linspace(-np.pi, np.pi, 200)
        y_true = np.sin(x)
        x_t = torch.tensor(x.reshape(-1, 1), dtype=torch.float32)
        y_pred = self.model(x_t).detach().numpy().flatten()
        plot_results(x, y_true, y_pred)
