import sys
from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QTextEdit,
    QVBoxLayout, QLabel, QFileDialog, QGroupBox, QHBoxLayout
)
from PyQt5.QtGui import QFont
from TextProcessing import process_text
from Classification import ClassificationModel
from IR import IRModel

class NLP_GUI(QWidget):
    def __init__(self):
        super().__init__()
        self.ir_model = IRModel()
        self.classification_model = ClassificationModel()
        self.setWindowTitle("NLP Project")
        self.resize(720, 800)
        self.setStyleSheet("""
            QWidget {
                background-color: #1e1e2e;
                color: #eaeaea;
                font-family: Segoe UI;
                font-size: 14px;
            }

            QGroupBox {
                border: 2px solid #3399ff;
                border-radius: 10px;
                margin-top: 5px;
                padding: 10px;
                font-weight: bold;
            }

            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top center;
                padding: 0 10px;
                color: #3399ff;
                font-size: 15px;
            }

            QPushButton {
                background-color: #2e2e4d;
                border: 1px solid #3399ff;
                border-radius: 8px;
                padding: 8px;
                font-size: 15px;
            }

            QPushButton:hover {
                background-color: #3399ff;
                color: black;
            }

            QPushButton:pressed {
                background-color: #1a75ff;
            }

            QTextEdit {
                background-color: #11111b;
                border-radius: 8px;
                padding: 8px;
                font-size: 15px;
                color: #ffffff;
            }

            QLabel {
                font-size: 15px;
                font-weight: bold;
            }
        """)

        self.output = QTextEdit()
        self.output.setReadOnly(True)

        self.query = QTextEdit()
        self.query.setFixedHeight(80)
        self.query.setFont(QFont("Segoe UI", 14))
        self.query.setPlaceholderText("Enter your query here...")
        
        # Buttons
        self.btn_text_processing = QPushButton("Text Processing")
        self.btn_classification = QPushButton("Classify File")
        self.btn_tfidf = QPushButton("TF-IDF Search")
        self.btn_vector = QPushButton("Vector Space Search")

        # Layouts
        main_layout = QVBoxLayout()

        # Text Processing Group
        tp_group = QGroupBox("Text Processing")
        tp_layout = QVBoxLayout()
        tp_layout.addWidget(self.btn_text_processing)
        tp_group.setLayout(tp_layout)

        # Classification Group
        cls_group = QGroupBox("Text Classification")
        cls_layout = QVBoxLayout()
        cls_layout.addWidget(self.btn_classification)
        cls_group.setLayout(cls_layout)

        # Information Retrieval Group
        ir_group = QGroupBox("Information Retrieval")
        ir_group.setMaximumHeight(200)  

        ir_layout = QVBoxLayout()
        ir_layout.setContentsMargins(5, 5, 5, 5)  
        ir_layout.setSpacing(5)  
        ir_layout.addWidget(self.query)
        
        ir_btn_layout = QHBoxLayout()
        ir_btn_layout.addWidget(self.btn_tfidf)
        ir_btn_layout.addWidget(self.btn_vector)
        ir_layout.addLayout(ir_btn_layout)
        ir_group.setLayout(ir_layout)

        # Add groups to main layout
        main_layout.addWidget(tp_group)
        main_layout.addWidget(cls_group)
        main_layout.addWidget(ir_group)
        main_layout.addWidget(QLabel("Output:"))
        main_layout.addWidget(self.output)

        self.setLayout(main_layout)

        # ---------- Signals ----------
        self.btn_text_processing.clicked.connect(self.do_text_processing)
        self.btn_classification.clicked.connect(self.do_classification)
        self.btn_tfidf.clicked.connect(self.do_tf_idf)
        self.btn_vector.clicked.connect(self.do_vsm)

    # ---------- Functions ----------
    def do_text_processing(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Test File", "", "Text Files (*.txt)")
        if file_path:
            out = process_text(file_path)
            self.output.setText("outputs in: "+ str(out))
          
    def do_classification(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Test File", "", "Text Files (*.txt)")
        if file_path:
            pred, path = self.classification_model.predict(file_path)
            self.output.setText(f"Class: {pred}\n\n ===Evaluation===\n\n{path}")

    def do_tf_idf(self):
        q = self.query.toPlainText().strip()
        if not q:
            return self.output.setText("⚠ Please enter a query first!")
        ranked = self.ir_model.query(q, method="tfidf_sum")
        text = "\n".join([f"{doc} ==> {score:.4f}" for doc, score in ranked[:10]])
        self.output.setText(text)

    def do_vsm(self):
        q = self.query.toPlainText().strip()
        if not q:
            return self.output.setText("⚠ Please enter a query first!")
        ranked = self.ir_model.query(q, method="vsm")
        text = "\n".join([f"{doc} ==> {score:.4f}" for doc, score in ranked[:10]])
        self.output.setText(text)


# ---------- Run ----------
app = QApplication(sys.argv)
window = NLP_GUI()
window.show()
sys.exit(app.exec_())
