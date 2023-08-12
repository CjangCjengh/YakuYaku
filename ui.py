import os
import json
import torch
from PyQt6.QtWidgets import QApplication, QMainWindow, QStyleFactory, QWidget, QVBoxLayout, QHBoxLayout, QTextEdit, QLabel, QPushButton, QComboBox, QFormLayout, QDialog, QSpinBox, QFontComboBox, QGridLayout, QListWidget, QFileDialog, QLineEdit, QMessageBox
from PyQt6.QtCore import QTranslator, Qt, QThread, pyqtSignal
from PyQt6.QtGui import QAction
from utils import Translator

class UISettingsDialog(QDialog):
    def __init__(self, parent):
        super().__init__(parent)

        self.setWindowTitle(self.tr("界面设置"))
        parent.translate_func.append([self.setWindowTitle, self, "界面设置"])

        layout = QFormLayout()

        self.language_combo = QComboBox()
        for lang in os.listdir('lang'):
            if lang.endswith('.qm'):
                self.language_combo.addItem(lang[:-3])
        language_label = QLabel(self.tr("界面语言"))
        parent.translate_func.append([language_label.setText, self, "界面语言"])
        layout.addRow(language_label, self.language_combo)

        self.style_combo = QComboBox()
        self.style_combo.addItems(QStyleFactory.keys())
        style_label = QLabel(self.tr("界面风格"))
        parent.translate_func.append([style_label.setText, self, "界面风格"])
        layout.addRow(style_label, self.style_combo)

        self.global_font_combo = QFontComboBox()
        global_font_label = QLabel(self.tr("全局字体"))
        parent.translate_func.append([global_font_label.setText, self, "全局字体"])
        layout.addRow(global_font_label, self.global_font_combo)
        
        self.font_size_spinbox = QSpinBox()
        self.font_size_spinbox.setMinimum(10)
        self.font_size_spinbox.setMaximum(30)
        font_size_label = QLabel(self.tr("全局字号"))
        parent.translate_func.append([font_size_label.setText, self, "全局字号"])
        layout.addRow(font_size_label, self.font_size_spinbox)

        self.original_font_combo = QFontComboBox()
        original_font_label = QLabel(self.tr("原文字体"))
        parent.translate_func.append([original_font_label.setText, self, "原文字体"])
        layout.addRow(original_font_label, self.original_font_combo)

        self.original_font_size_spinbox = QSpinBox()
        self.original_font_size_spinbox.setMinimum(10)
        self.original_font_size_spinbox.setMaximum(30)
        original_font_size_label = QLabel(self.tr("原文字号"))
        parent.translate_func.append([original_font_size_label.setText, self, "原文字号"])
        layout.addRow(original_font_size_label, self.original_font_size_spinbox)

        self.translated_font_combo = QFontComboBox()
        translated_font_label = QLabel(self.tr("译文字体"))
        parent.translate_func.append([translated_font_label.setText, self, "译文字体"])
        layout.addRow(translated_font_label, self.translated_font_combo)

        self.translated_font_size_spinbox = QSpinBox()
        self.translated_font_size_spinbox.setMinimum(10)
        self.translated_font_size_spinbox.setMaximum(30)
        translated_font_size_label = QLabel(self.tr("译文字号"))
        parent.translate_func.append([translated_font_size_label.setText, self, "译文字号"])
        layout.addRow(translated_font_size_label, self.translated_font_size_spinbox)

        self.save_button = QPushButton(self.tr("保存"))
        parent.translate_func.append([self.save_button.setText, self, "保存"])
        self.save_button.clicked.connect(self.save_ui_settings)
        layout.addRow(self.save_button)

        self.setLayout(layout)

    def save_ui_settings(self):
        parent = self.parent()
        settings = parent.settings

        translator = QTranslator()
        selected_language = self.language_combo.currentText()
        settings.setValue('language', f'lang/{selected_language}.qm')
        translator.load(f'lang/{selected_language}.qm')
        QApplication.instance().installTranslator(translator)
        parent.retranslate_ui()

        def set_font(widget, font_size_spinbox, font_combo, label):
            font = widget.font()
            font_size = font_size_spinbox.value()
            settings.setValue(f'{label}_font_size', font_size)
            font_family = font_combo.currentFont()
            settings.setValue(f'{label}_font', font_family.family())
            font.setPointSize(font_size)
            font.setFamily(font_family.family())
            widget.setFont(font)

        set_font(QApplication, self.font_size_spinbox, self.global_font_combo, 'global')
        set_font(parent.original_text_edit, self.original_font_size_spinbox, self.original_font_combo, 'original')
        set_font(parent.translated_text_edit, self.translated_font_size_spinbox, self.translated_font_combo, 'translated')

        style = self.style_combo.currentText()
        settings.setValue('style', style)
        QApplication.setStyle(style)

        self.accept()

class TranslateSettingsDialog(QDialog):
    def __init__(self, parent):
        super().__init__(parent)

        self.setWindowTitle(self.tr("翻译设置"))
        parent.translate_func.append([self.setWindowTitle, self, "翻译设置"])

        layout = QFormLayout()

        self.device_combobox = QComboBox()
        self.device_combobox.addItem("CPU")
        if torch.cuda.is_available():
            self.device_combobox.addItem("CUDA")
        self.device_combobox.setCurrentIndex(0 if parent.device == "cpu" else 1)
        inference_device_label = QLabel(self.tr("推理设备"))
        parent.translate_func.append([inference_device_label.setText, self, "推理设备"])
        layout.addRow(inference_device_label, self.device_combobox)

        self.beam_size_spinbox = QSpinBox()
        self.beam_size_spinbox.setMinimum(1)
        self.beam_size_spinbox.setMaximum(10)
        self.beam_size_spinbox.setValue(parent.beam_size)  
        beam_size_label = QLabel(self.tr("Beam Size"))
        parent.translate_func.append([beam_size_label.setText, self, "Beam Size"])
        layout.addRow(beam_size_label, self.beam_size_spinbox)

        self.save_button = QPushButton(self.tr("保存"))
        parent.translate_func.append([self.save_button.setText, self, "保存"])
        self.save_button.clicked.connect(self.save_translate_settings)
        layout.addRow(self.save_button)

        self.setLayout(layout)

    def save_translate_settings(self):
        parent = self.parent()
        settings = parent.settings
        parent.device = self.device_combobox.currentText().lower()
        settings.setValue('device', parent.device)
        parent.beam_size = self.beam_size_spinbox.value()
        settings.setValue('beam_size', parent.beam_size)
        self.accept()

class BatchTranslateDialog(QDialog):
    def __init__(self, parent):
        super().__init__(parent)
        
        self.source_files = []

        self.setWindowTitle(self.tr("批量翻译"))
        parent.translate_func.append([self.setWindowTitle, self, "批量翻译"])

        layout = QGridLayout()
        self.setLayout(layout)

        select_file_layout = QHBoxLayout()
        select_file_button = QPushButton(self.tr("选择文件"), self)
        parent.translate_func.append([select_file_button.setText, self, "选择文件"])
        select_file_button.clicked.connect(self.select_files)
        self.clear_files_button = QPushButton(self.tr("清空"), self)
        parent.translate_func.append([self.clear_files_button.setText, self, "清空"])
        self.clear_files_button.clicked.connect(self.clear_files)
        select_file_layout.addWidget(select_file_button)
        select_file_layout.addWidget(self.clear_files_button)
        layout.addLayout(select_file_layout, 0, 0)

        output_folder_layout = QHBoxLayout()
        output_folder_button = QPushButton(self.tr("选择输出位置"), self)
        parent.translate_func.append([output_folder_button.setText, self, "选择输出位置"])
        output_folder_button.clicked.connect(self.select_output_folder)
        self.output_folder_textbox = QLineEdit(self)
        self.output_folder_textbox.setReadOnly(True)
        output_folder_layout.addWidget(output_folder_button)
        output_folder_layout.addWidget(self.output_folder_textbox)
        layout.addLayout(output_folder_layout, 0, 1)

        self.source_list_widget = QListWidget(self)
        self.source_list_widget.keyPressEvent = self.delete_selected_file
        layout.addWidget(self.source_list_widget, 1, 0)

        self.target_list_widget = QListWidget(self)
        layout.addWidget(self.target_list_widget, 1, 1)

        start_translation_button = QPushButton(self.tr("开始翻译"), self)
        parent.translate_func.append([start_translation_button.setText, self, "开始翻译"])
        start_translation_button.clicked.connect(self.start_translation)
        layout.addWidget(start_translation_button, 2, 0)

        cancel_button = QPushButton(self.tr("取消"), self)
        parent.translate_func.append([cancel_button.setText, self, "取消"])
        cancel_button.clicked.connect(self.close)
        layout.addWidget(cancel_button, 2, 1)

    def select_output_folder(self):
        folder = QFileDialog.getExistingDirectory(self, self.tr("选择输出位置"))
        if folder:
            self.output_folder = folder
            self.output_folder_textbox.setText(folder)

    def select_files(self):
        files, _ = QFileDialog.getOpenFileNames(self, self.tr("选择文件"), filter=self.tr("文本文件 (*.txt)"))
        if files:
            files = [f for f in files if f not in self.source_files]
            self.source_files.extend(files)
            self.source_list_widget.addItems(files)

    def clear_files(self):
        self.source_files.clear()
        self.source_list_widget.clear()

    def delete_selected_file(self, event):
        if event.key() in [Qt.Key.Key_Delete, Qt.Key.Key_Backspace]: 
            selected_items = self.source_list_widget.selectedItems()
            for item in selected_items:
                row = self.source_list_widget.row(item)
                self.source_list_widget.takeItem(row)
                self.source_files.pop(row)
        else:
            super().keyPressEvent(event)

    def start_translation(self):
        # Perform batch translation logic here
        pass

class MainWindow(QMainWindow):
    def __init__(self, settings):
        super().__init__()
        self.translate_func = []
        self.max_text_length = 0
        self.beam_size = 3
        self.device = "cpu"
        self.settings = settings
        self.translator = None
        self.init_ui()
        self.init_settings()

    def init_ui(self):
        self.setWindowTitle(self.tr("翻译姬"))
        self.translate_func.append([self.setWindowTitle, self, "翻译姬"])
        self.setGeometry(100, 100, 800, 600)
        
        self.ui_settings_dialog = UISettingsDialog(self)
        self.translate_settings_dialog = TranslateSettingsDialog(self)
        self.batch_translate_dialog = BatchTranslateDialog(self)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)

        tool_bar = self.addToolBar(self.tr("工具栏"))
        self.translate_func.append([tool_bar.setWindowTitle, self, "工具栏"])

        ui_settings_button = QAction(self.tr("界面设置"))
        self.translate_func.append([ui_settings_button.setText, self, "界面设置"])
        ui_settings_button.triggered.connect(self.show_ui_settings_dialog)
        tool_bar.addAction(ui_settings_button)

        translate_settings_button = QAction(self.tr('翻译设置'))
        self.translate_func.append([translate_settings_button.setText, self, "翻译设置"])
        translate_settings_button.triggered.connect(self.show_translate_settings_dialog)
        tool_bar.addAction(translate_settings_button)

        batch_translate_action = QAction(self.tr("批量翻译"), self)
        self.translate_func.append([batch_translate_action.setText, self, "批量翻译"])
        batch_translate_action.triggered.connect(self.show_batch_translate_dialog)
        tool_bar.addAction(batch_translate_action)

        self.model_label = QLabel(self.tr("选择模型"))
        self.translate_func.append([self.model_label.setText, self, "选择模型"])
        main_layout.addWidget(self.model_label)
        self.model_combo = QComboBox()
        self.load_model_list(self.model_combo)
        self.model_combo.setCurrentIndex(-1)
        self.model_combo.currentIndexChanged.connect(self.load_model)
        main_layout.addWidget(self.model_combo)

        original_text_label_layout = QHBoxLayout()
        original_text_label = QLabel(self.tr("原文"))
        self.translate_func.append([original_text_label.setText, self, "原文"])
        self.text_count_label = QLabel("0/0")
        self.text_count_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        original_text_label_layout.addWidget(original_text_label)
        original_text_label_layout.addWidget(self.text_count_label)
        self.original_text_edit = QTextEdit()
        self.original_text_edit.setAcceptRichText(False)
        self.original_text_edit.textChanged.connect(self.check_text_limit)

        translated_text_label_layout = QHBoxLayout()
        translated_text_label = QLabel(self.tr("译文"))
        self.translate_func.append([translated_text_label.setText, self, "译文"])
        translated_index_layout = QFormLayout()
        translated_index_label = QLabel(self.tr("序号"))
        self.translate_func.append([translated_index_label.setText, self, "序号"])
        translated_index_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.translated_index_combo = QComboBox()
        self.translated_index_combo.currentIndexChanged.connect(self.load_translated_text)
        translated_index_layout.addRow(translated_index_label, self.translated_index_combo)
        translated_text_label_layout.addWidget(translated_text_label)
        translated_text_label_layout.addLayout(translated_index_layout)
        self.translated_text_edit = QTextEdit()
        self.translated_text_edit.setReadOnly(True)

        text_layout = QGridLayout()
        text_layout.addLayout(original_text_label_layout, 0, 0)
        text_layout.addLayout(translated_text_label_layout, 0, 1)
        text_layout.addWidget(self.original_text_edit, 1, 0)
        text_layout.addWidget(self.translated_text_edit, 1, 1)

        main_layout.addLayout(text_layout)

        translate_button = QPushButton(self.tr("翻译"))
        self.translate_func.append([translate_button.setText, self, "翻译"])
        translate_button.clicked.connect(self.translate)
        main_layout.addWidget(translate_button)

    def init_settings(self):
        def init_font(widget, label):
            font = widget.font()
            if self.settings.contains(f'{label}_font_size'):
                font.setPointSize(int(self.settings.value(f'{label}_font_size')))
            if self.settings.contains(f'{label}_font'):
                font.setFamily(self.settings.value(f'{label}_font'))
            widget.setFont(font)

        if self.settings.contains('device'):
            self.device = self.settings.value('device')
        if self.settings.contains('beam_size'):
            self.beam_size = self.settings.value('beam_size')

        init_font(QApplication, 'global')
        init_font(self.original_text_edit, 'original')
        init_font(self.translated_text_edit, 'translated')

    def show_ui_settings_dialog(self):
        self.ui_settings_dialog.exec()

    def show_translate_settings_dialog(self):
        self.translate_settings_dialog.exec()

    def show_batch_translate_dialog(self):
        self.batch_translate_dialog.exec()

    def retranslate_ui(self):
        for func, context, text in self.translate_func:
            func(context.tr(text))

    def check_text_limit(self):
        current_length = len(self.original_text_edit.toPlainText())

        if self.max_text_length - current_length >= 0:
            self.text_count_label.setText(f"{current_length}/{self.max_text_length}")
        else:
            self.original_text_edit.setPlainText(self.original_text_edit.toPlainText()[:self.max_text_length])

    def load_model_list(self, combo_box):
        for model_dir in os.listdir('models'):
            try:
                with open(f'models/{model_dir}/config.json', 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    combo_box.addItem(config['name'], model_dir)
            except:
                continue

    def load_model(self, index):
        try:
            self.translator = Translator(f'models/{self.model_combo.currentData()}', self.device)
            if self.translator.tokenizer is None:
                QMessageBox.critical(self, self.tr("错误"),
                                     self.tr("当前版本的翻译姬中不含有{}，请更新至最新版本")
                                     .format(self.translator.config['tokenizer']))
                return
            if self.translator.cleaner is None:
                QMessageBox.critical(self, self.tr("错误"),
                                     self.tr("当前版本的翻译姬中不含有{}，请更新至最新版本")
                                     .format(self.translator.config['cleaner']))
                return
            self.max_text_length = self.translator.config['max_len']
            self.text_count_label.setText(f"{len(self.original_text_edit.toPlainText())}/{self.max_text_length}")
        except:
            QMessageBox.critical(self, self.tr("错误"), self.tr("模型加载失败"))

    def get_dialog(self, title, text, close_func=None):
        dialog = QDialog(self)
        dialog.setWindowTitle(title)
        dialog.setGeometry(100, 100, 100, 50)
        layout = QVBoxLayout()
        dialog.setLayout(layout)
        layout.addWidget(QLabel(text))
        if close_func:
            dialog.finished.connect(close_func)
        dialog.move(self.geometry().center() - dialog.rect().center())
        return dialog

    def translate(self):
        if self.translator is None:
            return
        original_text = self.original_text_edit.toPlainText()

        info_dialog = self.get_dialog(self.tr("翻译中"), self.tr("正在翻译..."), self.translator.terminate)

        def _translate():
            self.translator._is_terminated = False
            translated_text = self.translator.translate(original_text, self.beam_size, self.device)
            if translated_text is None:
                return
            self.translated_index_combo.clear()
            for idx, text in enumerate(translated_text):
                self.translated_index_combo.addItem(str(idx+1), text)
            self.translated_index_combo.setCurrentIndex(0)

        self.translation_thread = QThread()
        self.translation_thread.run = _translate
        self.translation_thread.finished.connect(info_dialog.close)
        self.translation_thread.start()
        
        info_dialog.exec()

    def load_translated_text(self, index):
        self.translated_text_edit.setPlainText(self.translated_index_combo.currentData())
