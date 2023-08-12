from PyQt6.QtCore import QObject, pyqtSignal

class MyObject(QObject):
    my_signal = pyqtSignal()

def my_slot():
    print("Signal emitted!")

obj = MyObject()

# 连接信号和槽
obj.my_signal.connect(my_slot)

# 发射信号
obj.my_signal.emit()
