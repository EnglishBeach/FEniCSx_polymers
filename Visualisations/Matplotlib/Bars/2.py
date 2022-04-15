import PySimpleGUI as sg
import time

for i in range(100):
    sg.one_line_progress_meter('This is my progress meter!', i+1, 100, '-key-')
    time.sleep(0.1)

