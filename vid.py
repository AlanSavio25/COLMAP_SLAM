import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import time
import threading
from pathlib import Path
import os

'''
Class for the video window to display output and keyframes
'''
class VideoWindow:

    def __init__(self):
        self.rgb_images = []    

        self.window = gui.Application.instance.create_window(
            "Keyframes and live video", 1500, 488)
        self.window.set_on_layout(self._on_layout)
        self.window.set_on_close(self._on_close)


        em = self.window.theme.font_size
        margin = 0.5 * em
        self.panel = gui.Horiz(0.5 * em, gui.Margins(margin))

        default_img = (np.zeros((375,500)).astype(np.uint8))

        
        kf_panel = gui.Vert(0.5 * em, gui.Margins(margin))
        self.kf_widget = gui.ImageWidget(o3d.geometry.Image(default_img))
        kf_panel.add_child(self.kf_widget)
        
        settings_panel = gui.Vert(0.5 * em, gui.Margins(margin))

        settings_panel.add_child(gui.Label("Output:"))
        self.out_label = gui.Label("")
        settings_panel.add_child(self.out_label)

        self.panel.add_child(kf_panel)
        self.panel.add_child(settings_panel)


        self.window.add_child(self.panel)

        self.is_done = False


    def _on_layout(self, layout_context):
        contentRect = self.window.content_rect
        self.panel.frame = gui.Rect(contentRect.x, contentRect.y, contentRect.width, contentRect.height)

    def _on_close(self):
        self.is_done = True
        return True  