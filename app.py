import argparse
from getopt import getopt
import threading
from time import sleep
from pipeline import Pipeline
import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
from src import enums, viz
import os
from pathlib import Path
from vid import VideoWindow
import cv2
import matplotlib.cm as cm
# from PIL import Image


'''
For WSL need to install mesaos
    
    sudo apt-get install libosmesa6-dev

and set opengl version with

    export LIBGL_ALWAYS_INDIRECT=0
    export MESA_GL_VERSION_OVERRIDE=4.5
    export MESA_GLSL_VERSION_OVERRIDE=450
    export LIBGL_ALWAYS_SOFTWARE=1

'''

class AppWindow:

    def __init__(self, width, height, data_path='', skip_callback=False):

        self.window = gui.Application.instance.create_window("COLMAP Slam", width, height)

        # Default config stuff
        self.img_count = 0
        self.pt_count = -1
        self.frame_skip = 2
        self.extractor = enums.Extractors(1)
        self.matcher = enums.Matchers(1)
        self.image_path = data_path
        self.output_path = "./out/test1"
        self.export_name = "reconstruction.ply"
        self.frames = []
        self.init_frames = 30
        self.flow_thresh = 0.05

        self.per_frame = skip_callback

        try:
            self.raw_img_count = len(os.listdir(data_path))
        except:
            print("Error opening path")
            self.raw_img_count = 0

        self.frame_final = self.raw_img_count

        self.rec = Pipeline()
        self.show_cam = True
        self.show_path = True
        self.show_track = -1
        self.cam_scale = 10
        self.is_setup = False
        self.start_img = 0
        self.end_img = 0
        self.last_keyframe = 0
        self.current_frame = 0
        
        # default material
        self.mat = o3d.visualization.rendering.MaterialRecord()
        self.mat.shader = "defaultUnlit"
        self.mat.base_color = (1, 1, 1, 1)
        self.mat.base_reflectance = 0.1

        self.mat.point_size = 10 * self.window.scaling

        w = self.window 
        self.vid = VideoWindow()
        
        em = w.theme.font_size
        separation_height = int(round(0.5 * em))

        # Reconstruction 3d widget
        self._scene = gui.SceneWidget()
        self._scene.scene = rendering.Open3DScene(w.renderer)
        self._scene.set_view_controls(gui.SceneWidget.Controls.ROTATE_CAMERA)


        # Create the settings panel on the right
        self._settings_panel = gui.Vert(0, gui.Margins(0.25 * em, 0.25 * em, 0.25 * em, 0.25 * em))

        # Top bar to load/save reconstructions
        rec_loader = gui.CollapsableVert("Reconstruction Loader", 0.25 * em,
                                         gui.Margins(em, 0, 0, 0))

        rec_loader.set_is_open(False)
        rec_horz = gui.Horiz(0.5 * em, gui.Margins(0.5*em))
        _open_rec = gui.Button("Open Rec")
        _open_rec.set_on_clicked(self._load_rec)
        rec_horz.add_child(_open_rec)

        _export_rec = gui.Button("Export Rec")
        _export_rec.set_on_clicked(self._save_rec)
        rec_horz.add_child(_export_rec)
        
        rec_loader.add_child(rec_horz)

        self._settings_panel.add_fixed(separation_height)
        self._settings_panel.add_child(rec_loader)
        self._settings_panel.add_fixed(separation_height)


        # Main reconstruction settings
        ## Change data paths
        _data_loading = gui.Horiz(0,  gui.Margins(0.25 * em, 0.25 * em, 0.25 * em, 0.25 * em))
        self._settings_panel.add_child(gui.Label("Data Settings"))
        self._data_path = gui.Label(f"Data Path:\n {self.image_path}")
        self._settings_panel.add_child(self._data_path)

        self._out_path = gui.Label(f"Output Path:\n {self.output_path}")
        self._settings_panel.add_child(self._out_path)

        self._settings_panel.add_child(gui.Label("Edit Path"))
        _data_selector = gui.Button("Data")
        _data_selector.set_on_clicked(self._on_data_open)

        _output_selector = gui.Button("Output")
        _output_selector.set_on_clicked(self._on_out_open)

        _data_loading.add_child(_data_selector)
        _data_loading.add_child(_output_selector)

        self._settings_panel.add_child(_data_loading)
        
        ## Frame skip slider
        self._settings_panel.add_child(gui.Label("Number of frames to skip"))
        _frame_skip = gui.Slider(gui.Slider.INT)
        _frame_skip.set_limits(1, 40)
        _frame_skip.int_value = self.frame_skip
        _frame_skip.set_on_value_changed(self._on_frame_skip)
        self._settings_panel.add_child(_frame_skip)

        # Number of frames of the sequence to process
        self._settings_panel.add_child(gui.Label("Max number of frames"))
        self._frame_final = gui.Slider(gui.Slider.INT)
        self._frame_final.set_limits(0, self.raw_img_count // self.frame_skip)
        self._frame_final.int_value = self.raw_img_count // self.frame_skip
        self._frame_final.set_on_value_changed(self._on_frame_final)
        self._settings_panel.add_child(self._frame_final)

        # Frames for init
        self._settings_panel.add_child(gui.Label("Max frames for initialization"))
        _init_frames = gui.Slider(gui.Slider.INT)
        _init_frames.set_limits(0, 60)
        _init_frames.int_value = self.init_frames
        _init_frames.set_on_value_changed(self._on_init_frames)
        self._settings_panel.add_child(_init_frames)

        # Optical flow threshold setter
        self.add_slider("Optical Flow Threshold", .01, .15, self._on_set_thresh, self._settings_panel, gui.Slider.DOUBLE, self.flow_thresh)

        # Basic reconstruction settings
        self._settings_panel.add_child(gui.Label("Reconstruction Settings"))
        

        # Feature extractor
        _extractor = gui.Combobox()
        for name, _ in enums.Extractors.__members__.items():
            _extractor.add_item(name)

        _extractor.set_on_selection_changed(self._on_extractor)
        self._settings_panel.add_fixed(separation_height)
        self._settings_panel.add_child(gui.Label("Feature Extractor"))
        self._settings_panel.add_child(_extractor)

        # Feature matcher
        _matcher = gui.Combobox()
        for name, _ in enums.Matchers.__members__.items():
            _matcher.add_item(name)

        _matcher.set_on_selection_changed(self._on_matcher)
        self._settings_panel.add_fixed(separation_height)
        self._settings_panel.add_child(gui.Label("Feature Matcher"))
        self._settings_panel.add_child(_matcher)

        # Button to run the reconstruction
        _run = gui.Button("Run")
        _run.set_on_clicked(self._run_reconstruction)
        self._settings_panel.add_fixed(separation_height)
        self._settings_panel.add_child(_run)


        # Next add settings for the post reconstruction results
        view_ctrls = gui.CollapsableVert("Viz Settings", 0.25 * em,
                                         gui.Margins(em, 0, 0, 0))

        view_ctrls.set_is_open(False)

        # Changes the color of the reconstruction viewer
        view_ctrls.add_child(gui.Label("Background Color"))
        _bg_color = gui.ColorEdit()
        _bg_color.color_value = gui.Color(1,1,1)
        _bg_color.set_on_value_changed(self._on_bg_color)
        view_ctrls.add_child(_bg_color)

        # Shows camera path path through the scene
        _show_cam_path = gui.Checkbox("Show Camera Path")
        _show_cam_path.set_on_checked(self._on_show_path)
        _show_cam_path.checked = self.show_path
        view_ctrls.add_fixed(separation_height)
        view_ctrls.add_child(_show_cam_path)

        # Toggles cameras
        _show_cams = gui.Checkbox("Show Cameras")
        _show_cams.set_on_checked(self._on_show_cams)
        _show_cams.checked = self.show_cam
        view_ctrls.add_fixed(separation_height)
        view_ctrls.add_child(_show_cams)

        # Shows tracks for a given point
        view_ctrls.add_child(gui.Label("Show Camera Track"))
        self._camera_tracks = gui.Slider(gui.Slider.INT)
        self._camera_tracks.set_limits(-1, self.pt_count)
        self._camera_tracks.int_value = self.show_track
        self._camera_tracks.set_on_value_changed(self._on_show_tracks)
        view_ctrls.add_child(self._camera_tracks)

        # Change camera scale
        view_ctrls.add_child(gui.Label("Camera Scale"))
        _cam_size = gui.Slider(gui.Slider.DOUBLE)
        _cam_size.set_limits(0, 50)
        _cam_size.set_on_value_changed(self._on_cam_scale)
        _cam_size.double_value = self.cam_scale
        view_ctrls.add_child(_cam_size)

        # Changes the size of the points
        view_ctrls.add_child(gui.Label("Point Scale"))
        _pt_size = gui.Slider(gui.Slider.DOUBLE)
        _pt_size.set_limits(3, 50)
        _pt_size.set_on_value_changed(self._on_pt_scale)
        _pt_size.double_value = 10
        view_ctrls.add_child(_pt_size)

        # Resets the camera angle
        _reset_view = gui.Button("Reset Camera")
        _reset_view.set_on_clicked(self._reset_view)
        view_ctrls.add_fixed(separation_height)
        view_ctrls.add_child(_reset_view)

        view_ctrls.add_fixed(separation_height)

        # Following two restrict reconstruction viz to only a portion of the data
        view_ctrls.add_child(gui.Label("Start Image"))
        self._start_img = gui.Slider(gui.Slider.INT)
        self._start_img.set_limits(0, self.img_count)
        self._start_img.set_on_value_changed(self._on_start_img)
        view_ctrls.add_child(self._start_img)

        view_ctrls.add_child(gui.Label("End Image"))
        self._end_img = gui.Slider(gui.Slider.INT)
        self._end_img.set_limits(0, self.img_count)
        self._end_img.int_value = self.img_count
        self._end_img.set_on_value_changed(self._on_end_img)
        view_ctrls.add_child(self._end_img)


        self._settings_panel.add_fixed(separation_height)
        self._settings_panel.add_child(view_ctrls)
        
        # This is from the tutorial
        # http://www.open3d.org/docs/release/python_example/visualization/index.html#vis-gui-py
        # to render the panel on top of the scene and scale stuff as teh window size changes
        w.set_on_layout(self._on_layout)
        w.add_child(self._scene)
        w.add_child(self._settings_panel)

    # Function to create a slider in one line
    def add_slider(self, label, min, max, callback, parent, type=gui.Slider.INT, start_val=0):
        parent.add_child(gui.Label(label))
        slider = gui.Slider(type)
        slider.set_limits(min, max)
        slider.double_value = start_val
        slider.set_on_value_changed(callback)
        parent.add_child(slider)

    # Bunch of event listeners for different settings above changing
    def _on_set_thresh(self, val):
        self.flow_thresh = val

    def _on_frame_final(self, val):
        self.frame_final = int(val)

    def _on_frame_skip(self, val):
        self.frame_skip = int(val)
        self._frame_final.int_value = self.raw_img_count // self.frame_skip
        self._frame_final.set_limits(0, self.raw_img_count // self.frame_skip)
        
    def _on_init_frames(self, val):
        self.init_frames = int(val)

    ## Saves the reconstruction object to a selected folder
    def _save_rec(self):
        dlg = gui.FileDialog(gui.FileDialog.OPEN_DIR, "Choose folder to save reconstruction to", self.window.theme)

        dlg.set_on_cancel(self._on_file_dialog_cancel)
        dlg.set_on_done(self._on_rec_save_done)
        self.window.show_dialog(dlg)

    def _on_rec_save_done(self, filename):
        self.rec.reconstruction.write(filename)
        self.window.close_dialog()
        print(f"outputed to {os.path.join(filename,'.bin')}")

    ## Loads reconstruction from files
    def _load_rec(self):
        dlg = gui.FileDialog(gui.FileDialog.OPEN, "Select reconstruction to load", self.window.theme)
        dlg.add_filter('.bin', 'Reconstruction Binary')

        dlg.set_on_cancel(self._on_file_dialog_cancel)
        dlg.set_on_done(self._on_rec_load_done)
        self.window.show_dialog(dlg)


    def _on_rec_load_done(self, filename):
        print("Trying to load cameras, images, and points .bin")
        try:
            self.rec.reconstruction.read(os.path.dirname(filename))
        except:
            print(f"error... loading ")

        print(f"loaded {filename}")
        print(self.rec.reconstruction.summary())
        self.window.close_dialog()
        self.refresh()


    def _on_layout(self, layout_context):
        # The on_layout callback should set the frame (position + size) of every
        # child correctly. After the callback is done the window will layout
        # the grandchildren.
        r = self.window.content_rect
        self._scene.frame = r
        width = 17 * layout_context.theme.font_size
        height = min(r.height, self._settings_panel.calc_preferred_size(layout_context, gui.Widget.Constraints()).height)
        
        self._settings_panel.frame = gui.Rect(r.get_right() - width, r.y, width, height)

    def _on_extractor(self, name, idx):
        print(f"now using {name}")
        self.extractor = enums.Extractors(idx+1)

    def _on_matcher(self, name, idx):
        print(f"now using {name}")
        self.matcher = enums.Matchers(idx+1)

    def _on_show_path(self, show):
        if self._scene.scene.has_geometry("__path__"):
            self._scene.scene.remove_geometry("__path__")

        if show and self.start_img < self.end_img:
            path = viz.generate_path(self.rec.reconstruction, lambda img: img > self.start_img and img < self.end_img)

            # path = viz.generate_path(self.rec.reconstruction)  # uncomment to disable filtering if there are OpenGL errors
            if len(path.points) > 0:
                self._scene.scene.add_geometry("__path__", path, self.mat)

    def _on_show_tracks(self, pt_id):
        if self._scene.scene.has_geometry("__track__"):
            self._scene.scene.remove_geometry("__track__")

        if pt_id > 0:
            track = viz.generate_tracks(self.rec.reconstruction, int(pt_id),lambda elem: elem.image_id > self.start_img and elem.image_id < self.end_img)
            # track = viz.generate_tracks(self.rec.reconstruction, int(pt_id))  # Uncomment to disable filtering  if there are OpenGL errors

            if len(track.points) > 0:
                self._scene.scene.add_geometry("__track__", track, self.mat)

    def _on_show_cams(self, show):
        if self._scene.scene.has_geometry("__cams__"):
            self._scene.scene.remove_geometry("__cams__")

        if show and self.start_img < self.end_img:
            cams = viz.generate_cams(self.rec.reconstruction, self.cam_scale, lambda img: img > self.start_img and img < self.end_img)
            # cams = viz.generate_cams(self.rec.reconstruction, self.cam_scale)  # Uncomment to disable filtering  if there are OpenGL errors

            if len(cams.points) > 0:
                self._scene.scene.add_geometry(f"__cams__", cams, self.mat)

    def _on_cam_scale(self, val):
        self.cam_scale = val
        self._on_show_cams(self.show_cam)

    def _on_pt_scale(self, val):
        self.mat.point_size = val * self.window.scaling
        self._scene.scene.update_material(self.mat)

    def _on_start_img(self, val):
        self.start_img = val
        self._end_img.set_limits(val, self.img_count)
        self.update_pts()
        self._on_show_cams(self.show_cam)
        self._on_show_path(self.show_path)
        self._on_show_tracks(self.show_track)

    def _on_end_img(self, val):
        self.end_img = val
        self._start_img.set_limits(0,val)
        self.update_pts()
        self._on_show_cams(self.show_cam)
        self._on_show_path(self.show_path)
        self._on_show_tracks(self.show_track)

    def _on_bg_color(self, new_color):
        self._scene.scene.set_background([new_color.red, new_color.green, new_color.blue, new_color.alpha])

    def _on_point_size(self, size):
        self._point_size.double_value = int(size)

    ## Reset the camera view to just behind the center of the tracked cameras
    def _reset_view(self):
        # Replace with behind last camera?
        pt_bounds = viz.generate_pts(self.rec.reconstruction).get_axis_aligned_bounding_box()
        cam_bounds = viz.generate_cams(self.rec.reconstruction, self.cam_scale).get_oriented_bounding_box()
        self._scene.look_at(pt_bounds.get_center(), cam_bounds.get_center() + (cam_bounds.get_center() - pt_bounds.get_center())/3 , np.array([0,0,1])@cam_bounds.R)


    ## File selector callbacks
    def _on_data_open(self):
        dlg = gui.FileDialog(gui.FileDialog.OPEN_DIR, "Choose root folder of image data", self.window.theme)

        dlg.set_on_cancel(self._on_file_dialog_cancel)
        dlg.set_on_done(self._on_data_dialog_done)
        self.window.show_dialog(dlg)

    def _on_out_open(self):
        dlg = gui.FileDialog(gui.FileDialog.OPEN_DIR, "Choose output folder for reconstruction", self.window.theme)

        dlg.set_on_cancel(self._on_file_dialog_cancel)
        dlg.set_on_done(self._on_out_dialog_done)
        self.window.show_dialog(dlg)

    def _on_file_dialog_cancel(self):
        self.window.close_dialog()

    def _on_data_dialog_done(self, filename):
        self.window.close_dialog()
        self.image_path = filename
        print(filename)

        try:
            self.raw_img_count = len(os.listdir(filename))

        except:
            print("Error opening path")
            self.raw_img_count = 0

        
        self._frame_final.set_limits(0, self.raw_img_count // self.frame_skip)
        self._frame_final.int_value = self.raw_img_count // self.frame_skip

        self._data_path.text = 'Data Path:\n' + '/'.join(filename.split('/')[-2:])

    def _on_out_dialog_done(self, filename):
        self.window.close_dialog()
        self.output_path = filename
        print(filename)
        self._out_path.text = 'Output Path:\n'+'/'.join(filename.split('/')[-2:])

    # Starts the running of the reconstruction, clears old data and sets things up
    def _run_reconstruction(self):
        print("Checking settings....")
        
        self.is_setup = False
        self.rec.extractor = self.extractor
        self.rec.matcher = self.matcher

        self.rec.load_data(self.image_path, self.output_path, self.export_name, init_max_num_images=int(self.init_frames), frame_skip=int(self.frame_skip), max_frame=int(self.frame_final))
        # self.rec.reset()


        self.frames = []
        self.imgs = []
        self.last_keyframe = 0 

        # Start the reconstruction in a new thread
        threading.Thread(target=self.reconstruct).start()

    ## Updates the points in the reconstruction
    def update_pts(self):
        if self._scene.scene.has_geometry("__recon__"):
            self._scene.scene.remove_geometry(f"__recon__")
        
        pts = viz.generate_pts(self.rec.reconstruction, self.rec.image_path)#, lambda pt: len([e for e in pt.track.elements if e.image_id > self.start_img and e.image_id < self.end_img ]) > 0)
        self._scene.scene.add_geometry("__recon__", pts, self.mat)

    ## Refresh the UI
    def refresh(self):
        if len(self.rec.reconstruction.images) == 0:
            self.img_count = 0
        else:
            self.img_count = max(self.rec.reconstruction.images)
        self.pt_count = len(self.rec.reconstruction.points3D)
        self.end_img  = self.img_count


        self._start_img.set_limits(0,self.img_count)
        self._end_img.set_limits(0,self.img_count)
        self._end_img.int_value = self.img_count
        self._camera_tracks.set_limits(-1, self.pt_count)

        
        self._on_show_cams(self.show_cam)
        self._on_show_path(self.show_path)
        self._on_show_tracks(self.show_track)

        self.update_pts()
        
        if not self.is_setup:
            self.is_setup = True
            
            self._reset_view()


    # Callback to run when each frame is processed
    def process_frame(self, keyframe_id, display_img):
        print(f"Next keyframe: {keyframe_id}, rec has {len(self.rec.reconstruction.points3D)}")
        self.pt_count = len(self.rec.reconstruction.points3D)
        self._end_img.set_limits(0,self.img_count)
        self.refresh_counter += 1
        if not self.vid:
            print("ERROR: no video window init")
            return 0

        if display_img is not None:
            self.refresh_counter+=1
            self.kf_img = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)

        if self.refresh_counter > 3:
            # Update camera view every 3 registrations
            self.refresh_counter = 0
            self.is_setup = False


        gui.Application.instance.post_to_main_thread(self.window, self.refresh)
        gui.Application.instance.post_to_main_thread(self.vid.window, self.update_output)

    # Updates the frame/output window
    def update_output(self):
        out = self.rec.reconstruction.summary()
        self.vid.kf_widget.update_image(o3d.geometry.Image(self.kf_img.astype(np.uint8)))
        self.vid.out_label.text = out

    # Threaded function to actually call reconstruction.run
    def reconstruct(self):
        print(f"Running on data at {self.image_path} with {self.extractor.name} and {self.matcher.name}...")

        self._scene.scene.clear_geometry()
        self.refresh_counter = 0

        print(self.per_frame)
        if self.per_frame:
            self.rec.run(per_frame_callback=self.process_frame, optical_flow_threshold=self.flow_thresh)
        else:
            self.rec.run( optical_flow_threshold=self.flow_thresh)

        print(self.rec.reconstruction.images.keys())
        print(self.rec.reconstruction.reg_image_ids())

        self.refresh()

def main(skip_callback=False, data_dir='./data/rgbd_dataset_freiburg2_xyz/rgb/'):
    gui.Application.instance.initialize()

    AppWindow(1500, 827, data_dir, skip_callback)

    gui.Application.instance.run()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run COLMAP-SLAM on a video sequence.")
    parser.add_argument('-f', '--fast', action='store_false', help="Optional flag to run without a per-frame callback, helps on older systems/WSL systems if OpenGL crashes")
    parser.add_argument('-d', '--dir', nargs='?', default='./data/kitti/frames/', help="Starting data directory for the application")

    args = parser.parse_args()
    main(args.fast, args.dir)