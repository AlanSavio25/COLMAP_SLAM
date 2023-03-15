import pycolmap
import pyceres
import numpy as np
import cv2
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt

from Utility.logger_setup import get_logger
from src.optimization import BundleAdjuster

logger = get_logger(__name__)


# Class for defining object which sets options for IncrementalMapper 
class IncrementalMapperOptions:
    # Minimum number of inliers for initial image pair.
    init_min_num_inliers = 100

    # Maximum error in pixels for two-view geometry estimation for initial
    # image pair.
    init_max_error = 4.0

    # Maximum forward motion for initial image pair.
    init_max_forward_motion = 0.95

    # Minimum triangulation angle for initial image pair.
    init_min_tri_angle = 16.0

    # Maximum number of trials to use an image for initialization.
    init_max_reg_trials = 2

    # Maximum reprojection error in absolute pose estimation.
    abs_pose_max_error = 12.0

    # Minimum number of inliers in absolute pose estimation.
    abs_pose_min_num_inliers = 30

    # Minimum inlier ratio in absolute pose estimation.
    abs_pose_min_inlier_ratio = 0.25

    # Whether to estimate the focal length in absolute pose estimation.
    abs_pose_refine_focal_length = True

    # Whether to estimate the extra parameters in absolute pose estimation.
    abs_pose_refine_extra_params = True

    # Number of images to optimize in local bundle adjustment.
    local_ba_num_images = 6

    # Minimum triangulation for images to be chosen in local bundle adjustment.
    local_ba_min_tri_angle = 6

    # Thresholds for bogus camera parameters.Images with bogus camera
    # parameters are filtered and ignored in triangulation.
    min_focal_length_ratio = 0.1  # Opening angle of ~130deg
    max_focal_length_ratio = 10  # Opening angle of ~5deg
    max_extra_param = 1

    # Maximum reprojection error in pixels for observations.
    filter_max_reproj_error = 4.0

    # Minimum triangulation angle in degrees for stable 3D points.
    filter_min_tri_angle = 1.5

    # Maximum number of trials to register an image.
    max_reg_trials = 3

    # If reconstruction is provided as input fix the existing image poses.
    fix_existing_images = False

    # Number of threads.
    num_threads = -1

    # Number of images to check for finding InitialImage pair
    init_max_num_images = 60

    optical_flow_threshold = 0.05

    def check(self):
        return True

# Class that controls incremental mapping and wraps the core functions of the code
class IncrementalMapper:
    # =========================== "private" ===============================

    # Reconstruction object
    reconstruction_ = None

    # Correspondece graph object
    graph_ = None

    # Incremental triangulation object
    triangulator_ = None

    # Image manager class
    images_manager_ = None

    # Invalid imageId
    kInvalidImageId = -1

    # Number of images that are registered in at least on reconstruction.
    num_total_reg_images_ = 0

    # Number of shared images between current reconstruction and all other
    # previous reconstructions.
    num_shared_reg_images_ = 0

    # Estimated two-view geometry of last call to `FindFirstInitialImage`,
    # used as a cache for a subsequent call to `RegisterInitialImagePair`.
    prev_init_image_pair_id_ = (-1, -1)
    prev_init_two_view_geometry_ = None

    # Images and image pairs that have been used for initialization. Each image
    # and image pair is only tried once for initialization.
    init_num_reg_trials_ = {}
    init_image_pairs_ = []

    # The number of registered images per camera. This information is used
    # to avoid duplicate refinement of camera parameters and degradation of
    # already refined camera parameters in local bundle adjustment when multiple
    # images share intrinsics.
    num_reg_images_per_camera_ = {}

    # The number of reconstructions in which images are registered.
    num_registrations_ = {}

    # Images that have been filtered in current reconstruction.
    filtered_images_ = None

    # Number of trials to register image in current reconstruction. Used to set
    # an upper bound to the number of trials to register an image.
    num_reg_trials_ = {}

    # Images that were registered before beginning the reconstruction.
    # This image list will be non-empty, if the reconstruction is continued from
    # an existing reconstruction.
    existing_image_ids_ = None

    # This function does not seem to get exposed from pycolmap
    def DegToRad(self, deg):
        return deg * np.pi / 180

    # Find seed images for incremental reconstruction. Suitable seed images have
    # a large number of correspondences and have camera calibration priors. The
    # returned list is ordered such that most suitable images are in the front.
    def FindFirstInitialImage(self, options):

        init_max_reg_trials = options.init_max_reg_trials

        # Collect information of all not yet registered images with
        # correspondences. We consider only the options.
        image_infos = []

        max_len = min(self.reconstruction_.num_images(), options.init_max_num_images)
        # max_len = min(self.reconstruction_.num_images(), int(options.init_max_num_images * 2))
        for image in [self.reconstruction_.images[img_id] for img_id in self.images_manager_.image_ids[:max_len]]:
            # Only images with correspondences can be registered.
            if image.num_correspondences == 0:
                continue

            # Only use images for initialization a maximum number of times.
            if self.init_num_reg_trials_.get(image.image_id, 0) >= options.init_max_reg_trials:
                continue

            # Only use images for initialization that are not registered in any
            # of the other reconstructions.
            if self.num_registrations_.get(image.image_id, 0) > 0:
                continue

            camera = self.reconstruction_.cameras[image.camera_id]
            image_info = {
                "image_id": image.image_id,
                "prior_focal_length": camera.has_prior_focal_length,
                "num_correspondences": image.num_correspondences
            }
            image_infos.append(image_info)

        # Sort images such that images with a prior focal length and more
        # correspondences are preferred, i.e. they appear in the front of the list.
        image_infos = sorted(image_infos, key=lambda d: (not d["prior_focal_length"], d["num_correspondences"]),
                             reverse=True)

        # Extract image identifiers in sorted order.
        image_ids = [image_info["image_id"] for image_info in image_infos]

        return image_ids

    # For a given first seed image, find other images that are connected to the
    # first image. Suitable second images have a large number of correspondences
    # to the first image and have camera calibration priors. The returned list is
    # ordered such that most suitable images are in the front.
    def FindSecondInitialImage(self, options, image_id1):
        image1 = self.reconstruction_.images[image_id1]
        num_correspondences = {}

        for point2D_idx in range(image1.num_points2D()):
            for corr in self.graph_.find_correspondences(image_id1, point2D_idx):
                if self.num_registrations_.get(corr.image_id, 0) == 0:
                    num_correspondences[corr.image_id] = num_correspondences.get(corr.image_id, 0) + 1

        init_min_num_inliers = options.init_min_num_inliers
        image_infos = []
        for k, v in num_correspondences.items():
            if v >= init_min_num_inliers:
                image = self.reconstruction_.images[k]
                camera = self.reconstruction_.cameras[image.camera_id]
                image_info = {
                    "image_id": k,
                    "prior_focal_length": camera.has_prior_focal_length,
                    "num_correspondences": v
                }
                image_infos.append(image_info)
        # Sort images such that images with a prior focal length and more
        # correspondences are preferred, i.e. they appear in the front of the list.
        image_infos = sorted(image_infos, key=lambda d: (not d["prior_focal_length"], d["num_correspondences"]),
                             reverse=True)
        # Extract image identifiers in sorted order.
        image_ids = [image_info["image_id"] for image_info in image_infos]
        return image_ids

    # Find local bundle for given image in the reconstruction. The local bundle
    # is defined as the images that are most connected, i.e. maximum number of
    # shared 3D points, to the given image.
    def FindLocalBundle(self, options, image_id):
        image = self.reconstruction_.images[image_id]
        if not image.registered:
            print("Local bundle with unregistered image!")
            return False
        shared_observations = {}
        point3D_ids = []
        for point2D in image.get_valid_points2D():
            point3D_ids.append(point2D.point3D_id)
            point3D = self.reconstruction_.points3D[point2D.point3D_id]
            for track_el in point3D.track.elements:
                if track_el.image_id != image_id:
                    shared_observations[track_el.image_id] = shared_observations.get(track_el.image_id, 0) + 1

        shared_observations = sorted(shared_observations, reverse=True)
        overlapping_images = shared_observations
        # The local bundle is composed of the given image and its most connected
        # neighbor images, hence the subtraction of 1.
        num_images = options.local_ba_num_images - 1
        num_eff_images = min(num_images, len(overlapping_images))
        return overlapping_images[:num_eff_images]

    # Function called when image is being registered which properly handles variables within class
    def RegisterImageEvent(self, image_id):
        image = self.reconstruction_.images[image_id]
        num_reg_images_for_camera = self.num_reg_images_per_camera_.get(image.camera_id, 0)

        num_reg_images_for_camera += 1

        num_regs_for_image = self.num_registrations_.get(image_id, 0)
        num_regs_for_image += 1
        self.num_registrations_[image_id] += 1
        if num_regs_for_image == 1:
            self.num_total_reg_images_ += 1
        elif num_regs_for_image > 1:
            self.num_shared_reg_images_ += 1

    # Function called when image is being deregistered which properly handles variables within class
    def DeRegisterImageEvent(self, image_id):
        image = self.reconstruction_.images[image_id]
        num_reg_images_for_camera = self.num_reg_images_per_camera_.get(image.camera_id, 0)

        if num_reg_images_for_camera > 0:
            num_reg_images_for_camera -= 1

            num_regs_for_image = self.num_registrations_.get(image_id, 0)
            num_regs_for_image -= 1
            self.num_registrations_[image_id] -= 1
            if num_regs_for_image == 0:
                self.num_total_reg_images_ -= 1
            elif num_regs_for_image > 0:
                self.num_shared_reg_images_ -= 1

        self.images_manager_.deregister_image(image_id)


    def EstimateInitialTwoViewGeometry(self, options, image_id1, image_id2):
        image_pair_id = self.images_manager_.ImagePairToPairId(image_id1, image_id2)

        if self.prev_init_image_pair_id_ == image_pair_id:
            return True

        image1 = self.reconstruction_.images[image_id1]
        camera1 = self.reconstruction_.cameras[image1.camera_id]

        image2 = self.reconstruction_.images[image_id2]
        camera2 = self.reconstruction_.cameras[image2.camera_id]

        matches = self.graph_.find_correspondences_between_images(image_id1, image_id2)

        points1 = [image1.points2D[point_id[0]].xy for point_id in matches]

        points2 = [image2.points2D[point_id[1]].xy for point_id in matches]

        two_view_geometry_options = pycolmap.TwoViewGeometryOptions()
        two_view_geometry_options.ransac.min_num_trials = 30
        two_view_geometry_options.ransac.max_error = options.init_max_error

        answer = pycolmap.two_view_geometry_estimation(points1, points2, camera1, camera2, two_view_geometry_options)

        if not answer["success"]:
            return False

        flow_constr = self.OpticalFlowCalculator(points1, points2, answer["inliers"], camera1.focal_length_x,
                                                 camera1.focal_length_y)
        if flow_constr > options.optical_flow_threshold * 1.1 and answer["num_inliers"] >= options.init_min_num_inliers and abs(
                answer["tvec"][2]) < options.init_max_forward_motion:
            self.prev_init_image_pair_id_ = image_pair_id
            self.prev_init_two_view_geometry_ = answer
            return True

        return False

    # ========================= "public" ======================================

    # Create incremental mapper.
    def __init__(self):
        self.reconstruction_ = None
        self.num_total_reg_images_ = 0
        self.num_shared_reg_images_ = 0
        self.prev_init_image_pair_id_ = (-1, -1)

    # Prepare the mapper for a new reconstruction which is empty
    # (in which case `RegisterInitialImagePair` must be called).
    def BeginReconstruction(self, reconstruction, graph, images_manager):
        if self.reconstruction_ != None:
            logger.warning("Reconstruction object in Incremental Mapper should be empty!")
        self.reconstruction_ = reconstruction
        self.graph_ = graph
        self.images_manager_ = images_manager
        self.triangulator_ = pycolmap.IncrementalTriangulator(self.graph_, self.reconstruction_)

        self.num_shared_reg_images_ = 0
        self.num_reg_images_per_camera_ = {}

        for img in reconstruction.images:
            self.num_registrations_[img] = 0

        for img_id in reconstruction.reg_image_ids():
            self.RegisterImageEvent(img_id)

        self.existing_image_ids_ = reconstruction.reg_image_ids()

        self.prev_init_image_pair_id_ = (-1, -1)
        self.prev_init_two_view_geometry_ = None

        self.filtered_images_ = []
        self.num_reg_trials_ = {}

    # Cleanup the mapper after the current reconstruction is done. If the
    # model is discarded, the number of total and shared registered images will
    # be updated accordingly. Bool
    def EndReconstruction(self, discard):
        if self.reconstruction_ is None:
            logger.warning("Calling EndReconstruction on an empty reconstruction!")
        else:
            if discard:
                for img_ids in self.reconstruction_.reg_image_ids():
                    self.DeRegisterImageEvent(img_ids)

            self.reconstruction_ = None
            self.triangulator_ = None

    # Clears the done triangulation steps and all the associated 3D points
    # Should be called after failed initialization of the map
    def ClearReconstruction(self):
        if self.reconstruction_ is None:
            logger.warning("Calling EndReconstruction on an empty reconstruction!")
        else:
            for img_ids in self.reconstruction_.reg_image_ids():
                self.DeRegisterImageEvent(img_ids)


    # Find initial image pair to seed the incremental reconstruction. The image
    # pairs should be passed to `RegisterInitialImagePair`. This function
    # automatically ignores image pairs that failed to register previously.
    def FindInitialImagePair(self, options, image_id1, image_id2):
        options.check()

        image_ids1 = []
        if image_id1 != self.kInvalidImageId and image_id2 == self.kInvalidImageId:
            # Only first image provided
            if not self.images_manager_.exists_image(image_id1):
                return False, -1, -1

            image_ids1.append(image_id1)
        elif image_id1 == self.kInvalidImageId and image_id2 != self.kInvalidImageId:
            # Only second image provided
            if not self.images_manager_.exists_image(image_id2):
                return False, -1, -1

            image_ids1.push_back(image_id2)
        else:
            image_ids1 = self.FindFirstInitialImage(options)

        # Try to find good initial pair:
        for i1 in range(len(image_ids1)):
            image_id1 = image_ids1[i1]

            image_ids2 = self.FindSecondInitialImage(options, image_id1)

            for i2 in range(len(image_ids2)):
                image_id2 = image_ids2[i2]
                pair_id = self.images_manager_.ImagePairToPairId(image_id1, image_id2)

                if self.init_image_pairs_.count(pair_id) > 0:
                    continue

                self.init_image_pairs_.append(pair_id)

                if self.EstimateInitialTwoViewGeometry(options, image_id1, image_id2):
                    return True, image_id1, image_id2

        return False, -1, -1

    def OpticalFlowCalculator(self, point_list1, point_list2, inlier_mask, fx, fy):
        # Optical flow to make sure the baseline is big enough
        flow_constr = np.median(abs(np.array(point_list1) - np.array(point_list2))[inlier_mask])
        # Just using the one camera assuming we have only one
        f = np.array([fx, fy])
        flow_constr = sum(flow_constr / f)
        return flow_constr

    # Find best next image to register in the incremental reconstruction. The
    # images should be passed to `RegisterNextImage`. This function automatically
    # ignores images that failed to registered for `max_reg_trials`.
    def FindNextImages(self, options):
        num_correspondences = {}
        num_last_imgs = min(3, self.reconstruction_.num_reg_images())
        last_reg_images = self.reconstruction_.reg_image_ids()[-num_last_imgs:]
        if (last_reg_images[-1] == self.images_manager_.image_ids[-1]):
            return []
        valid_img_ids = self.images_manager_.image_ids[
                        last_reg_images[-1]:min(last_reg_images[-1] + 3, self.images_manager_.image_ids[-1] + 1)]
        for image_id in last_reg_images:

            # Matches the current image (image_id) with all images in valid_img_ids to fill the correspondence graph
            for other_img_id in valid_img_ids:
                self.images_manager_.add_to_correspondence_graph(image_id, other_img_id)

            image = self.reconstruction_.images[image_id]
            for point2D_idx in range(image.num_points2D()):
                point3D_id = image.points2D[point2D_idx].point3D_id
                if (point3D_id < 18446744073709551615):
                    for corr in self.graph_.find_correspondences(image_id, point2D_idx):
                        if corr.image_id in valid_img_ids and self.num_registrations_.get(corr.image_id, 0) == 0:
                            num_correspondences[corr.image_id] = num_correspondences.get(corr.image_id, 0) + 1

        init_min_num_inliers = options.init_min_num_inliers
        image_infos = []
        for k, v in num_correspondences.items():
            if v >= init_min_num_inliers:
                image = self.reconstruction_.images[k]
                camera = self.reconstruction_.cameras[image.camera_id]
                image_info = {
                    "image_id": k,
                    "prior_focal_length": camera.has_prior_focal_length,
                    "num_correspondences": v
                }
                image_infos.append(image_info)

        # Sort images such that images with a prior focal length and more
        # correspondences are preferred, i.e. they appear in the front of the list.
        image_infos = sorted(image_infos, key=lambda d: (not d["prior_focal_length"], d["num_correspondences"]),
                             reverse=True)

        # Extract image identifiers in sorted order.
        image_ids = [image_info["image_id"] for image_info in image_infos]

        return image_ids

    # Find next best image to register as keyframe and registers it into the reconstruction
    def FindAndRegisterNextKeyframe(self, options, per_frame_callback=None):

        last_keyframe_id = self.reconstruction_.reg_image_ids()[-1]
        last_keyframe = self.reconstruction_.images[last_keyframe_id]

        for current_img_id in self.images_manager_.image_ids[last_keyframe_id:]:

            matches = self.images_manager_.add_to_correspondence_graph(last_keyframe_id, current_img_id)
            if matches is None:
                continue

            current_img = self.reconstruction_.images[current_img_id]
            points_2D_current_img = []
            points_2D_last_keyframe = []
            points_3D = []
            for point2D_idx in range(last_keyframe.num_points2D()):
                point3D_id = last_keyframe.points2D[point2D_idx].point3D_id
                if (point3D_id < 18446744073709551615):
                    for corr in self.graph_.find_correspondences(last_keyframe_id, point2D_idx):
                        if corr.image_id == current_img_id and self.num_registrations_.get(current_img_id, 0) == 0:
                            points_3D.append(self.reconstruction_.points3D[point3D_id].xyz)
                            points_2D_current_img.append(current_img.points2D[corr.point2D_idx].xy)
                            points_2D_last_keyframe.append(last_keyframe.points2D[point2D_idx].xy)

            answer = pycolmap.absolute_pose_estimation(points_2D_current_img,
                                                       points_3D,
                                                       self.reconstruction_.cameras[0], max_error_px=2.0)  # 12.0
            if (answer['success']):
                current_img.tvec = answer['tvec']
                current_img.qvec = answer['qvec']

                camera = self.reconstruction_.cameras[current_img.camera_id]

                # Condition 1: Reject image if optical flow constraint is not satisfied
                flow_constr = self.OpticalFlowCalculator(points_2D_current_img, points_2D_last_keyframe, answer["inliers"], camera.focal_length_x,
                                                         camera.focal_length_y)


                self.reconstruction_.register_image(current_img_id)
                self.RegisterImageEvent(current_img_id)

                min_tri_angle_rad = self.DegToRad(options.init_min_tri_angle)
                triang_options = pycolmap.IncrementalTriangulatorOptions()
                triang_options.ignore_two_view_track = False
                self.triangulator_.triangulate_image(triang_options, current_img_id)
                # Filter3D points with large reprojection error, negative depth, or
                # insufficient triangulation angle
                self.reconstruction_.filter_all_points3D(options.init_max_error, min_tri_angle_rad)


                img1 = cv2.imread(
                    str(self.images_manager_.images_path / Path(self.images_manager_.frame_names[last_keyframe_id])))

                img2 = cv2.imread(
                    str(self.images_manager_.images_path / Path(self.images_manager_.frame_names[current_img_id])))
                matched_points = [(last_keyframe.points2D[q].xy, current_img.points2D[t].xy) for q,t in matches]
                for pt1, pt2 in matched_points:
                    u1, v1 = map(lambda x: int(round(x)), pt1)
                    u2, v2 = map(lambda x: int(round(x)), pt2)
                    cv2.circle(img1, (u1, v1), color=(0, 255, 0), radius=2)
                    cv2.circle(img2, (u2, v2), color=(0,255,0), radius=3)
                    cv2.line(img2, (u1,v1), (u2,v2), color=(255,0,0))

                font = cv2.FONT_HERSHEY_SIMPLEX
                green = (0, 255, 0)
                red = (255, 0, 0)
                blue = (0, 0, 255)
                thickness = 1
                lineType = cv2.LINE_AA

                cv2.putText(img1,
                            f'KEYFRAME id: {last_keyframe_id}',
                            (50, 50),
                            font,
                            1,
                            green,
                            thickness,
                            lineType)

                cv2.putText(img2,
                            f'id: {current_img_id}',
                            (50, 50),
                            font,
                            1,
                            blue,
                            thickness,
                            lineType)

                # Extra information about current frame
                cv2.putText(img2,
                            f'Optical flow: {flow_constr:.2f}',
                            (400, 20),
                            font,
                            (0.5),
                            green if flow_constr >= options.optical_flow_threshold else red,
                            thickness,
                            lineType)

                cv2.putText(img2,
                            f'Triangulated: {current_img.num_points3D():.2f}',
                            (400, 40),
                            font,
                            (0.5),
                            green if current_img.num_points3D() >= 50 else red,
                            thickness,
                            lineType)

                keyframe_decision = current_img.num_points3D() >= 50 and flow_constr >= options.optical_flow_threshold
                cv2.putText(img2,
                            f'Keyframe? {"yes" if keyframe_decision else "no"}',
                            (400, 60),
                            font,
                            (0.5),
                            green if keyframe_decision else red,
                            thickness,
                            lineType)
                horizontal_concat_images = np.concatenate((img1, img2), axis=1)

                # Trigger the callback with the new keyframe id
                if per_frame_callback:
                    per_frame_callback(current_img_id, horizontal_concat_images)

                if flow_constr < options.optical_flow_threshold:
                    continue

                # Condition 2: Reject registered image if it does not track sufficient points
                if current_img.num_points3D() < 50:
                    self.reconstruction_.deregister_image(current_img_id)
                    continue
                return current_img_id, True
        return None, False

    # Attempt to seed the reconstruction from an image pair.
    def RegisterInitialImagePair(self, options, image_id1, image_id2):
        if self.reconstruction_ is None:
            logger.warning("Incremental Mapper (RegisterInitialImagePair): reconstruction is NONE!")

        if self.reconstruction_.num_reg_images() != 0:
            logger.warning(
                "Incremental Mapper (RegisterInitialImagePair): reconstruction has already images registered!")

        options.check()

        self.init_num_reg_trials_[image_id1] = self.init_num_reg_trials_.get(image_id1, 0) + 1
        self.init_num_reg_trials_[image_id2] = self.init_num_reg_trials_.get(image_id2, 0) + 1
        self.num_reg_trials_[image_id1] = self.num_reg_trials_.get(image_id1, 0) + 1
        self.num_reg_trials_[image_id2] = self.num_reg_trials_.get(image_id2, 0) + 1

        pair_id = self.images_manager_.ImagePairToPairId(image_id1, image_id2)
        self.init_image_pairs_.append(pair_id)

        image1 = self.reconstruction_.images[image_id1]
        camera1 = self.reconstruction_.cameras[image1.camera_id]

        image2 = self.reconstruction_.images[image_id2]
        camera2 = self.reconstruction_.cameras[image2.camera_id]

 
        # Estimate two-view geometry
        if not self.EstimateInitialTwoViewGeometry(options, image_id1, image_id2):
            return False

        R = np.eye(3)
        qv = pycolmap.rotmat_to_qvec(R)
        tv = [0, 0, 0]

        image1.qvec = qv
        image1.tvec = tv
        image2.qvec = self.prev_init_two_view_geometry_["qvec"]
        image2.tvec = self.prev_init_two_view_geometry_["tvec"]

        proj_matrix1 = image1.projection_matrix()
        proj_matrix2 = image2.projection_matrix()
        proj_center1 = image1.projection_center()
        proj_center2 = image2.projection_center()

        # Update Reconstruction
        if (image_id1 > image_id2):
            image_id1, image_id2 = image_id2, image_id1
        self.reconstruction_.register_image(image_id1)
        self.reconstruction_.register_image(image_id2)
        self.RegisterImageEvent(image_id1)
        self.RegisterImageEvent(image_id2)

        min_tri_angle_rad = self.DegToRad(options.init_min_tri_angle)
        triang_options = pycolmap.IncrementalTriangulatorOptions()
        triang_options.ignore_two_view_track = False
        self.triangulator_.triangulate_image(triang_options, image_id1)
        # Filter3D points with large reprojection error, negative depth, or
        # insufficient triangulation angle
        self.reconstruction_.filter_all_points3D(options.init_max_error, min_tri_angle_rad)

        img1 = cv2.imread(str(self.images_manager_.images_path / Path(self.images_manager_.frame_names[image_id1])))
        img2 = cv2.imread(str(self.images_manager_.images_path / Path(self.images_manager_.frame_names[image_id2])))
        font = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (50, 50)
        fontScale = 1
        fontColor = (250, 50, 50)
        thickness = 1
        lineType = cv2.LINE_AA

        cv2.putText(img1, f'id: {image_id1} (Map Initialization)', bottomLeftCornerOfText, font, fontScale, fontColor, thickness, lineType)
        cv2.putText(img2, f'id: {image_id2} (Map Initialization)', bottomLeftCornerOfText, font, fontScale, fontColor, thickness, lineType)
        horizontal = np.concatenate((img1, img2), axis=1)
        return True, horizontal

    # Attempt to register image to the existing model. This requires that
    # a previous call to `RegisterInitialImagePair` was successful.
    def RegisterNextImage(self, options, image_id):
        query_img_id = image_id
        query_img = self.reconstruction_.images[query_img_id]
        points_2D = []
        points_3D = []
        num_last_imgs = min(3, self.reconstruction_.num_reg_images())
        last_reg_images = self.reconstruction_.reg_image_ids()[-num_last_imgs:]
        for image_id in last_reg_images:
            image = self.reconstruction_.images[image_id]
            for point2D_idx in range(image.num_points2D()):
                point3D_id = image.points2D[point2D_idx].point3D_id
                if (point3D_id < 18446744073709551615):
                    for corr in self.graph_.find_correspondences(image_id, point2D_idx):
                        if corr.image_id == query_img_id and self.num_registrations_.get(query_img_id, 0) == 0:
                            points_3D.append(self.reconstruction_.points3D[point3D_id].xyz)
                            points_2D.append(query_img.points2D[corr.point2D_idx].xy)

        answer = pycolmap.absolute_pose_estimation(points_2D,
                                                   points_3D,
                                                   self.reconstruction_.cameras[0], max_error_px=2.0)  # 12.0
        if (answer['success']):
            query_img.tvec = answer['tvec']
            query_img.qvec = answer['qvec']
            self.reconstruction_.register_image(query_img_id)
            self.RegisterImageEvent(query_img_id)

            min_tri_angle_rad = self.DegToRad(options.init_min_tri_angle)
            triang_options = pycolmap.IncrementalTriangulatorOptions()
            self.triangulator_.triangulate_image(triang_options, query_img_id)
            # Filter3D points with large reprojection error, negative depth, or
            # insufficient triangulation angle
            self.reconstruction_.filter_all_points3D(options.init_max_error, min_tri_angle_rad)

        return answer['success']

    # Adjust locally connected images and points of a reference image. In
    # addition, refine the provided 3D points. Only images connected to the
    # reference image are optimized. If the provided 3D points are not locally
    # connected to the reference image, their observing images are set as
    # constant in the adjustment.
    # options: IncrementalMapper Options
    # ba_options: Bundle adjustment options
    # tri_options: Triangulator options
    def AdjustLocalBundle(self, options, ba_options, tri_options, image_id, point3D_ids):
        # The number of images to optimize in local bundle adjustment.
        ba_local_num_images = 6

        if tri_options is None:
            tri_options = pycolmap.IncrementalTriangulatorOptions()

        local_bundle = self.FindLocalBundle(options, image_id)
        local_bundle = local_bundle[:ba_local_num_images]
        local_bundle.append(image_id)

        variable_points = []
        if len(local_bundle) > 0:
            ba = BundleAdjuster(self.reconstruction_)
            success, variable_points = ba.local_BA(local_bundle)
            num_merged_observations = self.triangulator_.merge_all_tracks(tri_options)
            logger.info("After Local BA ===============\n Merged " + str(num_merged_observations) + " tracks")
            num_completed_observations = self.triangulator_.complete_all_tracks(tri_options)
            num_completed_observations += self.triangulator_.complete_image(tri_options, image_id)
            logger.info("Completed " + str(num_completed_observations) + " observations")

        num_filtered_observations = self.reconstruction_.filter_points3D_in_images(options.filter_max_reproj_error,
                                                                                   options.filter_min_tri_angle,
                                                                                   set(local_bundle))
        num_filtered_observations += self.reconstruction_.filter_points3D(options.filter_max_reproj_error,
                                                                          options.filter_min_tri_angle, set(variable_points))
        logger.info("Filtered " + str(num_filtered_observations) + " observations")

    # Global bundle adjustment using Ceres Solver or PBA.
    def AdjustGlobalBundle(self, options, ba_options=None):
        reg_image_ids = self.reconstruction_.num_reg_images()
        if ba_options is None:
            ba_options = pyceres.SolverOptions()
        if reg_image_ids < 2:
            logger.warning("Not enough registered images for global BA")
            return False

        ba = BundleAdjuster(self.reconstruction_, options=ba_options)
        if not ba.global_BA():
            return False
        self.reconstruction_.normalize(10.0, 0.1, 0.9, True)
        tri_options = pycolmap.IncrementalTriangulatorOptions()
        num_merged_observations = self.triangulator_.merge_all_tracks(tri_options)
        num_completed_observations = self.triangulator_.complete_all_tracks(tri_options)
        num_filtered_observations = self.reconstruction_.filter_all_points3D(options.filter_max_reproj_error,
                                                                          options.filter_min_tri_angle)
        logger.info("After Global BA ===============\n Merged " + str(num_merged_observations) + " tracks" +
                    "\nCompleted " + str(num_completed_observations) + " observations" +
                    "\nFiltered " + str(num_filtered_observations) + " observations")
        return True


    # Filter images and point observations.
    # Mainly based on the focal length
    def FilterImages(self, options):
        kMinNumImages = 20
        if self.reconstruction_.num_reg_images() < kMinNumImages:
            return []
        image_ids = self.reconstruction_.filter_images(options.min_focal_length_ratio, options.max_focal_length_ratio,
                                                       options.max_extra_param)
        filtered_images = []
        for img_id in image_ids:
            self.DeRegisterImageEvent(img_id)
            filtered_images.append(img_id)
        return len(image_ids)

    def FilterPoints(self, options):
        return self.reconstruction_.filter_all_points3D(options.filter_max_reproj_error, options.filter_min_tri_angle)

