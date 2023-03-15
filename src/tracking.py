"""
B. Initial Pose Estimation from Previous Frame

If tracking was successful for last frame, we use a constant
velocity motion model to predict the camera pose and perform
a guided search of the map points observed in the last frame. If
not enough matches were found (i.e. motion model is clearly
violated), we use a wider search of the map points around
their position in the last frame. The pose is then optimized
with the found correspondences.
From: ORB-SLAM: a Versatile and Accurate Monocular SLAM System
"""
def initial_pose_estimate_previous_frame(motion_model):
    ret = {"success": False, "tvec": [0, 0, 0], "qvec": [0, 0, 0, 0]}
    # TODO
    return ret


"""
C. Initial Pose Estimation via Global Relocalization

If the tracking is lost, i.e. initial_pose_estimate_previous_frame
returns success = False, we convert the frame into bag
of words and query the recognition database for keyframe
candidates for global relocalization. We compute correspond-
ences with ORB associated to map points in each keyframe,
as explained in section III-E. We then perform alternatively
RANSAC iterations for each keyframe and try to find a camera
pose using the PnP algorithm [41]. If we find a camera
pose with enough inliers, we optimize the pose and perform
a guided search of more matches with the map points of
the candidate keyframe. Finally the camera pose is again
optimized, and if supported with enough inliers, tracking
procedure continues.
From: ORB-SLAM: a Versatile and Accurate Monocular SLAM System
"""
def initial_pose_estiamte_global():
    ret = {"success": False, "tvec": [0, 0, 0], "qvec": [0, 0, 0, 0]}
    # TODO
    return ret