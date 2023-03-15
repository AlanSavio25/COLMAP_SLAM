import pyceres

# Adapted from: https://github.com/cvg/pyceres/blob/main/examples/test_BA.ipynb

import pycolmap


class BundleAdjuster:
    # rec is the reconstruction object
    def __init__(self, rec, options=None, debug=True):
        self.rec = rec
        self.img_list = []
        self.prob = pyceres.Problem()
        self.debug = debug
        self.point3D_num_observations = {}
        self.constant_pose = []
        self.constant_tvec = []
        self.constant_points = []
        if options is None:
            self.options = pyceres.SolverOptions()
        else:
            self.options = options
        # self.prob = None

    # Full BA optimizing 3D points and poses
    # Should be used at map initialization
    # skip_pose indicates a list of image.id where the pose should not be optimized (i.e. origin)
    def global_BA(self):
        self.prob = pyceres.Problem()

        # Loss function types: Trivial(non - robust) and Cauchy(robust) loss.
        loss = pyceres.TrivialLoss()

        #Options from https://github.com/colmap/colmap/blob/dev/src/controllers/incremental_mapper.cc/#L219-L226
        self.options.gradient_tolerance = 10.0
        self.options.parameter_tolerance = 0.0
        self.options.max_linear_solver_iterations = 100

        # The maximum number of local bundle adjustment iterations.
        self.options.max_num_iterations = 50

        self.rec.filter_observations_with_negative_depth()

        if not self.img_list:
            self.img_list = [im for im in self.rec.images.values() if im.registered]
            img_list_id = [im.image_id for im in self.img_list]
            img_list_id.sort()

        if not self.constant_pose:
            # Do not optimize pose of first image
            self.constant_pose = [img_list_id[0]]
        if not self.constant_tvec:
            # Do not optimize x-translation of second image
            self.constant_tvec = [img_list_id[1]]

        varible_points = []
        for im in self.img_list:
            self.add_img_to_problem(im.image_id, loss)
            # for p in im.get_valid_points2D():
               # varible_points.append(p.point3D_id)

        for p in varible_points:
            self.add_point_to_problem(p, loss)

        self.parameterize_camera()
        self.parameterize_points()

        self.solve()

        return True

    # Only a part of the images and map points get optimized
    # That is specified by img_list
    def local_BA(self, img_list_id):
        self.prob = pyceres.Problem()

        # Loss function types: Trivial(non - robust) and Cauchy(robust) loss.
        loss = pyceres.SoftLOneLoss(1.0)

        #Options from https://github.com/colmap/colmap/blob/dev/src/controllers/incremental_mapper.cc/#L219-L226
        self.options.gradient_tolerance = 10.0
        self.options.parameter_tolerance = 0.0
        self.options.max_linear_solver_iterations = 100

        # The maximum number of local bundle adjustment iterations.
        self.options.max_num_iterations = 25

        #loss = pyceres.TrivialLoss()
        self.rec.filter_observations_with_negative_depth()

        self.img_list = [element for element in self.rec.images.values() if element.image_id in img_list_id and element.registered]

        if not self.constant_pose:
            # Do not optimize pose of first image
            self.constant_pose = [img_list_id[0]]
            if len(img_list_id) == 2:
                self.constant_tvec = [img_list_id[0]]
        if not self.constant_tvec and len(img_list_id) > 2:
            # Do not optimize x-translation of second image
            self.constant_tvec = [img_list_id[1]]

        variable_points = []
        for im in self.img_list:
            self.add_img_to_problem(im.image_id, loss)
            for p in im.get_valid_points2D():
                point3d = self.rec.points3D[p.point3D_id]
                if point3d.error > 0.001 and point3d.track.length() <= 15:
                    variable_points.append(p.point3D_id)

        for p in variable_points:
            self.add_point_to_problem(p, loss)

        self.parameterize_camera()
        self.parameterize_points()

        self.solve()

        return True, variable_points

    # Motion only BA: all points stay fixed and only image poses get optimized
    # rec is the reconstruction object
    # img_list stores the ids of the images where motion only BA should be applied to
    def motion_only_BA(self, img_list):
        self.prob = pyceres.Problem()

        # Loss function types: Trivial(non - robust) and Cauchy(robust) loss.
        loss = pyceres.TrivialLoss()

        self.rec.filter_observations_with_negative_depth()

        self.img_list = [element for element in self.rec.images.values() if
                         element.image_id in img_list and element.registered]
        # self.constant_pose = [self.rec.images.values()[0]]
        # self.constant_tvec = [self.rec.images.values()[1]]
        self.define_problem_pose()
        self.solve()

    def add_img_to_problem(self, image_id, loss):
        im = self.rec.images[image_id]
        cam = self.rec.cameras[im.camera_id]
        im.qvec = pycolmap.normalize_qvec(im.qvec)
        constant_pose = True if im.image_id in self.constant_pose else False
        is_constant_tvec = True if im.image_id in self.constant_tvec else False

        for p in im.get_valid_points2D():
            self.point3D_num_observations[p.point3D_id] = self.point3D_num_observations.get(p.point3D_id, 0) + 1
            assert self.rec.points3D[p.point3D_id].track.length() > 1
            if constant_pose:
                cost = pyceres.factors.BundleAdjustmentCost(cam.model_id, p.xy, im.qvec, im.tvec)
                self.prob.add_residual_block(cost, loss, [self.rec.points3D[p.point3D_id].xyz, cam.params])
            else:
                cost = pyceres.factors.BundleAdjustmentCost(cam.model_id, p.xy)
                self.prob.add_residual_block(cost, loss,
                                        [im.qvec, im.tvec, self.rec.points3D[p.point3D_id].xyz, cam.params])
        # If the pose should be optimized
        if len(im.get_valid_points2D()) > 0 and not constant_pose:
            self.prob.set_parameterization(im.qvec, pyceres.QuaternionParameterization())
            if is_constant_tvec:
                self.prob.set_parameterization(im.tvec, pyceres.SubsetParameterization(3, [0]))

    def add_point_to_problem(self, point3D_id, loss):
        # pyceres.Problem().
        point3D = self.rec.points3D[point3D_id]
        if self.point3D_num_observations.get(point3D_id, 0) != point3D.track.length():
            for track_el in point3D.track.elements:
                if track_el.image_id not in self.img_list:
                    self.point3D_num_observations[point3D_id] = self.point3D_num_observations.get(point3D_id, 0) + 1

                    im = self.rec.images[track_el.image_id]
                    cam = self.rec.cameras[im.camera_id]
                    point2D = im.points2D[track_el.point2D_idx]

                    cost = pyceres.factors.BundleAdjustmentCost(cam.model_id, point2D.xy, im.qvec, im.tvec)
                    self.prob.add_residual_block(cost, loss, [self.rec.points3D[point2D.point3D_id].xyz, cam.params])

    # Assuming only oe camera
    def parameterize_camera(self):
        cam = self.rec.cameras[0]
        self.prob.set_parameter_block_constant(cam.params)
        # TODO add camera refinemnets: https://github.com/colmap/colmap/blob/e180948665b03c4a12d45e2ca39a589f42fdbda6/src/optim/bundle_adjustment.cc#L484-L522

    def parameterize_points(self):
        for key in self.point3D_num_observations:
            point3D = self.rec.points3D[key]
            a = point3D.track.length()
            b = self.point3D_num_observations[key]
            c = a > b
            if point3D.track.length() > self.point3D_num_observations[key]:
                self.prob.set_parameter_block_constant(point3D.xyz)

        for point3D_id in self.constant_points:
            self.prob.set_parameter_block_constant(self.rec.points3D[point3D_id].xyz)

    # Optmizes all the 3D points
    def define_problem_points(self):
        prob = pyceres.Problem()
        loss = pyceres.TrivialLoss()
        for im in self.img_list:
            cam = self.rec.cameras[im.camera_id]
            im.qvec = pycolmap.normalize_qvec(im.qvec)
            for p in im.get_valid_points2D():
                cost = pyceres.factors.BundleAdjustmentCost(cam.model_id, p.xy, im.qvec, im.tvec)
                prob.add_residual_block(cost, loss, [self.rec.points3D[p.point3D_id].xyz, cam.params])
        for cam in self.rec.cameras.values():
            prob.set_parameter_block_constant(cam.params)
        return prob

    def define_problem_pose(self):
        self.prob = pyceres.Problem()
        loss = pyceres.TrivialLoss()
        points3D = []
        for im in self.img_list:
            cam = self.rec.cameras[im.camera_id]
            im.qvec = pycolmap.normalize_qvec(im.qvec)
            constant_pose = True if im.image_id in self.constant_pose else False
            is_constant_tvec = True if im.image_id in self.constant_tvec else False
            for p in im.get_valid_points2D():
                if constant_pose:
                    cost = pyceres.factors.BundleAdjustmentCost(cam.model_id, p.xy, im.qvec, im.tvec)
                    self.prob.add_residual_block(cost, loss, [self.rec.points3D[p.point3D_id].xyz, cam.params])
                else:
                    cost = pyceres.factors.BundleAdjustmentCost(cam.model_id, p.xy)
                    self.prob.add_residual_block(cost, loss,
                                            [im.qvec, im.tvec, self.rec.points3D[p.point3D_id].xyz, cam.params])
                points3D.append(self.rec.points3D[p.point3D_id])
            # If the pose should be optimized
            if len(im.get_valid_points2D()) > 0 and not constant_pose:
                self.prob.set_parameterization(im.qvec, pyceres.QuaternionParameterization())
                if is_constant_tvec:
                    self.prob.set_parameterization(im.tvec, pyceres.SubsetParameterization(3, [0]))
        for cam in self.rec.cameras.values():
            self.prob.set_parameter_block_constant(cam.params)
        # for p in rec.points3D.values():
        for p in points3D:
            self.prob.set_parameter_block_constant(p.xyz)

    def solve(self):
        if self.debug:
            print(self.prob.num_parameter_bocks(), self.prob.num_parameters(), self.prob.num_residual_blocks(), self.prob.num_residuals())

        # See: https://github.com/colmap/colmap/blob/e180948665b03c4a12d45e2ca39a589f42fdbda6/src/optim/bundle_adjustment.cc#L276-L286
        if len(self.img_list) <= 50:
            self.options.linear_solver_type = pyceres.LinearSolverType.DENSE_SCHUR
            # self.options.linear_solver_type = pyceres.LinearSolverType.DENSE_QR
        else:
            self.options.linear_solver_type = pyceres.LinearSolverType.ITERATIVE_SCHUR
            self.options.preconditioner_type = pyceres.PreconditionerType.SCHUR_JACOBI

        self.options.minimizer_progress_to_stdout = self.debug
        self.options.num_threads = -1
        summary = pyceres.SolverSummary()
        pyceres.solve(self.options, self.prob, summary)
        if self.debug:
            print(summary.BriefReport())
