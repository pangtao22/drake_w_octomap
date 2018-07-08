#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <drake/math/roll_pitch_yaw.h>
#include <drake/multibody/joints/floating_base_types.h>
#include <drake/multibody/parsers/urdf_parser.h>
#include <drake/multibody/rigid_body_tree.h>
#include <drake/systems/framework/basic_vector.h>
#include <drake/systems/framework/value.h>
#include <drake/systems/sensors/rgbd_camera.h>

#include <octomap/octomap.h>

using drake::systems::sensors::ImageDepth32F;
using Eigen::VectorXd;
using std::cout;
using std::endl;

const double kMaxRange = 100;

int main() {
  char const *urdf_path = "/Users/pangtao/drake_w_octomap/house.urdf";
  std::string camera_pose_file_name =
      "/Users/pangtao/drake_w_octomap/camera_poses.txt";
  RigidBodyTreed tree;
  drake::parsers::urdf::AddModelInstanceFromUrdfFileToWorld(
      urdf_path, drake::multibody::joints::kFixed, &tree);

  auto camera_frame = std::make_shared<RigidBodyFrame<double>>(
      "camera_frame", tree.get_mutable_body(2));
  tree.addFrame(camera_frame);
  drake::systems::sensors::RgbdCamera camera("camera", tree, *camera_frame, 0.5,
                                             kMaxRange);
  auto context = camera.CreateDefaultContext();
  std::unique_ptr<drake::systems::SystemOutput<double>> output =
      camera.AllocateOutput();

  std::ifstream pose_file;
  pose_file.open(camera_pose_file_name);
  int i = 0;
  double x, y, z, qx, qy, qz, qw;
  VectorXd x_camera(tree.get_num_positions() + tree.get_num_velocities());
  x_camera.setZero();
  std::vector<VectorXd> camera_poses;

  while (!pose_file.eof()) {
    pose_file >> x;
    pose_file >> y;
    pose_file >> z;
    pose_file >> qw;
    pose_file >> qx;
    pose_file >> qy;
    pose_file >> qz;
    i++;

    if (i % 20 == 0) {
      Eigen::Quaterniond q(qw, qx, qy, qz);
      drake::math::RollPitchYaw<double> rpy(q);
      x_camera.head(6) << x, y, z, rpy.roll_angle(), rpy.pitch_angle(),
          rpy.yaw_angle();
      camera_poses.push_back(x_camera);
    }
  }
  pose_file.close();
  cout << "poses loaded from file: " << camera_poses.size() << endl;
  cout << "tree size " << tree.get_num_bodies() << endl;

  const octomap::point3d camera_origin(0, 0, 0);

  Eigen::Matrix3Xf point_cloud_eigen;
  octomap::OcTree oct_tree(0.5);
  octomap::Pointcloud point_cloud_octomap;
  KinematicsCache<double> cache = tree.CreateKinematicsCache();

  for (int i = 0; i < camera_poses.size(); i++) {
    cout << "Working on frame " << i << endl;
    cout << "Camera pose:\n" << camera_poses[i] << endl;

    context->FixInputPort(0, camera_poses[i]);
    camera.CalcOutput(*context, output.get());
    auto depth_image_ptr =
        output->get_data(camera.depth_image_output_port().get_index());
    auto depth_image = depth_image_ptr->GetValue<ImageDepth32F>();
    drake::systems::sensors::RgbdCamera::ConvertDepthImageToPointCloud(
        depth_image, camera.depth_camera_info(), &point_cloud_eigen);

    // camera pose
    Eigen::Isometry3d T_BD = camera.depth_camera_optical_pose();
    cache.initialize(camera_poses[i].head(6), camera_poses[i].tail(6));
    tree.doKinematics(cache);
    Eigen::Isometry3d T_WB =
        tree.CalcBodyPoseInWorldFrame(cache, tree.get_body(2));
    Eigen::Isometry3d T_WD = T_WB * T_BD;

    for (int j = 0; j < point_cloud_eigen.cols(); j++) {
      Eigen::Vector3d point_d;
      point_d[0] = point_cloud_eigen(0, j);
      point_d[1] = point_cloud_eigen(1, j);
      point_d[2] = point_cloud_eigen(2, j);
      if (point_d.norm() > 0.2 &&
          point_d.norm() < std::numeric_limits<double>::infinity()) {
        Eigen::Vector3d point_w; // point in world frame.
        point_w = T_WD.translation() + T_WD.linear() * point_d;
        octomap::point3d p((float)point_w[0], (float)point_w[1],
                           (float)point_w[2]);
        point_cloud_octomap.push_back(p);
      }
    }
    oct_tree.insertPointCloud(point_cloud_octomap, camera_origin, kMaxRange);
  }

  std::ofstream data_file;
  data_file.open("data.ot");
  oct_tree.write(data_file);
  data_file.close();

  return 0;
}