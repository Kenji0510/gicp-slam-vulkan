use anyhow::{Context, Result};
use gicp_slam_vulkan::{
    convert_type::{
        convert_pcd_to_xyz, convert_point3_to_vec, convert_vec_point_cov_to_pcd_xyzcov,
        convert_vec_to_xyz, convert_xyz_to_vec,
    },
    deskew_points::deskew_points,
    file_handler::{
        load_imu_data, load_pcd_files, load_pcd_xyzit, save_pcd_xyzcov, save_pcd_xyzit,
        save_pcd_xyznormal,
    },
    gpu_covariances::{self, combine_pts_with_normals},
    gpu_gicp::{GicpGpuContext, GicpStaticBuffers},
    gpu_knn_search,
    gpu_search_neighbor::SearchGpuContext,
    gpu_transfer_data::GpuTransferDataContext,
    gpu_transform,
    gpu_voxel::VoxelGpuContext,
    init_gpu::VulkanContext,
    predict_pose_by_imu::{align_imu_timestamps, build_rotation_trajectory, predict_pose_by_imu},
};
use nalgebra::{Matrix4, Quaternion, UnitQuaternion, Vector3};

const LOAD_DIR: &str = "data/input/05172026/park01";
const SAVE_DIR: &str = "data/output/05182026/park01";

const DOWNSAMPLE_VOXEL_SIZE: f32 = 0.2; // m
const GICP_ITERATIONS: usize = 5;

const MIN_DIST: f32 = 0.1;
const MAX_DIST: f32 = 45.0;

const MAX_POINTS_PER_VOXEL: usize = 10;
const MIN_POINTS_PER_VOXEL: usize = 3;

const SEARCH_RANGE: i32 = 3; // Range of 5x5x5 voxels
const MAX_DIST_SQ: f32 = 1.0; // Optional maximum distance squared

// IMU coordination to LiDAR coordination (Robosense 96 beam)
// Quaternion (x, y, z, w): -0.705437, 0.708767, -0.00246579, 0.00097028
// Translation (x, y, z)  : 0.00425, 0.00418, -0.00446  [m]
const IMU_TO_LIDAR_QUAT_X: f64 = -0.705437;
const IMU_TO_LIDAR_QUAT_Y: f64 = 0.708767;
const IMU_TO_LIDAR_QUAT_Z: f64 = -0.00246579;
const IMU_TO_LIDAR_QUAT_W: f64 = 0.00097028;

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("debug")).init();

    // --- Initialize Vulkan context ---
    let vulkan_context = VulkanContext::new().context("Failed to initialize Vulkan context")?;
    let mut copy_gpu_context = GpuTransferDataContext::new(vulkan_context.clone())
        .context("Failed to create GPU transfer context")?;
    let mut voxel_gpu_context = VoxelGpuContext::new(vulkan_context.clone())?;
    let mut knn_gpu_context = gpu_knn_search::KnnSearchGpuContext::new(vulkan_context.clone())?;
    let mut covariances_gpu_context =
        gpu_covariances::CovarianceGpuContext::new(vulkan_context.clone())?;
    let mut transform_gpu_context =
        gpu_transform::TransformGpuContext::new(vulkan_context.clone())?;
    let mut search_neighbor_gpu_context = SearchGpuContext::new(vulkan_context.clone())?;
    let mut gicp_gpu_context = GicpGpuContext::new(vulkan_context.clone())?;
    // --- Initialize Vulkan context ---

    let pcd_dir = format!("{}/pcd", LOAD_DIR);
    let pcd_files = load_pcd_files(&pcd_dir)?;

    log::debug!(
        "Found {} PCD files in directory: {}",
        pcd_files.len(),
        pcd_dir
    );

    let imu_dir = format!("{}/imu", LOAD_DIR);
    let imu_file = format!("{}/imu_data.json", imu_dir);
    let imu_data = load_imu_data(&imu_file)?;
    let imu_data = align_imu_timestamps(&imu_data); // Align IMU timestamps to seconds

    // IMU coord to LiDAR coord transformation
    let imu_to_lidar = UnitQuaternion::new_normalize(Quaternion::new(
        IMU_TO_LIDAR_QUAT_W,
        IMU_TO_LIDAR_QUAT_X,
        IMU_TO_LIDAR_QUAT_Y,
        IMU_TO_LIDAR_QUAT_Z,
    ));

    let mut current_global_pose = Matrix4::<f64>::identity();
    let mut current_velocity = Vector3::<f64>::zeros();

    let pcd = load_pcd_xyzit(&pcd_files[0].to_string_lossy())?;
    // let points = convert_pcd_to_xyz(&pcd);

    let points_vec = convert_xyz_to_vec(&pcd);

    // --- Copy points to gpu memory ---
    copy_gpu_context.copy_data_to_gpu(&points_vec, points_vec.len())?;
    // --- Copy points to gpu memory ---

    // --- Downsample for density normalization ---
    let downsample_voxel_size = DOWNSAMPLE_VOXEL_SIZE;
    let points_num = points_vec.len();

    let downsampled_points_vec =
        voxel_gpu_context.voxelization(&copy_gpu_context, downsample_voxel_size)?;
    // let downsampled_init_points = voxel_downsample_points(&points, downsample_voxel_size);

    // let downsampled_pcd = convert_vec_to_xyz(&downsampled_points_vec);
    // let save_file = format!("{}/debug/downsampled_init.pcd", SAVE_DIR);
    // save_pcd_xyzit(&downsampled_pcd, &save_file)?;
    // --- Downsample for density normalization ---

    let mut prev_frame_start_time = pcd
        .iter()
        .map(|p| p.timestamp)
        .fold(f64::INFINITY, f64::min);

    for (i, pcd_path) in pcd_files.iter().enumerate().skip(1) {
        log::info!("Processing frame {}: {}", i, pcd_path.to_string_lossy());

        let source_pcd = load_pcd_xyzit(&pcd_path.to_string_lossy())?;

        // --- Predict pose by IMU ---
        let current_frame_start_time = source_pcd
            .iter()
            .map(|p| p.timestamp)
            .fold(f64::INFINITY, f64::min);
        let current_frame_end_time = source_pcd
            .iter()
            .map(|p| p.timestamp)
            .fold(f64::NEG_INFINITY, f64::max);

        // 既にGlobal座標でのcurrent_global_poseがある状態で、次フレームの開始時刻までのIMU積分を行う。
        let pose_prediction = predict_pose_by_imu(
            &imu_data,
            &imu_to_lidar,
            &current_global_pose,
            &current_velocity,
            prev_frame_start_time,
            current_frame_start_time,
        );

        let rotation_traj = build_rotation_trajectory(
            &imu_data,
            current_frame_start_time,
            current_frame_end_time,
            &imu_to_lidar,
        );

        // --- Deskew source pcd ---
        let deskewed_points = deskew_points(
            &source_pcd,
            &rotation_traj,
            &imu_to_lidar,
            current_frame_start_time,
            MIN_DIST,
            MAX_DIST,
        );
        // --- Deskew source pcd ---

        let points_vec = convert_point3_to_vec(&deskewed_points);

        // --- Copy points to gpu memory ---
        copy_gpu_context.copy_data_to_gpu(&points_vec, points_vec.len())?;
        // --- Copy points to gpu memory ---

        // --- Downsample for density normalization ---
        let downsampled_points_vec =
            voxel_gpu_context.voxelization(&copy_gpu_context, downsample_voxel_size)?;
        // --- Downsample for density normalization ---

        let mut current_transform = pose_prediction.0;

        // --- Compute covariance for each point ---
        knn_gpu_context.knn_search_neighbors(&voxel_gpu_context)?;

        let covariances =
            covariances_gpu_context.compute_covariances(&voxel_gpu_context, &knn_gpu_context)?;

        // --- Debug ---
        // let pcd_xyznormals = combine_pts_with_normals(&downsampled_points_vec, &normals)?;
        // let pcd_xyzcov =
        //     convert_vec_point_cov_to_pcd_xyzcov(&downsampled_points_vec, &covariances);
        // let save_path = format!("{}/debug/pcd_with_covariances_{:04}.pcd", SAVE_DIR, i);

        // save_pcd_xyzcov(&pcd_xyzcov, &save_path)?;
        // --- Debug ---
        // --- Compute covariance for each point ---

        // --- Transform the points ---
        let m = current_transform;
        let transform_params = gpu_transform::TransformParams {
            r00: m[(0, 0)] as f32,
            r01: m[(0, 1)] as f32,
            r02: m[(0, 2)] as f32,
            t0: m[(0, 3)] as f32,
            r10: m[(1, 0)] as f32,
            r11: m[(1, 1)] as f32,
            r12: m[(1, 2)] as f32,
            t1: m[(1, 3)] as f32,
            r20: m[(2, 0)] as f32,
            r21: m[(2, 1)] as f32,
            r22: m[(2, 2)] as f32,
            t2: m[(2, 3)] as f32,
            num_points: voxel_gpu_context.h_downsampled_pts_num as u32,
        };
        transform_gpu_context.transform(
            &voxel_gpu_context,
            &covariances_gpu_context,
            transform_params,
        )?;
        // --- Transform the points ---

        // --- Search neighbor points for each point ---
        search_neighbor_gpu_context.search_neighbor(
            &transform_gpu_context,
            voxel_gpu_context.h_downsampled_pts_num,
            &voxel_gpu_context,
            voxel_gpu_context.h_downsampled_pts_num,
        )?;
        // --- Search neighbor points for each point ---

        // --- GICP optimization ---
        let gicp_bufs = GicpStaticBuffers {
            d_source_pts: transform_gpu_context
                .d_buf_output_pts
                .as_ref()
                .context("Failed to get transformed source points buffer")?
                .clone(),
            d_source_covs: covariances_gpu_context
                .d_buf_covariances
                .as_ref()
                .context("Failed to get source covariances buffer")?
                .clone(),
            d_target_pts: voxel_gpu_context
                .d_buf_out_pts
                .as_ref()
                .context("Failed to get target points buffer")?
                .clone(),
            d_target_covs: covariances_gpu_context
                .d_buf_covariances
                .as_ref()
                .context("Failed to get target covariances buffer")?
                .clone(),
        };

        gicp_gpu_context.compute_gicp(
            &gicp_bufs,
            &search_neighbor_gpu_context,
            voxel_gpu_context.h_downsampled_pts_num,
            voxel_gpu_context.h_downsampled_pts_num,
            MAX_DIST_SQ,
        )?;
        // --- GICP optimization ---
    }

    Ok(())
}
