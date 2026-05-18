use anyhow::{Context, Result};
use gicp_slam_vulkan::{
    convert_type::{convert_pcd_to_xyz, convert_vec_to_xyz, convert_xyz_to_vec},
    file_handler::{load_imu_data, load_pcd_files, load_pcd_xyzit, save_pcd_xyzit},
    gpu_transfer_data::GpuTransferDataContext,
    gpu_voxel::VoxelGpuContext,
    init_gpu::VulkanContext,
    predict_pose_by_imu::align_imu_timestamps,
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
    let mut voxel_gpu_context = VoxelGpuContext::new(vulkan_context.clone());
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
        voxel_gpu_context?.voxelization(&copy_gpu_context, downsample_voxel_size)?;
    // let downsampled_init_points = voxel_downsample_points(&points, downsample_voxel_size);

    // let downsampled_pcd = convert_vec_to_xyz(&downsampled_points_vec);
    // let save_file = format!("{}/debug/downsampled_init.pcd", SAVE_DIR);
    // save_pcd_xyzit(&downsampled_pcd, &save_file)?;
    // --- Downsample for density normalization ---

    Ok(())
}
