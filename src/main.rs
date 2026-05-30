use std::time::Instant;

use anyhow::{Context, Result};
use gicp_slam_vulkan::{
    convert_type::{
        convert_pcd_to_xyz, convert_point3_to_vec, convert_vec_point_cov_to_pcd_xyzcov,
        convert_vec_to_point3, convert_vec_to_xyz, convert_xyz_to_vec,
    }, deskew_points::deskew_points, file_handler::{
        load_imu_data, load_pcd_files, load_pcd_xyzit, save_pcd_xyzcov, save_pcd_xyzit,
        save_pcd_xyznormal,
    }, gpu_copy::GpuTransferDataContext, gpu_covariances::{self, combine_pts_with_normals}, gpu_gicp::{GicpGpuContext, GicpStaticBuffers, solve_gicp}, gpu_knn_search, gpu_search_neighbor::SearchGpuContext, gpu_transform, gpu_voxel::VoxelGpuContext, init_gpu::VulkanContext, log_performance::PerformanceLogs, loop_closure::{LoopAlignmentConfig, LoopCandidateConfig, LoopCandidateFinder, align_loop_candidate}, pose_graph::{PoseGraph, default_loop_information, default_odometry_information}, predict_pose_by_imu::{align_imu_timestamps, build_rotation_trajectory, predict_pose_by_imu}, registration::{RegistrationParams, registration}, submap::{SubmapConfig, SubmapManager, matrix4_to_isometry3, transform_submap_points_to_world}, types::GPUContext, voxel_map::{LocalMap, LocalMapConfig}
};
use nalgebra::{Matrix4, Point3, Quaternion, UnitQuaternion, Vector3};

const LOAD_DIR: &str = "data/input/05242026/path03";
const SAVE_DIR: &str = "data/output/05242026/debug";

const DOWNSAMPLE_VOXEL_SIZE: f32 = 0.2; // m
const GICP_ITERATIONS: usize = 5; // Default: 5

const MIN_DIST: f32 = 0.1;
const MAX_DIST: f32 = 20.0;

const MAX_POINTS_PER_VOXEL: usize = 50;
const MIN_POINTS_PER_VOXEL: usize = 3;

const LOCAL_MAP_MAX_FRAMES: usize = 60;
const LOCAL_MAP_MAX_DISTANCE: f32 = 20.0;

const SEARCH_RANGE: usize = 3; // Range of 7x7x7 voxels
const MAX_DIST_SQ: f32 = 0.09; // Optional maximum distance squared

/// dist_sq <= MAX_DIST_SQ を満たす点の割合がこの値を下回るフレームはGICPをスキップする（0.0 で無効）
const MIN_MATCH_RATIO: f32 = 0.65;

/// LocalMap のハッシュグリッドセルサイズ。
/// downsample_voxel_size とは独立に設定する。大きいほど query が高速。
const LOCAL_MAP_INDEX_VOXEL_SIZE: f32 = 0.25;

const SUBMAP_MAX_FRAMES: usize = 30;
const SUBMAP_MAX_DISTANCE: f32 = 5.0;
const SUBMAP_MAX_POINTS: usize = 300_000;

const MIN_SUBMAP_SEPARATION: u64 = 10;
const SEARCH_RADIUS: f32 = 10.0;
const MAX_CANDIDATES: usize = 5;
const TARGET_NEIGHBOR_COUNT: u64 = 2;
const USE_XY_DISTANCE: bool = true;

// --- For loop closuer ---
const LOOP_GICP_ITERATIONS: usize = 10;
const LOOP_ALIGNMENT_VOXEL_SIZE: f32 = 0.25;
const LOOP_SEARCH_RANGE: usize = 4;
const LOOP_MAX_DIST_SQ: f32 = 1.0;
const LOOP_MIN_MATCH_RATIO: f32 = 0.25;

const LOOP_MIN_VALID_RATIO: f32 = 0.35;
const LOOP_MAX_RMSE: f32 = 0.5;
const LOOP_MAX_TRANSLATION_CORRECTION: f32 = 5.0;
const LOOP_MAX_ROTATION_CORRECTION_RAD: f32 = 20.0_f32.to_radians();
// --- For loop closuer ---

// IMU coordination to LiDAR coordination (Robosense 96 beam)
// Quaternion (x, y, z, w): -0.705437, 0.708767, -0.00246579, 0.00097028
// Translation (x, y, z)  : 0.00425, 0.00418, -0.00446  [m]
const IMU_TO_LIDAR_QUAT_X: f64 = -0.705437;
const IMU_TO_LIDAR_QUAT_Y: f64 = 0.708767;
const IMU_TO_LIDAR_QUAT_Z: f64 = -0.00246579;
const IMU_TO_LIDAR_QUAT_W: f64 = 0.00097028;

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("debug")).init();

    let mut performance_logs = PerformanceLogs {
        voxelization_time_ms: Vec::new(),
        // pub covariance_time_ms: f32,
        create_voxel_map_time_ms: Vec::new(),
        knn_search_time_ms: Vec::new(),
        compute_covariances_time_ms: Vec::new(),
        transform_points_time_ms: Vec::new(),
        find_neighbors_time_ms: Vec::new(),
        each_gicp_time_ms: Vec::new(),
        total_gicp_time_ms: Vec::new(),
        query_voxel_time_ms: Vec::new(),
        update_voxel_map_time_ms: Vec::new(),
        merge_time_ms: Vec::new(),
        total_average_time_ms: Vec::new(),
        iteration_count: 0,
    };

    // --- Initialize Vulkan context ---
    let vulkan_context = VulkanContext::new().context("Failed to initialize Vulkan context")?;
    let mut copy_source_gpu_context = GpuTransferDataContext::new(vulkan_context.clone())
        .context("Failed to create GPU transfer context")?;
    let mut copy_target_gpu_context = GpuTransferDataContext::new(vulkan_context.clone())
        .context("Failed to create GPU transfer context")?;
    let mut source_voxel_gpu_context = VoxelGpuContext::new(vulkan_context.clone())?;
    let mut target_voxel_gpu_context = VoxelGpuContext::new(vulkan_context.clone())?;
    let mut source_knn_gpu_context =
        gpu_knn_search::KnnSearchGpuContext::new(vulkan_context.clone())?;
    let mut target_knn_gpu_context =
        gpu_knn_search::KnnSearchGpuContext::new(vulkan_context.clone())?;
    let mut source_covariances_gpu_context =
        gpu_covariances::CovarianceGpuContext::new(vulkan_context.clone())?;
    let mut target_covariances_gpu_context =
        gpu_covariances::CovarianceGpuContext::new(vulkan_context.clone())?;
    let mut transform_gpu_context =
        gpu_transform::TransformGpuContext::new(vulkan_context.clone())?;
    let mut search_neighbor_gpu_context = SearchGpuContext::new(vulkan_context.clone())?;
    let mut gicp_gpu_context = GicpGpuContext::new(vulkan_context.clone())?;
    // --- Initialize Vulkan context ---

    let mut gpu_context = GPUContext {
        copy_source_gpu_context,
        copy_target_gpu_context,
        source_voxel_gpu_context,
        target_voxel_gpu_context,
        source_covariances_gpu_context,
        target_covariances_gpu_context,
        transform_gpu_context,
        search_neighbor_gpu_context,
        gicp_gpu_context,
    };

    let mut submap_manager = SubmapManager::new(SubmapConfig {
        max_frames_per_submap: SUBMAP_MAX_FRAMES,
        max_distance_per_submap: SUBMAP_MAX_DISTANCE,
        max_points_per_submap: SUBMAP_MAX_POINTS,
    });

    let loop_candidate_finder = LoopCandidateFinder::new(LoopCandidateConfig {
        min_submap_separation: MIN_SUBMAP_SEPARATION,
        search_radius: SEARCH_RADIUS,
        max_candidates: MAX_CANDIDATES,
        target_neighbor_count: TARGET_NEIGHBOR_COUNT,
        use_xy_distance: USE_XY_DISTANCE,
    });

    let loop_alignment_config = LoopAlignmentConfig {
        voxel_size: LOOP_ALIGNMENT_VOXEL_SIZE,
        gicp_iterations: LOOP_GICP_ITERATIONS,
        search_range: LOOP_SEARCH_RANGE,
        max_dist_sq: LOOP_MAX_DIST_SQ,
        min_match_ratio: LOOP_MIN_MATCH_RATIO,
        min_valid_ratio: LOOP_MIN_VALID_RATIO,
        max_rmse: LOOP_MAX_RMSE,
        max_translation_correction: LOOP_MAX_TRANSLATION_CORRECTION,
        max_rotation_correction_rad: LOOP_MAX_ROTATION_CORRECTION_RAD,
    };

    let mut pose_graph = PoseGraph::new();

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
    gpu_context
        .copy_target_gpu_context
        .copy_data_to_gpu(&points_vec, points_vec.len())?;
    // --- Copy points to gpu memory ---

    // --- Downsample for density normalization ---
    let downsample_voxel_size = DOWNSAMPLE_VOXEL_SIZE;
    let points_num = points_vec.len();

    let downsampled_points_vec = gpu_context
        .target_voxel_gpu_context
        .voxelization(&gpu_context.copy_target_gpu_context, downsample_voxel_size)?;
    // let downsampled_init_points = voxel_downsample_points(&points, downsample_voxel_size);

    // let downsampled_pcd = convert_vec_to_xyz(&downsampled_points_vec);
    // let save_file = format!("{}/debug/downsampled_init.pcd", SAVE_DIR);
    // save_pcd_xyzit(&downsampled_pcd, &save_file)?;
    // --- Downsample for density normalization ---

    // --- Build local voxel map (sliding window) ---
    let downsampled_points = convert_vec_to_point3(&downsampled_points_vec);
    let mut global_voxel_map = LocalMap::new(LocalMapConfig {
        index_voxel_size: LOCAL_MAP_INDEX_VOXEL_SIZE,
        max_points_per_voxel: MAX_POINTS_PER_VOXEL,
        min_points_per_voxel: MIN_POINTS_PER_VOXEL,
        max_frames: usize::MAX,      // No limit on frames for global map
        max_distance: f32::INFINITY, // No distance-based eviction for global map
    });
    global_voxel_map.insert_frame(&downsampled_points, Point3::origin());

    let mut local_voxel_map = LocalMap::new(LocalMapConfig {
        index_voxel_size: LOCAL_MAP_INDEX_VOXEL_SIZE,
        max_points_per_voxel: MAX_POINTS_PER_VOXEL,
        min_points_per_voxel: MIN_POINTS_PER_VOXEL,
        max_frames: LOCAL_MAP_MAX_FRAMES,
        max_distance: LOCAL_MAP_MAX_DISTANCE,
    });
    local_voxel_map.insert_frame(&downsampled_points, Point3::origin());
    // --- Build local voxel map (sliding window) ---

    let init_pose = matrix4_to_isometry3(&Matrix4::<f64>::identity());

    submap_manager.insert_frame(0, init_pose, &downsampled_points, &downsampled_points);

    let mut prev_frame_start_time = pcd
        .iter()
        .map(|p| p.timestamp)
        .fold(f64::INFINITY, f64::min);

    for (i, pcd_path) in pcd_files.iter().enumerate().skip(1) {
        log::info!("Processing frame {}: {}", i, pcd_path.to_string_lossy());
        performance_logs.iteration_count += 1;

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

        // --- Create local map ---
        let start = Instant::now();
        let t = pose_prediction.0.column(3);
        let sensor_origin_global = Point3::new(t[0] as f32, t[1] as f32, t[2] as f32);
        let local_map = local_voxel_map
            .query_points_within_radius(&sensor_origin_global, LOCAL_MAP_MAX_DISTANCE);
        let duration = start.elapsed();
        performance_logs
            .create_voxel_map_time_ms
            .push(duration.as_secs_f32() * 1000.0);
        log::debug!(
            "Created local map with {} points in {:.2} ms",
            local_map.len(),
            performance_logs.create_voxel_map_time_ms.last().unwrap()
        );
        // --- Create local map ---

        // --- Convert data to Vec format for GPU ---
        let points_vec = convert_point3_to_vec(&deskewed_points);
        let local_map_points_vec = convert_point3_to_vec(&local_map);
        // --- Convert data to Vec format for GPU ---

        let registration_params = RegistrationParams {
            gicp_iterations: GICP_ITERATIONS,
            search_range: SEARCH_RANGE,
            max_dist_sq: MAX_DIST_SQ,
            min_match_ratio: MIN_MATCH_RATIO,
        };

        // --- Registration ---
        let (frame_valid, current_transform, downsampled_source_points_vec) = registration(
            &points_vec,
            &local_map_points_vec,
            downsample_voxel_size,
            &pose_prediction.0,
            &mut gpu_context,
            &registration_params,
            &mut performance_logs,
        )?;
        // --- Registration ---

        if !frame_valid {
            prev_frame_start_time = current_frame_start_time;
            continue;
        }

        // --- Update local map ---
        let t = current_transform.column(3);
        let origin = Point3::new(t[0] as f32, t[1] as f32, t[2] as f32);

        let mut downsampled_source_points = convert_vec_to_point3(&downsampled_source_points_vec);

        let frame_pose_world = matrix4_to_isometry3(&current_transform);

        if let Some(new_submap_id) = submap_manager.insert_frame(
            i as u64,
            frame_pose_world,
            &downsampled_source_points,
            &downsampled_source_points,
        ) {
            log::info!(
                "Created submap {} / total submaps = {}",
                new_submap_id,
                submap_manager.len()
            );

            // --- PoseGraph: add node and odometry edge ---
            {
                let new_submap = submap_manager
                    .get(new_submap_id)
                    .expect("new submap must exist");

                pose_graph.add_node_from_submap(new_submap);

                if new_submap_id > 0 {
                    if let Some(prev_submap) = submap_manager.get(new_submap_id - 1) {
                        pose_graph.add_odometry_edge(
                            prev_submap,
                            new_submap,
                            default_odometry_information(),
                        );

                        log::info!(
                            "PoseGraph odometry edge added: {} -> {}",
                            prev_submap.id,
                            new_submap.id,
                        );
                    }
                }

                pose_graph.print_summary();
            }

            let candidates = loop_candidate_finder.find_candidates(&submap_manager, new_submap_id);

            if candidates.is_empty() {
                log::debug!("No loop candidates for submap {}", new_submap_id);
            } else {
                log::info!(
                    "Loop candidates for submap {}: {} candidates",
                    new_submap_id,
                    candidates.len()
                );

                for c in &candidates {
                    log::info!(
                        "  candidate={} distance={:.2}m separation={} target_submaps={:?}",
                        c.candidate_id,
                        c.distance,
                        c.submap_separation,
                        c.target_submap_ids,
                    );
                }
            }

            for cand in &candidates {
                let target_cloud = loop_candidate_finder.build_target_cloud(&submap_manager, cand);

                let Some(source_cloud) =
                    loop_candidate_finder.build_source_cloud(&submap_manager, cand)
                else {
                    continue;
                };

                log::info!(
                    "Loop target built: current={} candidate={} target_submaps={:?} source_pts={} target_pts={}",
                    cand.current_id,
                    cand.candidate_id,
                    cand.target_submap_ids,
                    source_cloud.points_world.len(),
                    target_cloud.points_world.len(),
                );

                match align_loop_candidate(
                    &loop_alignment_config,
                    &submap_manager,
                    cand,
                    &source_cloud,
                    &target_cloud,
                    &mut gpu_context,
                    &mut performance_logs,
                ) {
                    Ok(Some(mut loop_constraint)) => {
                        log::info!(
                            "LoopConstraint candidate created: candidate={} -> current={} valid_ratio={:.1}% rmse={:.4}",
                            loop_constraint.candidate_id,
                            loop_constraint.current_id,
                            loop_constraint.score.valid_ratio * 100.0,
                            loop_constraint.score.rmse,
                        );

                        loop_constraint.information = default_loop_information();

                        pose_graph.add_loop_constraint(&loop_constraint);

                        log::info!(
                            "PoseGraph loop edge added: {} -> {}",
                            loop_constraint.candidate_id,
                            loop_constraint.current_id,
                        );

                        pose_graph.print_summary();
                    }
                    Ok(None) => {
                        log::debug!(
                            "Loop alignment not accepted: current={} candidate={}",
                            cand.current_id,
                            cand.candidate_id,
                        );
                    }
                    Err(e) => {
                        log::warn!(
                            "Loop alignment error: current={} candidate={} error={:?}",
                            cand.current_id,
                            cand.candidate_id,
                            e,
                        );
                    }
                }
            }
        }

        let rotation = current_transform
            .fixed_view::<3, 3>(0, 0)
            .into_owned()
            .cast::<f32>();
        let translation = current_transform
            .fixed_view::<3, 1>(0, 3)
            .into_owned()
            .cast::<f32>();

        // lidar coord to global coord
        for p in &mut downsampled_source_points {
            let rotated = rotation * p.coords + translation;
            p.coords = rotated;
        }

        let start_update_voxel_map = Instant::now();
        local_voxel_map.insert_frame(&downsampled_source_points, origin);
        global_voxel_map.insert_frame(&downsampled_source_points, origin);
        let duration_update_voxel_map = start_update_voxel_map.elapsed();
        performance_logs
            .update_voxel_map_time_ms
            .push(duration_update_voxel_map.as_secs_f32() * 1000.0);
        log::debug!(
            "Local map updated in {:.2} ms",
            performance_logs.update_voxel_map_time_ms.last().unwrap()
        );
        // --- Update local map ---

        let prev_pos = current_global_pose.fixed_view::<3, 1>(0, 3).into_owned();
        let new_pos = current_transform.fixed_view::<3, 1>(0, 3).into_owned();
        let dt = (current_frame_start_time - prev_frame_start_time).max(1e-6);
        current_velocity = ((new_pos - prev_pos) / dt).cap_magnitude(2.0); // 速度の上限を2 m/sに設定
        current_global_pose = current_transform;
        prev_frame_start_time = current_frame_start_time; // 次フレームのIMU積分の開始時刻を更新
    }

    submap_manager.finalize_active();

    let mut submap_global_points = Vec::<Point3<f32>>::new();

    for submap in &submap_manager.submaps {
        let pts = transform_submap_points_to_world(submap);
        submap_global_points.extend(pts);
    }

    let submap_global_points_vec = convert_point3_to_vec(&submap_global_points);

    let voxel_size = 0.25;
    let inv_voxel_size = 1.0 / voxel_size;
    let mut voxel_map: std::collections::HashMap<(i64, i64, i64), ([f64; 3], usize)> =
        std::collections::HashMap::new();
    for p in &submap_global_points_vec {
        let key = (
            (p[0] as f64 * inv_voxel_size as f64).floor() as i64,
            (p[1] as f64 * inv_voxel_size as f64).floor() as i64,
            (p[2] as f64 * inv_voxel_size as f64).floor() as i64,
        );
        let entry = voxel_map.entry(key).or_insert(([0.0; 3], 0));
        entry.0[0] += p[0] as f64;
        entry.0[1] += p[1] as f64;
        entry.0[2] += p[2] as f64;
        entry.1 += 1;
    }
    let downsampled_sub_map_points_vec: Vec<[f32; 3]> = voxel_map
        .values()
        .map(|(sum, count)| {
            let n = *count as f64;
            [
                (sum[0] / n) as f32,
                (sum[1] / n) as f32,
                (sum[2] / n) as f32,
            ]
        })
        .collect();

    let save_path = format!("{}/final_submap_map.pcd", SAVE_DIR);
    save_pcd_xyzit(
        &convert_vec_to_xyz(&downsampled_sub_map_points_vec),
        &save_path,
    )?;

    log::info!(
        "Saved final submap map: {}, submaps={}",
        save_path,
        submap_manager.len()
    );

    // --- Save logs ---
    let log_save_path = format!(
        "{}/performance_logs_v-{}.json",
        SAVE_DIR, DOWNSAMPLE_VOXEL_SIZE
    );
    let log_file = std::fs::File::create(&log_save_path)?;
    serde_json::to_writer_pretty(log_file, &performance_logs)?;
    log::info!("Performance logs saved to {}", log_save_path);
    // --- Save logs ---

    // --- Print performance statistics ---
    let avg = |v: &Vec<f32>| -> f32 {
        if v.is_empty() {
            0.0
        } else {
            v.iter().sum::<f32>() / v.len() as f32
        }
    };
    let n = performance_logs.iteration_count;
    let each_gicp_avg = avg(&performance_logs.each_gicp_time_ms);
    // each_gicp_time_ms は GICP_ITERATIONS 分のエントリが含まれるので 1イテレーションあたりに換算
    let each_gicp_per_iter = if n > 0 {
        performance_logs.each_gicp_time_ms.iter().sum::<f32>() / n as f32
    } else {
        0.0
    };

    log::info!("===== Performance Statistics ({} frames) =====", n);
    log::info!("  Downsample voxel size   : {} m", DOWNSAMPLE_VOXEL_SIZE);
    log::info!("  GICP iteration count          : {}", GICP_ITERATIONS);
    log::info!(
        "  Query local map        : {:>8.2} ms/frame",
        avg(&performance_logs.create_voxel_map_time_ms)
    );
    log::info!(
        "  Voxelization           : {:>8.2} ms/frame",
        avg(&performance_logs.voxelization_time_ms)
    );
    log::info!(
        "  KNN search             : {:>8.2} ms/frame",
        avg(&performance_logs.knn_search_time_ms)
    );
    log::info!(
        "  Covariance computation : {:>8.2} ms/frame",
        avg(&performance_logs.compute_covariances_time_ms)
    );
    log::info!(
        "  Point transform        : {:>8.2} ms/call ",
        avg(&performance_logs.transform_points_time_ms)
    );
    log::info!(
        "  Neighbor search        : {:>8.2} ms/call ",
        avg(&performance_logs.find_neighbors_time_ms)
    );
    log::info!("  GICP solve (per call)  : {:>8.2} ms/call ", each_gicp_avg);
    log::info!(
        "  GICP total             : {:>8.2} ms/frame",
        avg(&performance_logs.total_gicp_time_ms)
    );
    log::info!(
        "  Map update             : {:>8.2} ms/frame",
        avg(&performance_logs.update_voxel_map_time_ms)
    );
    log::info!("================================================");

    // --- Debug ---

    // --- Debug ---

    // --- Save final local map for visualization ---
    let final_global_map_points_vec = convert_point3_to_vec(&global_voxel_map.get_all_points());

    // --- CPU voxelization of final global map ---
    let voxel_size = 0.25;
    let inv_voxel_size = 1.0 / voxel_size;
    let mut voxel_map: std::collections::HashMap<(i64, i64, i64), ([f64; 3], usize)> =
        std::collections::HashMap::new();
    for p in &final_global_map_points_vec {
        let key = (
            (p[0] as f64 * inv_voxel_size as f64).floor() as i64,
            (p[1] as f64 * inv_voxel_size as f64).floor() as i64,
            (p[2] as f64 * inv_voxel_size as f64).floor() as i64,
        );
        let entry = voxel_map.entry(key).or_insert(([0.0; 3], 0));
        entry.0[0] += p[0] as f64;
        entry.0[1] += p[1] as f64;
        entry.0[2] += p[2] as f64;
        entry.1 += 1;
    }
    let downsampled_global_map_points_vec: Vec<[f32; 3]> = voxel_map
        .values()
        .map(|(sum, count)| {
            let n = *count as f64;
            [
                (sum[0] / n) as f32,
                (sum[1] / n) as f32,
                (sum[2] / n) as f32,
            ]
        })
        .collect();
    log::info!(
        "CPU voxelization: {} -> {} points",
        final_global_map_points_vec.len(),
        downsampled_global_map_points_vec.len()
    );
    // --- CPU voxelization of final global map ---

    // let final_local_map_points = convert_vec_to_point3(&final_local_map_points_vec);
    let save_path = format!(
        "{}/final_global_map_v-{}.pcd",
        SAVE_DIR, DOWNSAMPLE_VOXEL_SIZE
    );
    // save_pcd_xyzit(
    //     &convert_vec_to_xyz(&final_global_map_points_vec),
    //     &save_path,
    // )?;
    // --- Save final local map for visualization ---
    let save_path = format!("{}/final_global_map_v-{}.pcd", SAVE_DIR, voxel_size);
    save_pcd_xyzit(
        &convert_vec_to_xyz(&downsampled_global_map_points_vec),
        &save_path,
    )?;

    Ok(())
}
