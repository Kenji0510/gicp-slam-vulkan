use std::time::Instant;

use anyhow::{Context, Result};
use nalgebra::Matrix4;

use crate::{
    gpu_gicp::{GicpStaticBuffers, solve_gicp},
    gpu_transform,
    log_performance::PerformanceLogs,
    registration,
    types::GPUContext,
};

pub struct RegistrationParams {
    pub gicp_iterations: usize,
    pub search_range: usize,
    pub max_dist_sq: f32,
    pub min_match_ratio: f32,
}

// pub fn registration(
//     source_points: &[[f32; 3]],
//     target_points: &[[f32; 3]],
//     downsample_voxel_size: f32,
//     init_pose: &Matrix4<f64>,
//     gpu_context: &mut GPUContext,
//     registration_params: &RegistrationParams,
//     performance_logs: &mut PerformanceLogs,
// ) -> Result<(bool, Matrix4<f64>, Vec<[f32; 3]>)> {
//     // --- Copy points to gpu memory ---
//     gpu_context
//         .copy_source_gpu_context
//         .copy_data_to_gpu(&source_points, source_points.len())?;
//     gpu_context
//         .copy_target_gpu_context
//         .copy_data_to_gpu(&target_points, target_points.len())?;
//     // --- Copy points to gpu memory ---

//     // --- Voxelization ---
//     let start = Instant::now();
//     let downsampled_source_points_vec = gpu_context
//         .source_voxel_gpu_context
//         .voxelization(&gpu_context.copy_source_gpu_context, downsample_voxel_size)?;
//     gpu_context
//         .target_voxel_gpu_context
//         .voxelization_gpu_only(&gpu_context.copy_target_gpu_context, downsample_voxel_size)?;
//     let duration = start.elapsed();
//     performance_logs
//         .voxelization_time_ms
//         .push(duration.as_secs_f32() * 1000.0);
//     log::debug!(
//         "Voxelization time: {:.2} ms",
//         performance_logs.voxelization_time_ms.last().unwrap()
//     );
//     // --- Voxelization ---

//     // --- Compute the covariance for each points ---
//     let start = Instant::now();
//     gpu_context
//         .source_covariances_gpu_context
//         .compute_covariances_gpu_only(&gpu_context.source_voxel_gpu_context)?;
//     gpu_context
//         .target_covariances_gpu_context
//         .compute_covariances_gpu_only(&gpu_context.target_voxel_gpu_context)?;
//     let duration = start.elapsed();
//     performance_logs
//         .compute_covariances_time_ms
//         .push(duration.as_secs_f32() * 1000.0);
//     log::debug!(
//         "Covariance computation completed in {:.2} ms",
//         performance_logs.compute_covariances_time_ms.last().unwrap()
//     );
//     // --- Compute the covariance for each points ---

//     // --- GICP optimization iterations ---
//     let mut current_transform = *init_pose;
//     let gicp_start = Instant::now();
//     let mut frame_valid = true;
//     'gicp: for gicp_iter in 0..registration_params.gicp_iterations {
//         log::info!(
//             "GICP iteration {}/{}",
//             gicp_iter + 1,
//             registration_params.gicp_iterations
//         );

//         let m = current_transform;
//         let mut transform_params = gpu_transform::TransformParams {
//             r00: m[(0, 0)] as f32,
//             r01: m[(0, 1)] as f32,
//             r02: m[(0, 2)] as f32,
//             t0: m[(0, 3)] as f32,
//             r10: m[(1, 0)] as f32,
//             r11: m[(1, 1)] as f32,
//             r12: m[(1, 2)] as f32,
//             t1: m[(1, 3)] as f32,
//             r20: m[(2, 0)] as f32,
//             r21: m[(2, 1)] as f32,
//             r22: m[(2, 2)] as f32,
//             t2: m[(2, 3)] as f32,
//             num_points: gpu_context.source_voxel_gpu_context.h_downsampled_pts_num as u32,
//         };

//         // --- Transform the points ---
//         let start_transform = Instant::now();
//         gpu_context.transform_gpu_context.transform(
//             &gpu_context.source_voxel_gpu_context,
//             &gpu_context.source_covariances_gpu_context,
//             transform_params,
//         )?;
//         let duration_transform = start_transform.elapsed();
//         performance_logs
//             .transform_points_time_ms
//             .push(duration_transform.as_secs_f32() * 1000.0);
//         log::debug!(
//             "Point transformation completed in {:.2} ms",
//             performance_logs.transform_points_time_ms.last().unwrap()
//         );
//         // --- Transform the points ---

//         // --- Search neighbor points for each point ---
//         let start_search_neighbor = Instant::now();
//         let (_h_neighbor_indices, h_neighbor_dists_sq) =
//             gpu_context.search_neighbor_gpu_context.search_neighbor(
//                 &gpu_context.transform_gpu_context,
//                 gpu_context.source_voxel_gpu_context.h_downsampled_pts_num,
//                 &gpu_context.target_voxel_gpu_context,
//                 gpu_context.target_voxel_gpu_context.h_downsampled_pts_num,
//                 registration_params.search_range,
//                 registration_params.max_dist_sq,
//             )?;
//         let duration_search_neighbor = start_search_neighbor.elapsed();
//         performance_logs
//             .find_neighbors_time_ms
//             .push(duration_search_neighbor.as_secs_f32() * 1000.0);
//         log::debug!(
//             "Neighbor search completed in {:.2} ms",
//             performance_logs.find_neighbors_time_ms.last().unwrap()
//         );

//         // --- Check match ratio on first iteration ---
//         if gicp_iter == 0 {
//             let total = h_neighbor_dists_sq.len();
//             let valid_count = h_neighbor_dists_sq
//                 .iter()
//                 .filter(|&&d| d >= 0.0 && d <= registration_params.max_dist_sq)
//                 .count();
//             let ratio = if total > 0 {
//                 valid_count as f32 / total as f32
//             } else {
//                 0.0
//             };
//             log::debug!(
//                 "Match ratio (dist <= {:.3}): {:.1}% ({}/{})",
//                 registration_params.max_dist_sq,
//                 ratio * 100.0,
//                 valid_count,
//                 total
//             );
//             if ratio < registration_params.min_match_ratio {
//                 log::warn!(
//                     "Frame skipped: match ratio {:.1}% < {:.1}% threshold ({}/{})",
//                     ratio * 100.0,
//                     registration_params.min_match_ratio * 100.0,
//                     valid_count,
//                     total
//                 );
//                 frame_valid = false;
//                 break 'gicp;
//             }
//         }
//         // --- Check match ratio on first iteration ---
//         // --- Search neighbor points for each point ---

//         // --- GICP optimization ---
//         let gicp_bufs = GicpStaticBuffers {
//             d_source_pts: gpu_context
//                 .transform_gpu_context
//                 .d_buf_output_pts
//                 .as_ref()
//                 .context("Failed to get transformed source points buffer")?
//                 .clone(),
//             d_source_covs: gpu_context
//                 .transform_gpu_context
//                 .d_buf_output_covs
//                 .as_ref()
//                 .context("Failed to get transformed source covariances buffer")?
//                 .clone(),
//             d_target_pts: gpu_context
//                 .target_voxel_gpu_context
//                 .d_buf_out_pts
//                 .as_ref()
//                 .context("Failed to get target points buffer")?
//                 .clone(),
//             d_target_covs: gpu_context
//                 .target_covariances_gpu_context
//                 .d_buf_covariances
//                 .as_ref()
//                 .context("Failed to get target covariances buffer")?
//                 .clone(),
//         };

//         let gicp_iteration_start = Instant::now();
//         let (h_vec, b_vec) = gpu_context.gicp_gpu_context.compute_gicp(
//             &gicp_bufs,
//             &gpu_context.search_neighbor_gpu_context,
//             gpu_context.source_voxel_gpu_context.h_downsampled_pts_num,
//             gpu_context.target_voxel_gpu_context.h_downsampled_pts_num,
//             registration_params.max_dist_sq,
//         )?;
//         let gicp_iteration_duration = gicp_iteration_start.elapsed();
//         performance_logs
//             .each_gicp_time_ms
//             .push(gicp_iteration_duration.as_secs_f32() * 1000.0);
//         log::debug!(
//             "GICP iteration {} completed in {:.2} ms",
//             gicp_iter + 1,
//             performance_logs.each_gicp_time_ms.last().unwrap()
//         );

//         let h = nalgebra::Matrix6::from_row_slice(&h_vec);
//         let b = nalgebra::Vector6::from_row_slice(&b_vec);
//         if let Some(delta) = solve_gicp((h, b), 1.0e-4) {
//             current_transform = delta * current_transform;
//         }
//         // --- GICP optimization ---
//     }
//     let gicp_duration = gicp_start.elapsed();
//     performance_logs
//         .total_gicp_time_ms
//         .push(gicp_duration.as_secs_f32() * 1000.0);
//     log::debug!(
//         "Total GICP optimization completed in {:.2} ms",
//         performance_logs.total_gicp_time_ms.last().unwrap()
//     );
//     // --- GICP optimization iterations ---

//     Ok((
//         frame_valid,
//         current_transform,
//         downsampled_source_points_vec,
//     ))
// }

pub fn registration(
    source_points: &[[f32; 3]],
    target_points: &[[f32; 3]],
    downsample_voxel_size: f32,
    init_pose: &Matrix4<f64>,
    gpu_context: &mut GPUContext,
    registration_params: &RegistrationParams,
    performance_logs: &mut PerformanceLogs,
) -> Result<(bool, Matrix4<f64>, Vec<[f32; 3]>)> {
    let result = registration_with_metrics(
        source_points,
        target_points,
        downsample_voxel_size,
        init_pose,
        gpu_context,
        registration_params,
        performance_logs,
    )?;

    Ok((
        result.frame_valid,
        result.transform,
        result.downsampled_source_points,
    ))
}

#[derive(Debug, Clone)]
pub struct MatchMetrics {
    pub num_total: usize,
    pub num_valid: usize,
    pub valid_ratio: f32,
    pub mean_dist_sq: f32,
    pub rmse: f32,
}

#[derive(Debug, Clone)]
pub struct RegistrationMetrics {
    pub first_iter: MatchMetrics,
    pub final_iter: MatchMetrics,
}

#[derive(Debug, Clone)]
pub struct RegistrationResult {
    pub frame_valid: bool,
    pub transform: Matrix4<f64>,
    pub downsampled_source_points: Vec<[f32; 3]>,
    pub metrics: RegistrationMetrics,
}

fn compute_match_metrics(dists_sq: &[f32], max_dist_sq: f32) -> MatchMetrics {
    let num_total = dists_sq.len();

    let mut num_valid = 0usize;
    let mut sum_dist_sq = 0.0f32;

    for &d in dists_sq {
        if d.is_finite() && d >= 0.0 && d <= max_dist_sq {
            num_valid += 1;
            sum_dist_sq += d;
        }
    }

    let valid_ratio = if num_total > 0 {
        num_valid as f32 / num_total as f32
    } else {
        0.0
    };

    let mean_dist_sq = if num_valid > 0 {
        sum_dist_sq / num_valid as f32
    } else {
        f32::INFINITY
    };

    let rmse = mean_dist_sq.sqrt();

    MatchMetrics {
        num_total,
        num_valid,
        valid_ratio,
        mean_dist_sq,
        rmse,
    }
}

pub fn registration_with_metrics(
    source_points: &[[f32; 3]],
    target_points: &[[f32; 3]],
    downsample_voxel_size: f32,
    init_pose: &Matrix4<f64>,
    gpu_context: &mut GPUContext,
    registration_params: &RegistrationParams,
    performance_logs: &mut PerformanceLogs,
) -> Result<RegistrationResult> {
    gpu_context
        .copy_source_gpu_context
        .copy_data_to_gpu(&source_points, source_points.len())?;
    gpu_context
        .copy_target_gpu_context
        .copy_data_to_gpu(&target_points, target_points.len())?;

    let start = Instant::now();
    let downsampled_source_points_vec = gpu_context
        .source_voxel_gpu_context
        .voxelization(&gpu_context.copy_source_gpu_context, downsample_voxel_size)?;

    gpu_context
        .target_voxel_gpu_context
        .voxelization_gpu_only(&gpu_context.copy_target_gpu_context, downsample_voxel_size)?;

    let duration = start.elapsed();
    performance_logs
        .voxelization_time_ms
        .push(duration.as_secs_f32() * 1000.0);

    let start = Instant::now();
    gpu_context
        .source_covariances_gpu_context
        .compute_covariances_gpu_only(&gpu_context.source_voxel_gpu_context)?;
    gpu_context
        .target_covariances_gpu_context
        .compute_covariances_gpu_only(&gpu_context.target_voxel_gpu_context)?;

    let duration = start.elapsed();
    performance_logs
        .compute_covariances_time_ms
        .push(duration.as_secs_f32() * 1000.0);

    let mut current_transform = *init_pose;
    let gicp_start = Instant::now();
    let mut frame_valid = true;

    let mut first_iter_metrics = MatchMetrics {
        num_total: 0,
        num_valid: 0,
        valid_ratio: 0.0,
        mean_dist_sq: f32::INFINITY,
        rmse: f32::INFINITY,
    };

    let mut final_iter_metrics = first_iter_metrics.clone();

    'gicp: for gicp_iter in 0..registration_params.gicp_iterations {
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
            num_points: gpu_context.source_voxel_gpu_context.h_downsampled_pts_num as u32,
        };

        let start_transform = Instant::now();
        gpu_context.transform_gpu_context.transform(
            &gpu_context.source_voxel_gpu_context,
            &gpu_context.source_covariances_gpu_context,
            transform_params,
        )?;
        let duration_transform = start_transform.elapsed();
        performance_logs
            .transform_points_time_ms
            .push(duration_transform.as_secs_f32() * 1000.0);

        let start_search_neighbor = Instant::now();
        let (_h_neighbor_indices, h_neighbor_dists_sq) =
            gpu_context.search_neighbor_gpu_context.search_neighbor(
                &gpu_context.transform_gpu_context,
                gpu_context.source_voxel_gpu_context.h_downsampled_pts_num,
                &gpu_context.target_voxel_gpu_context,
                gpu_context.target_voxel_gpu_context.h_downsampled_pts_num,
                registration_params.search_range,
                registration_params.max_dist_sq,
            )?;
        let duration_search_neighbor = start_search_neighbor.elapsed();
        performance_logs
            .find_neighbors_time_ms
            .push(duration_search_neighbor.as_secs_f32() * 1000.0);

        let metrics = compute_match_metrics(
            &h_neighbor_dists_sq,
            registration_params.max_dist_sq,
        );

        if gicp_iter == 0 {
            first_iter_metrics = metrics.clone();

            log::debug!(
                "First iter match ratio: {:.1}% ({}/{}), rmse={:.4}",
                first_iter_metrics.valid_ratio * 100.0,
                first_iter_metrics.num_valid,
                first_iter_metrics.num_total,
                first_iter_metrics.rmse,
            );

            if first_iter_metrics.valid_ratio < registration_params.min_match_ratio {
                log::warn!(
                    "Registration rejected: match ratio {:.1}% < {:.1}%",
                    first_iter_metrics.valid_ratio * 100.0,
                    registration_params.min_match_ratio * 100.0,
                );
                frame_valid = false;
                final_iter_metrics = first_iter_metrics.clone();
                break 'gicp;
            }
        }

        final_iter_metrics = metrics;

        let gicp_bufs = GicpStaticBuffers {
            d_source_pts: gpu_context
                .transform_gpu_context
                .d_buf_output_pts
                .as_ref()
                .context("Failed to get transformed source points buffer")?
                .clone(),
            d_source_covs: gpu_context
                .transform_gpu_context
                .d_buf_output_covs
                .as_ref()
                .context("Failed to get transformed source covariances buffer")?
                .clone(),
            d_target_pts: gpu_context
                .target_voxel_gpu_context
                .d_buf_out_pts
                .as_ref()
                .context("Failed to get target points buffer")?
                .clone(),
            d_target_covs: gpu_context
                .target_covariances_gpu_context
                .d_buf_covariances
                .as_ref()
                .context("Failed to get target covariances buffer")?
                .clone(),
        };

        let gicp_iteration_start = Instant::now();
        let (h_vec, b_vec) = gpu_context.gicp_gpu_context.compute_gicp(
            &gicp_bufs,
            &gpu_context.search_neighbor_gpu_context,
            gpu_context.source_voxel_gpu_context.h_downsampled_pts_num,
            gpu_context.target_voxel_gpu_context.h_downsampled_pts_num,
            registration_params.max_dist_sq,
        )?;
        let gicp_iteration_duration = gicp_iteration_start.elapsed();
        performance_logs
            .each_gicp_time_ms
            .push(gicp_iteration_duration.as_secs_f32() * 1000.0);

        let h = nalgebra::Matrix6::from_row_slice(&h_vec);
        let b = nalgebra::Vector6::from_row_slice(&b_vec);

        if let Some(delta) = solve_gicp((h, b), 1.0e-4) {
            current_transform = delta * current_transform;
        }
    }

    let gicp_duration = gicp_start.elapsed();
    performance_logs
        .total_gicp_time_ms
        .push(gicp_duration.as_secs_f32() * 1000.0);

    Ok(RegistrationResult {
        frame_valid,
        transform: current_transform,
        downsampled_source_points: downsampled_source_points_vec,
        metrics: RegistrationMetrics {
            first_iter: first_iter_metrics,
            final_iter: final_iter_metrics,
        },
    })
}