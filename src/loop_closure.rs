use anyhow::{Context, Result};
use nalgebra::{Isometry3, Matrix4, Matrix6, Point3};

use crate::{
    log_performance::PerformanceLogs,
    registration::{RegistrationParams, registration_with_metrics},
    submap::{SubmapId, SubmapManager, matrix4_to_isometry3},
    types::GPUContext,
};

#[derive(Debug, Clone)]
pub struct LoopCandidateConfig {
    pub min_submap_separation: u64,

    pub search_radius: f32,

    pub max_candidates: usize,

    pub target_neighbor_count: u64,

    pub use_xy_distance: bool,
}

#[derive(Debug, Clone)]
pub struct LoopCandidate {
    pub current_id: SubmapId,
    pub candidate_id: SubmapId,

    pub distance: f32,

    pub submap_separation: u64,

    pub target_submap_ids: Vec<SubmapId>,
}

pub struct LoopCandidateFinder {
    pub config: LoopCandidateConfig,
}

impl LoopCandidateFinder {
    pub fn new(config: LoopCandidateConfig) -> Self {
        Self { config }
    }

    pub fn find_candidates(
        &self,
        submap_manager: &SubmapManager,
        current_id: SubmapId,
    ) -> Vec<LoopCandidate> {
        let Some(current) = submap_manager.get(current_id) else {
            return Vec::new();
        };

        let mut candidates = Vec::<LoopCandidate>::new();

        for past in &submap_manager.submaps {
            if past.id == current.id {
                continue;
            }

            let separation = current.id.saturating_sub(past.id);

            if separation < self.config.min_submap_separation {
                continue;
            }

            let distance = if self.config.use_xy_distance {
                let dx = current.center_world.x - past.center_world.x;
                let dy = current.center_world.y - past.center_world.y;
                (dx * dx + dy * dy).sqrt()
            } else {
                (current.center_world - past.center_world).norm()
            };

            if distance > self.config.search_radius {
                continue;
            }

            let target_submap_ids = self.collect_neighbor_submap_ids(submap_manager, past.id);

            candidates.push(LoopCandidate {
                current_id,
                candidate_id: past.id,
                distance,
                submap_separation: separation,
                target_submap_ids,
            });
        }

        candidates.sort_by(|a, b| {
            a.distance
                .partial_cmp(&b.distance)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        candidates.truncate(self.config.max_candidates);
        candidates
    }

    fn collect_neighbor_submap_ids(
        &self,
        submap_manager: &SubmapManager,
        center_id: SubmapId,
    ) -> Vec<SubmapId> {
        let n = self.config.target_neighbor_count;

        let start = center_id.saturating_sub(n);
        let end = center_id.saturating_add(n);

        let mut ids = Vec::new();

        for id in start..=end {
            if submap_manager.get(id).is_some() {
                ids.push(id);
            }
        }

        ids
    }

    pub fn build_target_cloud(
        &self,
        submap_manager: &SubmapManager,
        candidate: &LoopCandidate,
    ) -> LoopTargetCloud {
        build_loop_target_cloud(
            submap_manager,
            candidate.candidate_id,
            &candidate.target_submap_ids,
        )
    }

    pub fn build_source_cloud(
        &self,
        submap_manager: &SubmapManager,
        candidate: &LoopCandidate,
    ) -> Option<LoopSourceCloud> {
        build_loop_source_cloud(submap_manager, candidate.current_id)
    }
}

#[derive(Debug, Clone)]
pub struct LoopTargetCloud {
    pub candidate_id: SubmapId,
    pub target_submap_ids: Vec<SubmapId>,
    pub points_world: Vec<Point3<f32>>,
}

pub fn build_loop_target_cloud(
    submap_manager: &SubmapManager,
    candidate_id: SubmapId,
    target_submap_ids: &[SubmapId],
) -> LoopTargetCloud {
    let mut points_world = Vec::<Point3<f32>>::new();

    for &sid in target_submap_ids {
        let Some(submap) = submap_manager.get(sid) else {
            continue;
        };

        points_world.reserve(submap.points_local_reg.len());

        for p_local in &submap.points_local_reg {
            let p_world = submap
                .pose_world
                .transform_point(&p_local.cast::<f64>())
                .cast::<f32>();

            points_world.push(p_world);
        }
    }

    LoopTargetCloud {
        candidate_id,
        target_submap_ids: target_submap_ids.to_vec(),
        points_world,
    }
}

#[derive(Debug, Clone)]
pub struct LoopSourceCloud {
    pub current_id: SubmapId,
    pub points_world: Vec<Point3<f32>>,
}

pub fn build_loop_source_cloud(
    submap_manager: &SubmapManager,
    current_id: SubmapId,
) -> Option<LoopSourceCloud> {
    let current = submap_manager.get(current_id)?;

    let mut points_world = Vec::<Point3<f32>>::with_capacity(current.points_local_reg.len());

    for p_local in &current.points_local_reg {
        let p_world = current
            .pose_world
            .transform_point(&p_local.cast::<f64>())
            .cast::<f32>();

        points_world.push(p_world);
    }

    Some(LoopSourceCloud {
        current_id,
        points_world,
    })
}

#[derive(Debug, Clone)]
pub struct LoopAlignmentConfig {
    pub voxel_size: f32,
    pub gicp_iterations: usize,
    pub search_range: usize,
    pub max_dist_sq: f32,
    pub min_match_ratio: f32,

    pub min_valid_ratio: f32,
    pub max_rmse: f32,
    pub max_translation_correction: f32,
    pub max_rotation_correction_rad: f32,
}

#[derive(Debug, Clone)]
pub struct LoopScore {
    pub valid_ratio: f32,
    pub rmse: f32,
    pub mean_dist_sq: f32,
    pub num_source: usize,
    pub num_valid: usize,
}

#[derive(Debug, Clone)]
pub struct LoopConstraintCandidate {
    pub current_id: SubmapId,
    pub candidate_id: SubmapId,
    pub target_submap_ids: Vec<SubmapId>,

    /// source_world を target_world に合わせる補正
    pub delta_world: Isometry3<f64>,

    /// delta_world * current.pose_world
    pub corrected_current_pose_world: Isometry3<f64>,

    /// pose graph edge用: candidate -> current
    pub relative_pose_candidate_to_current: Isometry3<f64>,

    /// 後でpose graphに渡すため。最初はidentityでOK。
    pub information: Matrix6<f64>,

    pub score: LoopScore,
}

fn isometry3_to_matrix4(iso: &Isometry3<f64>) -> Matrix4<f64> {
    iso.to_homogeneous()
}

fn validate_loop_constraint(
    config: &LoopAlignmentConfig,
    constraint: &LoopConstraintCandidate,
) -> bool {
    if constraint.score.valid_ratio < config.min_valid_ratio {
        return false;
    }

    if constraint.score.rmse > config.max_rmse {
        return false;
    }

    let trans_norm = constraint.delta_world.translation.vector.norm() as f32;
    if trans_norm > config.max_translation_correction {
        return false;
    }

    let rot_norm = constraint.delta_world.rotation.angle() as f32;
    if rot_norm > config.max_rotation_correction_rad {
        return false;
    }

    true
}

pub fn align_loop_candidate(
    config: &LoopAlignmentConfig,
    submap_manager: &SubmapManager,
    candidate: &LoopCandidate,
    source_cloud: &LoopSourceCloud,
    target_cloud: &LoopTargetCloud,
    gpu_context: &mut GPUContext,
    performance_logs: &mut PerformanceLogs,
) -> Result<Option<LoopConstraintCandidate>> {
    if source_cloud.points_world.is_empty() || target_cloud.points_world.is_empty() {
        log::warn!(
            "Loop alignment skipped: empty source/target. current={} candidate={} source_pts={} target_pts={}",
            candidate.current_id,
            candidate.candidate_id,
            source_cloud.points_world.len(),
            target_cloud.points_world.len(),
        );
        return Ok(None);
    }

    let current = submap_manager
        .get(candidate.current_id)
        .context("Failed to get current submap")?;

    let candidate_submap = submap_manager
        .get(candidate.candidate_id)
        .context("Failed to get candidate submap")?;

    let source_points: Vec<[f32; 3]> = source_cloud
        .points_world
        .iter()
        .map(|p| [p.x, p.y, p.z])
        .collect();

    let target_points: Vec<[f32; 3]> = target_cloud
        .points_world
        .iter()
        .map(|p| [p.x, p.y, p.z])
        .collect();

    let init_pose = Matrix4::<f64>::identity();

    let registration_params = RegistrationParams {
        gicp_iterations: config.gicp_iterations,
        search_range: config.search_range,
        max_dist_sq: config.max_dist_sq,
        min_match_ratio: config.min_match_ratio,
    };

    let result = registration_with_metrics(
        &source_points,
        &target_points,
        config.voxel_size,
        &init_pose,
        gpu_context,
        &registration_params,
        performance_logs,
    )?;

    if !result.frame_valid {
        log::warn!(
            "Loop alignment rejected by registration: current={} candidate={} first_valid_ratio={:.2}",
            candidate.current_id,
            candidate.candidate_id,
            result.metrics.first_iter.valid_ratio,
        );
        return Ok(None);
    }

    let delta_world = matrix4_to_isometry3(&result.transform);

    let corrected_current_pose_world = delta_world * current.pose_world;

    let relative_pose_candidate_to_current =
        candidate_submap.pose_world.inverse() * corrected_current_pose_world;

    let score = LoopScore {
        valid_ratio: result.metrics.final_iter.valid_ratio,
        rmse: result.metrics.final_iter.rmse,
        mean_dist_sq: result.metrics.final_iter.mean_dist_sq,
        num_source: result.metrics.final_iter.num_total,
        num_valid: result.metrics.final_iter.num_valid,
    };

    let constraint = LoopConstraintCandidate {
        current_id: candidate.current_id,
        candidate_id: candidate.candidate_id,
        target_submap_ids: candidate.target_submap_ids.clone(),
        delta_world,
        corrected_current_pose_world,
        relative_pose_candidate_to_current,
        information: Matrix6::<f64>::identity(),
        score,
    };

    let trans_norm = constraint.delta_world.translation.vector.norm();
    let rot_deg = constraint.delta_world.rotation.angle().to_degrees();

    log::info!(
        "Loop alignment result: current={} candidate={} valid_ratio={:.1}% rmse={:.4} delta_t={:.3}m delta_r={:.2}deg",
        constraint.current_id,
        constraint.candidate_id,
        constraint.score.valid_ratio * 100.0,
        constraint.score.rmse,
        trans_norm,
        rot_deg,
    );

    if !validate_loop_constraint(config, &constraint) {
        log::warn!(
            "Loop constraint rejected: current={} candidate={} valid_ratio={:.1}% rmse={:.4} delta_t={:.3}m delta_r={:.2}deg",
            constraint.current_id,
            constraint.candidate_id,
            constraint.score.valid_ratio * 100.0,
            constraint.score.rmse,
            trans_norm,
            rot_deg,
        );
        return Ok(None);
    }

    log::info!(
        "Loop constraint accepted: candidate={} -> current={} target_submaps={:?}",
        constraint.candidate_id,
        constraint.current_id,
        constraint.target_submap_ids,
    );

    Ok(Some(constraint))
}
