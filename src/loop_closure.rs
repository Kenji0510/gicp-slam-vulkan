use crate::submap::{SubmapId, SubmapManager};

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
}
