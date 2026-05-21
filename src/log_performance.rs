use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct PerformanceLogs {
    pub voxelization_time_ms: Vec<f32>,
    // pub covariance_time_ms: f32,
    pub create_voxel_map_time_ms: Vec<f32>,
    pub knn_search_time_ms: Vec<f32>,
    pub compute_covariances_time_ms: Vec<f32>,
    pub transform_points_time_ms: Vec<f32>,
    pub find_neighbors_time_ms: Vec<f32>,
    pub each_gicp_time_ms: Vec<f32>,
    pub total_gicp_time_ms: Vec<f32>,
    pub query_voxel_time_ms: Vec<f32>,
    pub update_voxel_map_time_ms: Vec<f32>,
    pub merge_time_ms: Vec<f32>,
    pub total_average_time_ms: Vec<f32>,
    pub iteration_count: usize,
}
