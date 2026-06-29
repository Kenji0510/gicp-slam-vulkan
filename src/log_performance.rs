use serde::{Deserialize, Serialize};

/// Loop closure が受理されたときの RMSE エントリ
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct LoopClosureRmseEntry {
    /// loop closure が発生したフレームインデックス
    pub frame: usize,
    pub current_submap_id: u64,
    pub candidate_submap_id: u64,
    pub rmse: f32,
    pub valid_ratio: f32,
}

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
    /// フレームごとの odometry GICP 最終イテレーションの RMSE（フレームスキップ時は None）
    pub gicp_rmse_per_frame: Vec<Option<f32>>,
    /// 受理された loop closure ごとの RMSE
    pub loop_closure_rmse: Vec<LoopClosureRmseEntry>,
}
