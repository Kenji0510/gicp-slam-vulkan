use std::collections::VecDeque;

use nalgebra::{Matrix3, Point3, Vector3};
use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct VoxelKey {
    pub ix: i32,
    pub iy: i32,
    pub iz: i32,
}

/// 1つのvoxelを1つの3D Gaussianとして扱うためのcell。
///
/// GICP用の `gicp_covariance` ではなく、Gaussian分布としての
/// `covariance` と、その逆行列 `information` を持つ。
#[derive(Debug, Clone)]
pub struct VoxelCell {
    /// このvoxel内に入った代表点。
    /// 初期段階ではデバッグ性を優先して保持する。
    /// 将来的には Welford の count / mean / m2 のみへ移行可能。
    /// タプルの 2 要素目はフレーム ID。
    pub points: Vec<(Point3<f32>, u64)>,

    /// Gaussianの平均 μ。
    pub mean: Point3<f32>,

    /// voxel内点群から直接計算した共分散。
    // pub raw_covariance: Matrix3<f32>,

    // /// 固有値clamp後の安定化済みGaussian共分散 Σ。
    // pub covariance: Matrix3<f32>,

    /// Gaussianとしてregistrationに使えるだけの点数と数値安定性があるか。
    pub valid: bool,

    pub voxel_key: VoxelKey,
}

impl VoxelCell {
    pub fn new() -> Self {
        Self {
            points: Vec::new(),
            mean: Point3::new(0.0, 0.0, 0.0),
            // raw_covariance: Matrix3::identity(),
            // covariance: Matrix3::identity(),
            valid: false,
            voxel_key: VoxelKey {
                ix: 0,
                iy: 0,
                iz: 0,
            },
        }
    }

    pub fn point_count(&self) -> usize {
        self.points.len()
    }

    pub fn push_point(
        &mut self,
        p: Point3<f32>,
        frame_id: u64,
        max_points_per_gaussian: usize,
    ) -> bool {
        if self.points.len() >= max_points_per_gaussian {
            return false;
        }
        self.points.push((p, frame_id));
        true
    }

    pub fn compute_average(&self) -> Point3<f32> {
        let mut sum = Vector3::zeros();
        for (p, _) in &self.points {
            sum += p.coords;
        }
        Point3::from(sum / self.points.len() as f32)
    }

    pub fn compute_covariance(cell: &VoxelCell) {}

    // pub fn recompute_covariance(&mut self, min_points_per_gaussian: usize) {
    //     self.valid = false;

    //     if self.points.is_empty() {
    //         self.mean = Point3::new(0.0, 0.0, 0.0);
    //         self.raw_covariance = Matrix3::identity();
    //         self.covariance = Matrix3::identity();
    //         self.valid = false;
    //         return;
    //     }

    //     self.mean = compute_mean_from_points(&self.points);

    //     if self.points.len() < min_points_per_gaussian {
    //         self.raw_covariance = Matrix3::identity();
    //         self.covariance = Matrix3::identity();
    //         self.valid = false;
    //         return;
    //     }

    //     let Some(raw_covariance) = compute_raw_covariance_from_points(&self.points, &self.mean)
    //     else {
    //         self.raw_covariance = Matrix3::identity();
    //         self.covariance = Matrix3::identity();
    //         self.valid = false;
    //         return;
    //     };

    //     let covariance = regularize_gicp_covariance(raw_covariance);

    //     self.raw_covariance = raw_covariance;
    //     self.covariance = covariance;
    //     self.valid = true;
    // }
}

pub type VoxelMap = FxHashMap<VoxelKey, VoxelCell>;

#[inline]
pub fn voxel_key(p: &Point3<f32>, voxel_size: f32) -> VoxelKey {
    VoxelKey {
        ix: (p.x / voxel_size).floor() as i32,
        iy: (p.y / voxel_size).floor() as i32,
        iz: (p.z / voxel_size).floor() as i32,
    }
}

#[inline]
pub fn is_finite_matrix3(m: &Matrix3<f32>) -> bool {
    m.iter().all(|v| v.is_finite())
}

pub fn invert_matrix3_safe(m: Matrix3<f32>) -> Option<Matrix3<f32>> {
    if !is_finite_matrix3(&m) {
        return None;
    }

    let det = m.determinant();
    if !det.is_finite() || det.abs() < 1.0e-12 {
        return None;
    }

    m.try_inverse()
}

fn neighbor_keys(key: &VoxelKey) -> impl IntoIterator<Item = VoxelKey> {
    let mut neighbors = Vec::with_capacity(27);

    for dx in -1..=1 {
        for dy in -1..=1 {
            for dz in -1..=1 {
                neighbors.push(VoxelKey {
                    ix: key.ix + dx,
                    iy: key.iy + dy,
                    iz: key.iz + dz,
                });
            }
        }
    }

    neighbors
}

// fn compute_raw_covariance_from_points(
//     points: &[Point3<f32>],
//     mean: &Point3<f32>,
// ) -> Option<Matrix3<f32>> {
//     if points.len() < 3 {
//         return None;
//     }

//     let mut cov = Matrix3::<f32>::zeros();
//     for p in points {
//         let d = p - mean;
//         cov += d * d.transpose();
//     }
//     cov /= (points.len() - 1) as f32;

//     if !is_finite_matrix3(&cov) {
//         return None;
//     }

//     Some(cov)
// }

// pub fn regularize_gicp_covariance(cov: Matrix3<f32>) -> Matrix3<f32> {
//     let eigen = SymmetricEigen::new(cov);
//     let rot = eigen.eigenvectors;
//     let mut vals = eigen.eigenvalues;

//     let mut pairs: Vec<(f32, usize)> = vals
//         .iter()
//         .cloned()
//         .enumerate()
//         .map(|(i, v)| (v, i))
//         .collect();
//     pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

//     let min_idx = pairs[0].1;
//     vals[min_idx] = 1e-3; // 法線方向を薄くする
//     vals[pairs[1].1] = 1.0;
//     vals[pairs[2].1] = 1.0;

//     // C = R * S * R^T
//     let regularized_cov = rot * Matrix3::from_diagonal(&vals) * rot.transpose();
//     regularized_cov
// }

pub fn build_gicp_voxel_map(
    points: &[Point3<f32>],
    gicp_voxel_size: f32,
    max_points_per_voxel: usize,
    min_points_per_voxel: usize,
) -> VoxelMap {
    let mut voxel_map = VoxelMap::default();

    for &p in points {
        let key = voxel_key(&p, gicp_voxel_size);
        let cell = voxel_map.entry(key).or_insert_with(VoxelCell::new);
        cell.push_point(p, 0, max_points_per_voxel);
    }

    // Mean (parallel)
    voxel_map.par_iter_mut().for_each(|(_, cell)| {
        cell.mean = cell.compute_average();
    });

    // Covariance: 近傍voxelの平均を収集して共分散を一括並列計算
    // let updates: Vec<(VoxelKey, Matrix3<f32>, Matrix3<f32>)> = voxel_map
    //     .par_iter()
    //     .filter_map(|(key, cell)| {
    //         let neighbor_means: Vec<Point3<f32>> = neighbor_keys(key)
    //             .into_iter()
    //             .filter_map(|nk| voxel_map.get(&nk))
    //             .map(|c| c.mean)
    //             .collect();
    //         if neighbor_means.len() < min_points_per_voxel {
    //             return None;
    //         }
    //         let raw_cov = compute_raw_covariance_from_points(&neighbor_means, &cell.mean)?;
    //         let cov = regularize_gicp_covariance(raw_cov);
    //         Some((*key, raw_cov, cov))
    //     })
    //     .collect();

    // for (key, raw_cov, cov) in updates {
    //     if let Some(cell) = voxel_map.get_mut(&key) {
    //         cell.raw_covariance = raw_cov;
    //         cell.covariance = cov;
    //         cell.valid = true;
    //     }
    // }

    voxel_map
}

// ---------------------------------------------------------------------------
// LocalMap: フレームスタンプ付きスライディングウィンドウ管理
// ---------------------------------------------------------------------------

pub struct LocalMapConfig {
    pub voxel_size: f32,
    pub max_points_per_voxel: usize,
    pub min_points_per_voxel: usize,
    /// フレーム数ベースの追い出し上限。
    pub max_frames: usize,
    /// 距離ベースの追い出し上限 [m]。
    pub max_distance: f32,
}

struct FrameEntry {
    frame_id: u64,
    origin: Point3<f32>,
    dirty_keys: FxHashSet<VoxelKey>,
}

pub struct LocalMap {
    pub voxel_map: VoxelMap,
    frame_index: VecDeque<FrameEntry>,
    config: LocalMapConfig,
    next_frame_id: u64,
}

impl LocalMap {
    pub fn new(config: LocalMapConfig) -> Self {
        Self {
            voxel_map: VoxelMap::default(),
            frame_index: VecDeque::new(),
            config,
            next_frame_id: 0,
        }
    }

    pub fn get_all_points(&self) -> Vec<Point3<f32>> {
        self.voxel_map
            .values()
            .flat_map(|cell| cell.points.iter().map(|(p, _)| *p))
            .collect()
    }

    /// ワールド座標系に変換済みの点群を 1 フレームとして挿入する。
    /// `origin` は当該フレームのセンサー原点（距離ベース追い出しに使用）。
    pub fn insert_frame(&mut self, points: &[Point3<f32>], origin: Point3<f32>) {
        let frame_id = self.next_frame_id;
        self.next_frame_id += 1;

        let mut dirty_keys = FxHashSet::default();

        for &p in points {
            let key = voxel_key(&p, self.config.voxel_size);
            let cell = self.voxel_map.entry(key).or_insert_with(VoxelCell::new);
            if cell.push_point(p, frame_id, self.config.max_points_per_voxel) {
                dirty_keys.insert(key);
            }
        }

        for &key in &dirty_keys {
            if let Some(cell) = self.voxel_map.get_mut(&key) {
                cell.mean = cell.compute_average();
                cell.valid = cell.points.len() >= self.config.min_points_per_voxel;
            }
        }

        self.frame_index.push_back(FrameEntry {
            frame_id,
            origin,
            dirty_keys,
        });

        self.evict_if_needed();
    }

    pub fn frame_count(&self) -> usize {
        self.frame_index.len()
    }

    /// max_frames と max_distance の OR 条件で古いフレームを追い出す。
    fn evict_if_needed(&mut self) {
        loop {
            let should_evict = match self.frame_index.front() {
                None => break,
                Some(front) => {
                    if self.frame_index.len() > self.config.max_frames {
                        true
                    } else if let Some(back) = self.frame_index.back() {
                        let dx = back.origin.x - front.origin.x;
                        let dy = back.origin.y - front.origin.y;
                        let dz = back.origin.z - front.origin.z;
                        dx * dx + dy * dy + dz * dz
                            > self.config.max_distance * self.config.max_distance
                    } else {
                        false
                    }
                }
            };
            if !should_evict {
                break;
            }
            self.evict_oldest();
        }
    }

    /// 最古フレームの点を voxel_map から除去し、空になった voxel を削除する。
    fn evict_oldest(&mut self) {
        let Some(entry) = self.frame_index.pop_front() else {
            return;
        };

        let mut to_remove = Vec::new();

        for &key in &entry.dirty_keys {
            if let Some(cell) = self.voxel_map.get_mut(&key) {
                cell.points.retain(|(_, fid)| *fid != entry.frame_id);
                if cell.points.is_empty() {
                    to_remove.push(key);
                } else {
                    cell.mean = cell.compute_average();
                    cell.valid = cell.points.len() >= self.config.min_points_per_voxel;
                }
            }
        }

        for key in to_remove {
            self.voxel_map.remove(&key);
        }
    }

    /// `center` から `radius` [m] 以内の全点を収集して返す。
    ///
    /// voxel の AABB で候補 voxel を O(radius³/voxel_size³) に絞り込んだ後、
    /// 各点について正確な距離判定を行う。
    pub fn query_points_within_radius(
        &self,
        center: &Point3<f32>,
        radius: f32,
    ) -> Vec<Point3<f32>> {
        let vs = self.config.voxel_size;
        let r_sq = radius * radius;

        // 半径をカバーする voxel 範囲を整数グリッドで計算
        let half = (radius / vs).ceil() as i32;
        let cx = (center.x / vs).floor() as i32;
        let cy = (center.y / vs).floor() as i32;
        let cz = (center.z / vs).floor() as i32;

        let mut result = Vec::new();

        for ix in (cx - half)..=(cx + half) {
            for iy in (cy - half)..=(cy + half) {
                for iz in (cz - half)..=(cz + half) {
                    let key = VoxelKey { ix, iy, iz };
                    let Some(cell) = self.voxel_map.get(&key) else {
                        continue;
                    };
                    for (p, _) in &cell.points {
                        let dx = p.x - center.x;
                        let dy = p.y - center.y;
                        let dz = p.z - center.z;
                        if dx * dx + dy * dy + dz * dz <= r_sq {
                            result.push(*p);
                        }
                    }
                }
            }
        }

        result
    }
}
