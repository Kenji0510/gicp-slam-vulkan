use std::collections::VecDeque;

use nalgebra::{Matrix3, Point3, SymmetricEigen, Vector3};
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
    /// タプルの 2 要素目はフレーム ID。
    pub points: Vec<(Point3<f32>, u64)>,

    /// 座標の累積和（インクリメンタルな平均計算用）。
    sum: Vector3<f32>,

    /// Gaussianの平均 μ。
    pub mean: Point3<f32>,

    /// Gaussianとしてregistrationに使えるだけの点数と数値安定性があるか。
    pub valid: bool,

    pub voxel_key: VoxelKey,

    pub covariance: Matrix3<f32>,
    pub normal: Vector3<f32>,
    pub linearity: f32,
    pub planarity: f32,
    pub scattering: f32,
    pub surface_valid: bool,
}

impl VoxelCell {
    pub fn new() -> Self {
        Self {
            points: Vec::new(),
            sum: Vector3::zeros(),
            mean: Point3::new(0.0, 0.0, 0.0),
            valid: false,
            voxel_key: VoxelKey {
                ix: 0,
                iy: 0,
                iz: 0,
            },
            covariance: Matrix3::identity(),
            normal: Vector3::z_axis().into_inner(),
            linearity: 0.0,
            planarity: 0.0,
            scattering: 1.0,
            surface_valid: false,
        }
    }

    pub fn observed_frame_count(&self) -> usize {
        let mut frames = FxHashSet::default();

        for (_, frame_id) in &self.points {
            frames.insert(*frame_id);
        }

        frames.len()
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
        self.sum += p.coords;
        self.points.push((p, frame_id));
        // O(1) でインクリメンタルに mean を更新
        self.mean = Point3::from(self.sum / self.points.len() as f32);
        true
    }

    /// 指定フレームの点を除去し、sum と mean を O(1) で更新する。
    /// 戻り値: 除去後に点が残っているか
    fn remove_frame_points(&mut self, frame_id: u64, min_points: usize) -> bool {
        let mut removed_sum = Vector3::zeros();
        self.points.retain(|(p, fid)| {
            if *fid == frame_id {
                removed_sum += p.coords;
                false
            } else {
                true
            }
        });
        self.sum -= removed_sum;
        let n = self.points.len();
        if n == 0 {
            self.mean = Point3::new(0.0, 0.0, 0.0);
            self.valid = false;
        } else {
            self.mean = Point3::from(self.sum / n as f32);
            self.valid = n >= min_points;
        }
        n > 0
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

fn neighbor_keys_with_range(key: &VoxelKey, range: i32) -> Vec<VoxelKey> {
    let side = 2 * range + 1;
    let mut neighbors = Vec::with_capacity((side * side * side) as usize);

    for dx in -range..=range {
        for dy in -range..=range {
            for dz in -range..=range {
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

    // Mean は push_point 内で O(1) 更新済みのため、事後計算は不要。

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

#[derive(Debug, Clone, Copy)]
pub struct SurfaceFeature {
    pub normal: Vector3<f32>,
    pub linearity: f32,
    pub planarity: f32,
    pub scattering: f32,
}

// ---------------------------------------------------------------------------
// LocalMap: フレームスタンプ付きスライディングウィンドウ管理
// ---------------------------------------------------------------------------

pub struct LocalMapConfig {
    /// ハッシュグリッドのセルサイズ [m]。
    /// query_points_within_radius のハッシュルックアップ数 = (2*ceil(radius/index_voxel_size)+1)³ を決定する。
    /// データの精度（downsample_voxel_size）とは独立に設定できる。
    /// 大きいほどクエリが速く、小さいほどセルあたりの点数が減る。
    pub index_voxel_size: f32,
    pub max_points_per_voxel: usize,
    pub min_points_per_voxel: usize,

    pub min_observed_frames_per_voxel: usize,
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

#[derive(Debug, Clone)]
struct SurfaceStats {
    covariance: Matrix3<f32>,
    normal: Vector3<f32>,
    linearity: f32,
    planarity: f32,
    scattering: f32,
    surface_valid: bool,
}

#[derive(Debug, Clone)]
enum SurfaceUpdate {
    Valid { key: VoxelKey, stats: SurfaceStats },
    Invalid { key: VoxelKey },
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

    pub fn surface_feature_at(&self, p: &Point3<f32>) -> Option<SurfaceFeature> {
        let key = voxel_key(p, self.config.index_voxel_size);

        let mut best_dist_sq = f32::INFINITY;
        let mut best_feature = None;

        for nk in neighbor_keys_with_range(&key, 2) {
            let Some(cell) = self.voxel_map.get(&nk) else {
                continue;
            };

            if !cell.surface_valid {
                continue;
            }

            let d = p.coords - cell.mean.coords;
            let dist_sq = d.norm_squared();

            if dist_sq < best_dist_sq {
                best_dist_sq = dist_sq;
                best_feature = Some(SurfaceFeature {
                    normal: cell.normal,
                    linearity: cell.linearity,
                    planarity: cell.planarity,
                    scattering: cell.scattering,
                });
            }
        }

        best_feature
    }

    pub fn get_all_points(&self) -> Vec<Point3<f32>> {
        self.voxel_map
            .values()
            .flat_map(|cell| cell.points.iter().map(|(p, _)| *p))
            .collect()
    }

    /// ワールド座標系に変換済みの点群を 1 フレームとして挿入する。
    ///
    /// 注意: この関数は normal gate を **内部では実行しない**。
    /// main 側で `surface_insert_mask()` 済みの点群を渡す前提にすることで、
    /// normal gate の二重実行を避ける。
    pub fn insert_frame(&mut self, points: &[Point3<f32>], origin: Point3<f32>) {
        self.insert_frame_no_gate(points, origin);
    }

    /// normal gate も LocalMap 側で実行したい場合の互換用関数。
    /// 通常の SLAM ループでは、mask を一度だけ作って `insert_frame()` に渡す方が速い。
    pub fn insert_frame_with_surface_gate(&mut self, points: &[Point3<f32>], origin: Point3<f32>) {
        let filtered_points: Vec<Point3<f32>> = points
            .par_iter()
            .copied()
            .filter(|p| self.should_accept_by_normal_gate(p))
            .collect();

        self.insert_frame_no_gate(&filtered_points, origin);
    }

    fn insert_frame_no_gate(&mut self, points: &[Point3<f32>], origin: Point3<f32>) {
        let frame_id = self.next_frame_id;
        self.next_frame_id += 1;

        let mut dirty_keys = FxHashSet::default();

        // HashMap への挿入は排他 mutable が必要なので、ここは逐次で安全に行う。
        // 重い normal gate と surface 統計再計算は別途 Rayon で並列化する。
        for &p in points {
            let key = voxel_key(&p, self.config.index_voxel_size);
            let cell = self.voxel_map.entry(key).or_insert_with(VoxelCell::new);

            if cell.push_point(p, frame_id, self.config.max_points_per_voxel) {
                dirty_keys.insert(key);

                // 周辺cellの共分散も変わるので近傍もdirtyにする
                for nk in neighbor_keys_with_range(&key, 2) {
                    dirty_keys.insert(nk);
                }
            }
        }

        self.refresh_valid_and_surface_stats_parallel(&dirty_keys);

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
        let mut recompute_keys = FxHashSet::default();

        for &key in &entry.dirty_keys {
            recompute_keys.insert(key);
            for nk in neighbor_keys_with_range(&key, 2) {
                recompute_keys.insert(nk);
            }

            if let Some(cell) = self.voxel_map.get_mut(&key) {
                let has_points =
                    cell.remove_frame_points(entry.frame_id, self.config.min_points_per_voxel);
                if !has_points {
                    to_remove.push(key);
                }
            }
        }

        for key in to_remove {
            self.voxel_map.remove(&key);
        }

        // 古いフレームの除去後、点数・観測フレーム数・周辺surface統計を更新する。
        // surface統計の再計算は Rayon で並列化する。
        self.refresh_valid_and_surface_stats_parallel(&recompute_keys);
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
        let vs = self.config.index_voxel_size;
        let r_sq = radius * radius;

        // 半径をカバーする voxel 範囲を整数グリッドで計算
        let half = (radius / vs).ceil() as i32;
        let cx = (center.x / vs).floor() as i32;
        let cy = (center.y / vs).floor() as i32;
        let cz = (center.z / vs).floor() as i32;

        let mut keys = Vec::new();
        for ix in (cx - half)..=(cx + half) {
            for iy in (cy - half)..=(cy + half) {
                for iz in (cz - half)..=(cz + half) {
                    keys.push(VoxelKey { ix, iy, iz });
                }
            }
        }

        keys.par_iter()
            .flat_map_iter(|key| {
                self.voxel_map
                    .get(key)
                    .into_iter()
                    .flat_map(|cell| cell.points.iter())
                    .filter_map(move |(p, _)| {
                        let dx = p.x - center.x;
                        let dy = p.y - center.y;
                        let dz = p.z - center.z;
                        if dx * dx + dy * dy + dz * dz <= r_sq {
                            Some(*p)
                        } else {
                            None
                        }
                    })
            })
            .collect()
    }

    pub fn query_stable_points_within_radius(
        &self,
        center: &Point3<f32>,
        radius: f32,
    ) -> Vec<Point3<f32>> {
        let vs = self.config.index_voxel_size;
        let r_sq = radius * radius;

        let half = (radius / vs).ceil() as i32;
        let cx = (center.x / vs).floor() as i32;
        let cy = (center.y / vs).floor() as i32;
        let cz = (center.z / vs).floor() as i32;

        let mut keys = Vec::new();
        for ix in (cx - half)..=(cx + half) {
            for iy in (cy - half)..=(cy + half) {
                for iz in (cz - half)..=(cz + half) {
                    keys.push(VoxelKey { ix, iy, iz });
                }
            }
        }

        keys.par_iter()
            .flat_map_iter(|key| {
                self.voxel_map
                    .get(key)
                    .filter(|cell| cell.valid)
                    .into_iter()
                    .flat_map(|cell| cell.points.iter())
                    .filter_map(move |(p, _)| {
                        let dx = p.x - center.x;
                        let dy = p.y - center.y;
                        let dz = p.z - center.z;
                        if dx * dx + dy * dy + dz * dz <= r_sq {
                            Some(*p)
                        } else {
                            None
                        }
                    })
            })
            .collect()
    }

    fn refresh_valid_and_surface_stats_parallel(&mut self, keys: &FxHashSet<VoxelKey>) {
        if keys.is_empty() {
            return;
        }

        // まず valid だけ逐次更新する。
        // observed_frame_count() は cell.points を見るため、HashMap のmutable更新中に並列化しない。
        for &key in keys {
            if let Some(cell) = self.voxel_map.get_mut(&key) {
                cell.valid = cell.point_count() >= self.config.min_points_per_voxel
                    && cell.observed_frame_count() >= self.config.min_observed_frames_per_voxel;
            }
        }

        // surface統計は immutable read だけで計算できるため Rayon で並列化する。
        // 結果の書き戻しだけ最後に逐次で行う。
        let key_vec: Vec<VoxelKey> = keys.iter().copied().collect();
        let updates: Vec<SurfaceUpdate> = key_vec
            .par_iter()
            .map(|&key| self.compute_surface_update_for_key(key))
            .collect();

        for update in updates {
            self.apply_surface_update(update);
        }
    }

    fn compute_surface_update_for_key(&self, key: VoxelKey) -> SurfaceUpdate {
        let neighbor_means: Vec<Point3<f32>> = neighbor_keys_with_range(&key, 2)
            .into_iter()
            .filter_map(|nk| self.voxel_map.get(&nk))
            .filter(|c| c.point_count() >= self.config.min_points_per_voxel)
            .map(|c| c.mean)
            .collect();

        let Some(cov) = compute_covariance_from_points(&neighbor_means) else {
            return SurfaceUpdate::Invalid { key };
        };

        let Some((normal, linearity, planarity, scattering)) = compute_shape_from_covariance(cov)
        else {
            return SurfaceUpdate::Invalid { key };
        };

        let Some(cell) = self.voxel_map.get(&key) else {
            return SurfaceUpdate::Invalid { key };
        };

        let surface_valid = cell.valid
            && cell.observed_frame_count() >= self.config.min_observed_frames_per_voxel
            && planarity > 0.35
            && scattering < 0.25;

        SurfaceUpdate::Valid {
            key,
            stats: SurfaceStats {
                covariance: cov,
                normal,
                linearity,
                planarity,
                scattering,
                surface_valid,
            },
        }
    }

    fn apply_surface_update(&mut self, update: SurfaceUpdate) {
        match update {
            SurfaceUpdate::Invalid { key } => {
                if let Some(cell) = self.voxel_map.get_mut(&key) {
                    cell.surface_valid = false;
                }
            }
            SurfaceUpdate::Valid { key, stats } => {
                if let Some(cell) = self.voxel_map.get_mut(&key) {
                    cell.covariance = stats.covariance;
                    cell.normal = stats.normal;
                    cell.linearity = stats.linearity;
                    cell.planarity = stats.planarity;
                    cell.scattering = stats.scattering;
                    cell.surface_valid = stats.surface_valid;
                }
            }
        }
    }

    fn should_accept_by_normal_gate(&self, p: &Point3<f32>) -> bool {
        let key = voxel_key(p, self.config.index_voxel_size);

        let mut found_surface = false;
        let mut best_normal_dist = f32::INFINITY;

        for nk in neighbor_keys_with_range(&key, 2) {
            let Some(cell) = self.voxel_map.get(&nk) else {
                continue;
            };

            if !cell.surface_valid {
                continue;
            }

            let diff = p.coords - cell.mean.coords;
            let signed_normal_dist = diff.dot(&cell.normal);
            let normal_dist = signed_normal_dist.abs();

            let tangent_vec = diff - cell.normal * signed_normal_dist;
            let tangent_dist = tangent_vec.norm();

            // 同じ面パッチ周辺だけを見る。
            // 離れた別の平面のnormalで誤って弾かないための制限。
            if tangent_dist > self.config.index_voxel_size * 1.5 {
                continue;
            }

            found_surface = true;
            best_normal_dist = best_normal_dist.min(normal_dist);
        }

        // 近くに安定平面がなければ、新規構造候補として許可。
        if !found_surface {
            return true;
        }

        // 近くに安定平面があるなら、法線方向に離れた点は追加しない。
        best_normal_dist < 0.01
    }

    /// 既存のstable surfaceに対して、各点をmapへ追加してよいか判定するmaskを返す。
    ///
    /// 入力点群は LocalMap と同じ座標系、つまり現在の使い方では world 座標系を想定する。
    /// `true` ならmap保存用点群へ残し、`false` なら壁などの法線方向に浮いた点として除外する。
    pub fn surface_insert_mask(&self, points_world: &[Point3<f32>]) -> Vec<bool> {
        points_world
            .par_iter()
            .map(|p| self.should_accept_by_normal_gate(p))
            .collect()
    }

    /// `surface_insert_mask` の簡易版。
    /// world座標点群から、normal gateを通過した点だけを返す。
    pub fn filter_points_by_surface_gate(&self, points_world: &[Point3<f32>]) -> Vec<Point3<f32>> {
        points_world
            .par_iter()
            .copied()
            .filter(|p| self.should_accept_by_normal_gate(p))
            .collect()
    }
}

fn compute_covariance_from_points(points: &[Point3<f32>]) -> Option<Matrix3<f32>> {
    if points.len() < 5 {
        return None;
    }

    let mut mean = Vector3::zeros();
    for p in points {
        mean += p.coords;
    }
    mean /= points.len() as f32;

    let mut cov = Matrix3::<f32>::zeros();
    for p in points {
        let d = p.coords - mean;
        cov += d * d.transpose();
    }
    cov /= points.len() as f32;

    if !cov.iter().all(|v| v.is_finite()) {
        return None;
    }

    Some(cov)
}

fn compute_shape_from_covariance(cov: Matrix3<f32>) -> Option<(Vector3<f32>, f32, f32, f32)> {
    let eig = SymmetricEigen::new(cov);

    let mut ids = [0usize, 1, 2];
    ids.sort_by(|&a, &b| {
        eig.eigenvalues[a]
            .partial_cmp(&eig.eigenvalues[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let l0 = eig.eigenvalues[ids[0]].max(1.0e-6);
    let l1 = eig.eigenvalues[ids[1]].max(1.0e-6);
    let l2 = eig.eigenvalues[ids[2]].max(1.0e-6);

    if !l0.is_finite() || !l1.is_finite() || !l2.is_finite() || l2 <= 1.0e-6 {
        return None;
    }

    let normal_col = eig.eigenvectors.column(ids[0]);
    let normal = Vector3::new(normal_col[0], normal_col[1], normal_col[2]);

    if !normal.iter().all(|v| v.is_finite()) || normal.norm_squared() < 1.0e-8 {
        return None;
    }

    let normal = normal.normalize();

    let linearity = (l2 - l1) / l2;
    let planarity = (l1 - l0) / l2;
    let scattering = l0 / l2;

    if !linearity.is_finite() || !planarity.is_finite() || !scattering.is_finite() {
        return None;
    }

    Some((normal, linearity, planarity, scattering))
}
