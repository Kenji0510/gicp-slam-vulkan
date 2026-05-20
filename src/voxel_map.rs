use nalgebra::{Isometry3, Matrix3, Matrix4, Point3, SymmetricEigen, Vector3};
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
    pub points: Vec<Point3<f32>>,

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

    pub fn push_point(&mut self, p: Point3<f32>, max_points_per_gaussian: usize) -> bool {
        if self.points.len() >= max_points_per_gaussian {
            return false;
        }

        self.points.push(p);
        true
    }

    pub fn compute_average(&self) -> Point3<f32> {
        compute_mean_from_points(&self.points)
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

fn compute_mean_from_points(points: &[Point3<f32>]) -> Point3<f32> {
    let mut sum = Vector3::zeros();

    for p in points {
        sum += p.coords;
    }

    Point3::from(sum / points.len() as f32)
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
        cell.push_point(p, max_points_per_voxel);
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

pub fn recompute_all_covariance(voxel_map: &mut VoxelMap, min_points_per_gaussian: usize) {
    voxel_map.par_iter_mut().for_each(|(_, cell)| {
        cell.recompute_covariance(min_points_per_gaussian);
    });
}

pub fn recompute_gaussians_for_keys(
    voxel_map: &mut VoxelMap,
    keys: &[VoxelKey],
    min_points_per_gaussian: usize,
    min_variance: f32,
    max_variance: f32,
    information_regularization: f32,
) {
    let results: Vec<(VoxelKey, VoxelCell)> = keys
        .par_iter()
        .filter_map(|&key| {
            let mut cell = voxel_map.get(&key)?.clone();
            cell.recompute_covariance(min_points_per_gaussian);
            Some((key, cell))
        })
        .collect();

    for (key, cell) in results {
        if let Some(dst) = voxel_map.get_mut(&key) {
            *dst = cell;
        }
    }
}

/// VoxelMap に剛体変換を適用して新しい VoxelMap を返す。
/// - mean:       R * p + t
/// - covariance: R * Σ * R^T
/// - points:     R * p + t (各点)
/// voxel_key はtransform後の mean から再計算する。
pub fn transform_voxel_map(map: &VoxelMap, pose: &Matrix4<f64>, voxel_size: f32) -> VoxelMap {
    let rot = pose.fixed_view::<3, 3>(0, 0).into_owned().cast::<f32>();
    let trans: Vector3<f32> = pose.fixed_view::<3, 1>(0, 3).into_owned().cast::<f32>();

    map.par_iter()
        .map(|(_, cell)| {
            let new_mean = Point3::from(rot * cell.mean.coords + trans);
            let new_covariance = rot * cell.covariance * rot.transpose();
            let new_raw_covariance = rot * cell.raw_covariance * rot.transpose();
            let new_key = voxel_key(&new_mean, voxel_size);
            (
                new_key,
                VoxelCell {
                    points: Vec::new(),
                    mean: new_mean,
                    raw_covariance: new_raw_covariance,
                    covariance: new_covariance,
                    valid: cell.valid,
                    voxel_key: new_key,
                },
            )
        })
        .collect()
}

/// 変換済み点群を既存のGaussian voxel mapに追加する。
///
/// Gaussian版では「1 voxel = 1 Gaussian」なので、GICP版のように周辺3x3x3を
/// 再計算しない。変更されたvoxel自身だけを再計算する。
pub fn merge_points_into_gaussian_voxel_map(
    map: &mut VoxelMap,
    points: &[Point3<f32>],
    pose: &Matrix4<f64>,
    gicp_voxel_size: f32,
    max_points_per_voxel: usize,
    min_points_per_voxel: usize,
) {
    let rot = pose.fixed_view::<3, 3>(0, 0).into_owned().cast::<f32>();
    let trans: Vector3<f32> = pose.fixed_view::<3, 1>(0, 3).into_owned().cast::<f32>();

    let mut modified_keys = FxHashSet::<VoxelKey>::default();

    for p in points {
        let transformed = Point3::from(rot * p.coords + trans);
        let key = voxel_key(&transformed, gicp_voxel_size);
        let cell = map.entry(key).or_insert_with(VoxelCell::new);

        if cell.push_point(transformed, max_points_per_voxel) {
            modified_keys.insert(key);
        }
    }

    if modified_keys.is_empty() {
        return;
    }

    // Mean: 変更されたvoxelのみ再計算
    for key in &modified_keys {
        if let Some(cell) = map.get_mut(key) {
            cell.mean = cell.compute_average();
        }
    }

    // Covariance: 変更voxel + その隣接voxelのみ再計算
    // (あるvoxelのmeanが変わると、そのvoxelを隣接に持つ全voxelの共分散も変わる)
    let keys_to_update: FxHashSet<VoxelKey> = modified_keys
        .iter()
        .flat_map(|key| {
            let mut ks: Vec<VoxelKey> = neighbor_keys(key).into_iter().collect();
            ks.push(*key);
            ks
        })
        .filter(|key| map.contains_key(key))
        .collect();

    let keys_and_neighbors: Vec<(VoxelKey, Vec<Point3<f32>>, Point3<f32>)> = keys_to_update
        .into_iter()
        .filter_map(|key| {
            let cell = map.get(&key)?;
            let mean = cell.mean;
            let neighbor_means: Vec<Point3<f32>> = neighbor_keys(&key)
                .into_iter()
                .filter_map(|nk| map.get(&nk))
                .map(|c| c.mean)
                .collect();
            Some((key, neighbor_means, mean))
        })
        .collect();

    let updates: Vec<(VoxelKey, Matrix3<f32>, Matrix3<f32>)> = keys_and_neighbors
        .par_iter()
        .filter_map(|(key, neighbor_means, mean)| {
            if neighbor_means.len() < min_points_per_voxel {
                return None;
            }
            let raw_cov = compute_raw_covariance_from_points(neighbor_means, mean)?;
            let cov = regularize_gicp_covariance(raw_cov);
            Some((*key, raw_cov, cov))
        })
        .collect();

    for (key, raw_cov, cov) in updates {
        if let Some(cell) = map.get_mut(&key) {
            cell.raw_covariance = raw_cov;
            cell.covariance = cov;
            cell.valid = true;
        }
    }
}
