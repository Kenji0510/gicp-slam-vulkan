use nalgebra::Point3;
use pcd_rs::{PcdDeserialize, PcdSerialize};
use serde::{Deserialize, Serialize};

use crate::{
    gpu_copy::GpuTransferDataContext, gpu_covariances::CovarianceGpuContext,
    gpu_gicp::GicpGpuContext, gpu_search_neighbor::SearchGpuContext,
    gpu_transform::TransformGpuContext, gpu_voxel::VoxelGpuContext,
};

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct LoadIMU {
    pub timestamp: u64,
    pub angular_velocity: [f32; 3],
    pub linear_acceleration: [f32; 3],
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct IMU {
    pub timestamp: f64,
    pub angular_velocity: [f32; 3],
    pub linear_acceleration: [f32; 3],
}

#[derive(Debug, Clone, PcdDeserialize, PcdSerialize)]
pub struct PointXYZ {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

#[derive(Debug, Clone, PcdDeserialize, PcdSerialize)]
pub struct PointXYZIT {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub intensity: f32,
    pub timestamp: f64,
}

#[derive(Debug, Clone, PcdDeserialize, PcdSerialize)]
pub struct PointXYZCov {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub cov_xx: f32,
    pub cov_xy: f32,
    pub cov_xz: f32,
    pub cov_yy: f32,
    pub cov_yz: f32,
    pub cov_zz: f32,
}

#[derive(Debug, Clone)]
pub struct FrameData {
    pub points: Vec<Point3<f32>>,
    pub covariances: Vec<[[f32; 3]; 3]>,
}

#[derive(Debug, Clone, PcdDeserialize, PcdSerialize)]
pub struct PointXYZNormal {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub normal_x: f32,
    pub normal_y: f32,
    pub normal_z: f32,
}

pub struct GPUContext {
    pub copy_source_gpu_context: GpuTransferDataContext,
    pub copy_target_gpu_context: GpuTransferDataContext,
    pub source_voxel_gpu_context: VoxelGpuContext,
    pub target_voxel_gpu_context: VoxelGpuContext,
    pub source_covariances_gpu_context: CovarianceGpuContext,
    pub target_covariances_gpu_context: CovarianceGpuContext,
    pub transform_gpu_context: TransformGpuContext,
    pub search_neighbor_gpu_context: SearchGpuContext,
    pub gicp_gpu_context: GicpGpuContext,
}

#[derive(Debug, Clone, PcdDeserialize, PcdSerialize)]
pub struct PointXYZShape {
    pub x: f32,
    pub y: f32,
    pub z: f32,

    pub normal_x: f32,
    pub normal_y: f32,
    pub normal_z: f32,

    pub linearity: f32,
    pub planarity: f32,
    pub scattering: f32,
}
