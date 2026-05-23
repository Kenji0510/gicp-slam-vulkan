use anyhow::{Context, Result};
use foldhash::{HashMap, HashMapExt};
use log::debug;
use std::{sync::Arc, time::Instant};

use vulkano::{
    buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer},
    command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage, CopyBufferInfo},
    descriptor_set::{DescriptorSet, layout::DescriptorSetLayout},
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter},
    pipeline::{
        ComputePipeline, Pipeline, PipelineLayout, PipelineShaderStageCreateInfo,
        compute::ComputePipelineCreateInfo, layout::PipelineDescriptorSetLayoutCreateInfo,
    },
    shader::{ShaderModule, ShaderModuleCreateInfo, SpecializationConstant},
    sync::{self, GpuFuture},
};

use crate::{
    gpu_knn_search::KnnSearchGpuContext, gpu_voxel::VoxelGpuContext, init_gpu::VulkanContext,
    types::PointXYZNormal,
};

const K_NEIGHBORS: usize = 8;

#[derive(bytemuck::Pod, bytemuck::Zeroable, Clone, Copy)]
#[repr(C)]
pub struct CovarianceParams {
    pub num_points: u32,
    pub table_size: i32,
    pub voxel_size: f32,
    _pad: i32,
}

pub struct CovarianceGpuContext {
    vulkan_context: VulkanContext,

    compute_pipeline_search: Arc<ComputePipeline>,
    pipeline_layout_search: Arc<PipelineLayout>,
    descriptor_set_layout_search: Arc<DescriptorSetLayout>,

    pub d_buf_covariances: Option<Subbuffer<[f32]>>,

    pub staging_buf_covariances: Option<Subbuffer<[f32]>>,

    pub current_capacity_pts: usize,
}

impl CovarianceGpuContext {
    pub fn new(vulkan_context: VulkanContext) -> Result<Self> {
        let shader_normals = unsafe {
            ShaderModule::new(
                vulkan_context.device.clone(),
                ShaderModuleCreateInfo::new(&bytemuck::pod_collect_to_vec::<u8, u32>(
                    include_bytes!(concat!(env!("OUT_DIR"), "/shaders/covariance.spv")),
                )),
            )
        }
        .context("Failed to load normals shader")?;

        let mut spec = HashMap::new();
        spec.insert(0u32, SpecializationConstant::U32(K_NEIGHBORS as u32));

        let cs_normals = shader_normals
            .specialize(spec)
            .context("Failed to specialize normals shader with constants")?
            .entry_point("main")
            .context("Failed to find entry point in normals shader")?;

        let stage_normals = PipelineShaderStageCreateInfo::new(cs_normals);

        let layout_normals = PipelineLayout::new(
            vulkan_context.device.clone(),
            PipelineDescriptorSetLayoutCreateInfo::from_stages([&stage_normals])
                .into_pipeline_layout_create_info(vulkan_context.device.clone())
                .context("Failed to create pipeline layout")?,
        )
        .context("Failed to create pipeline layout")?;

        let compute_pipeline_normals = ComputePipeline::new(
            vulkan_context.device.clone(),
            None,
            ComputePipelineCreateInfo::stage_layout(stage_normals, layout_normals),
        )
        .context("Failed to create compute pipeline normals")?;

        let pipeline_layout_normals = compute_pipeline_normals.layout();

        let descriptor_set_layout_normals = pipeline_layout_normals
            .set_layouts()
            .get(0)
            .context("Failed to get descriptor set layout for normals")?;

        Ok(Self {
            vulkan_context: vulkan_context.clone(),
            compute_pipeline_search: compute_pipeline_normals.clone(),
            pipeline_layout_search: pipeline_layout_normals.clone(),
            descriptor_set_layout_search: descriptor_set_layout_normals.clone(),
            d_buf_covariances: None,
            staging_buf_covariances: None,
            current_capacity_pts: 0,
        })
    }

    fn compute_covariances_impl(
        &mut self,
        target_voxel_gpu_context: &VoxelGpuContext,
        // knn_search_gpu_context: &KnnSearchGpuContext,
        read_back: bool,
    ) -> Result<Vec<[f32; 9]>> {
        let device = &self.vulkan_context.device;
        let queue = &self.vulkan_context.queue;
        let memory_allocator = &self.vulkan_context.memory_allocator;
        let descriptor_set_allocator = &self.vulkan_context.descriptor_set_allocator;
        let command_buffer_allocator = &self.vulkan_context.command_buffer_allocator;
        let pipeline_layout = &self.pipeline_layout_search;
        let compute_pipeline = &self.compute_pipeline_search;

        let target_pts_num = target_voxel_gpu_context.h_downsampled_pts_num;
        let covariance_params = CovarianceParams {
            num_points: target_pts_num as u32,
            table_size: target_voxel_gpu_context.table_size,
            voxel_size: target_voxel_gpu_context.voxel_size,
            _pad: 0,
        };

        if self.current_capacity_pts < target_pts_num {
            debug!("Reallocating buffers for {} points", target_pts_num);

            let new_capacity = (target_pts_num as f32 * 1.5) as usize;
            self.current_capacity_pts = new_capacity;

            self.d_buf_covariances = Some(Buffer::new_slice::<f32>(
                memory_allocator.clone(),
                BufferCreateInfo {
                    usage: BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_SRC,
                    ..Default::default()
                },
                AllocationCreateInfo {
                    memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                    ..Default::default()
                },
                (new_capacity * 9) as u64,
            )?);

            self.staging_buf_covariances = Some(Buffer::new_slice::<f32>(
                memory_allocator.clone(),
                BufferCreateInfo {
                    usage: BufferUsage::TRANSFER_DST,
                    ..Default::default()
                },
                AllocationCreateInfo {
                    memory_type_filter: MemoryTypeFilter::PREFER_HOST
                        | MemoryTypeFilter::HOST_RANDOM_ACCESS,
                    ..Default::default()
                },
                (new_capacity * 9) as u64,
            )?);
        }

        let descriptor_set = DescriptorSet::new(
            descriptor_set_allocator.clone(),
            self.descriptor_set_layout_search.clone(),
            [
                vulkano::descriptor_set::WriteDescriptorSet::buffer(
                    0,
                    target_voxel_gpu_context
                        .d_buf_out_pts
                        .as_ref()
                        .context("Failed to get output points buffer")?
                        .clone(),
                ),
                vulkano::descriptor_set::WriteDescriptorSet::buffer(
                    1,
                    target_voxel_gpu_context
                        .d_buf_keys
                        .as_ref()
                        .context("Failed to get keys buffer")?
                        .clone(),
                ),
                vulkano::descriptor_set::WriteDescriptorSet::buffer(
                    2,
                    target_voxel_gpu_context
                        .d_buf_centroids
                        .as_ref()
                        .context("Failed to get centroids buffer")?
                        .clone(),
                ),
                vulkano::descriptor_set::WriteDescriptorSet::buffer(
                    3,
                    target_voxel_gpu_context
                        .d_buf_counts
                        .as_ref()
                        .context("Failed to get counts buffer")?
                        .clone(),
                ),
                vulkano::descriptor_set::WriteDescriptorSet::buffer(
                    4,
                    self.d_buf_covariances
                        .as_ref()
                        .context("Failed to get covariances buffer")?
                        .clone(),
                ),
            ],
            [],
        )
        .context("Failed to create descriptor set for compute covariances")?;

        let mut command_buffer_builder = AutoCommandBufferBuilder::primary(
            command_buffer_allocator.clone(),
            queue.queue_family_index().clone(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .context("Failed to create command buffer builder")?;

        const LOCAL_SIZE: u32 = 256;
        let group_count_x = (target_pts_num as u32 + LOCAL_SIZE - 1) / LOCAL_SIZE;
        let work_group_count = [group_count_x, 1, 1];

        unsafe {
            command_buffer_builder
                .bind_pipeline_compute(compute_pipeline.clone())
                .context("Failed to bind compute pipeline")?
                .push_constants(pipeline_layout.clone(), 0, covariance_params)
                .context("Failed to push constants")?
                .bind_descriptor_sets(
                    vulkano::pipeline::PipelineBindPoint::Compute,
                    pipeline_layout.clone(),
                    0,
                    descriptor_set.clone(),
                )
                .context("Failed to bind descriptor sets")?
                .dispatch(work_group_count)
                .context("Failed to dispatch compute shader")?;
        }

        // <!--- Copy target covariances from GPU to staging buffer --->
        if read_back {
            let copy_output_covariances_src = self
                .d_buf_covariances
                .as_ref()
                .context("Failed to get output covariances buffer for copy")?
                .clone()
                .slice(0..(target_pts_num * 9) as u64);
            let copy_output_covariances_dst = self
                .staging_buf_covariances
                .as_ref()
                .context("Failed to get staging buffer for covariances copy")?
                .clone()
                .slice(0..(target_pts_num * 9) as u64);
            command_buffer_builder
                .copy_buffer(CopyBufferInfo::buffers(
                    copy_output_covariances_src,
                    copy_output_covariances_dst,
                ))
                .context("Failed to copy output covariances to staging buffer")?;
        }
        // <!--- Copy target covariances from GPU to staging buffer --->

        let command_buffer = command_buffer_builder.build()?;

        let compute_start_time = Instant::now();
        let future = sync::now(device.clone())
            .then_execute(queue.clone(), command_buffer)?
            .then_signal_fence_and_flush()?;

        future.wait(None)?;

        let compute_end_time = compute_start_time.elapsed();
        debug!(
            "Compute covariance shader execution time: {:?}",
            compute_end_time
        );

        // <!--- Copy results from staging buffer to CPU --->
        let output_covariances = if read_back {
            let covariances_content = self
                .staging_buf_covariances
                .as_ref()
                .context("Failed to get staging buffer for covariances read")?
                .read()?;
            covariances_content
                .chunks_exact(9)
                .take(target_pts_num as usize)
                .map(|c| [c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7], c[8]])
                .collect()
        } else {
            vec![]
        };
        // <!--- Copy results from staging buffer to CPU --->

        Ok(output_covariances)
    }

    pub fn compute_covariances(
        &mut self,
        target_voxel_gpu_context: &VoxelGpuContext,
    ) -> Result<Vec<[f32; 9]>> {
        self.compute_covariances_impl(target_voxel_gpu_context, true)
    }

    pub fn compute_covariances_gpu_only(
        &mut self,
        target_voxel_gpu_context: &VoxelGpuContext,
    ) -> Result<()> {
        self.compute_covariances_impl(target_voxel_gpu_context, false)?;
        Ok(())
    }
}

pub fn combine_pts_with_normals(
    pts: &Vec<[f32; 3]>,
    normals: &Vec<[f32; 3]>,
) -> Result<Vec<PointXYZNormal>> {
    let num_pts = pts.len();
    let mut pts_with_normals = Vec::with_capacity(num_pts);
    for i in 0..num_pts {
        pts_with_normals.push(PointXYZNormal {
            x: pts[i][0],
            y: pts[i][1],
            z: pts[i][2],
            normal_x: normals[i][0],
            normal_y: normals[i][1],
            normal_z: normals[i][2],
        });
    }
    Ok(pts_with_normals)
}
