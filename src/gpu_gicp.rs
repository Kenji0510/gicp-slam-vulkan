use std::{sync::Arc, time::Instant};

use anyhow::{Context, Result};

use log::debug;
use nalgebra::{Matrix4, Matrix6, UnitQuaternion, Vector3, Vector6};
use vulkano::{
    buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer},
    command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage, CopyBufferInfo},
    descriptor_set::{DescriptorSet, layout::DescriptorSetLayout},
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter},
    pipeline::{
        ComputePipeline, Pipeline, PipelineLayout, PipelineShaderStageCreateInfo,
        compute::ComputePipelineCreateInfo, layout::PipelineDescriptorSetLayoutCreateInfo,
    },
    shader::{ShaderModule, ShaderModuleCreateInfo},
    sync::{self, GpuFuture},
};

use crate::{gpu_search_neighbor::SearchGpuContext, init_gpu::VulkanContext};

pub struct GicpStaticBuffers {
    pub d_source_pts: Subbuffer<[f32]>,
    pub d_source_covs: Subbuffer<[f32]>,
    pub d_target_pts: Subbuffer<[f32]>,
    pub d_target_covs: Subbuffer<[f32]>,
}

#[derive(bytemuck::Pod, bytemuck::Zeroable, Clone, Copy)]
#[repr(C)]
pub struct GicpParams {
    pub num_source: i32,
    pub num_target: i32,
    pub max_dist_sq: f32,
}

pub struct GicpGpuContext {
    vulkan_context: VulkanContext,

    compute_pipeline_search: Arc<ComputePipeline>,
    pipeline_layout_search: Arc<PipelineLayout>,
    descriptor_set_layout_search: Arc<DescriptorSetLayout>,

    // ワークグループごとの部分和バッファ (size = capacity_groups × 36/6)
    pub d_buf_partial_h: Option<Subbuffer<[f32]>>,
    pub d_buf_partial_b: Option<Subbuffer<[f32]>>,

    pub staging_buf_partial_h: Option<Subbuffer<[f32]>>,
    pub staging_buf_partial_b: Option<Subbuffer<[f32]>>,

    capacity_groups: usize,
}

impl GicpGpuContext {
    pub fn new(vulkan_context: VulkanContext) -> Result<Self> {
        let shader_icp = unsafe {
            ShaderModule::new(
                vulkan_context.device.clone(),
                ShaderModuleCreateInfo::new(&bytemuck::pod_collect_to_vec::<u8, u32>(
                    include_bytes!(concat!(env!("OUT_DIR"), "/shaders/gicp.spv")),
                )),
            )
        }
        .context("Failed to load ICP shader")?;

        let cs_icp = shader_icp
            .entry_point("main")
            .context("Failed to find entry point in ICP shader")?;

        let stage_icp = PipelineShaderStageCreateInfo::new(cs_icp);

        let layout_icp = PipelineLayout::new(
            vulkan_context.device.clone(),
            PipelineDescriptorSetLayoutCreateInfo::from_stages([&stage_icp])
                .into_pipeline_layout_create_info(vulkan_context.device.clone())
                .context("Failed to create pipeline layout")?,
        )
        .context("Failed to create pipeline layout")?;

        let compute_pipeline_icp = ComputePipeline::new(
            vulkan_context.device.clone(),
            None,
            ComputePipelineCreateInfo::stage_layout(stage_icp, layout_icp),
        )
        .context("Failed to create compute pipeline ICP")?;

        let pipeline_layout_icp = compute_pipeline_icp.layout();

        let descriptor_set_layout_icp = pipeline_layout_icp
            .set_layouts()
            .get(0)
            .context("Failed to get descriptor set layout for ICP")?;

        Ok(Self {
            vulkan_context: vulkan_context.clone(),
            compute_pipeline_search: compute_pipeline_icp.clone(),
            pipeline_layout_search: pipeline_layout_icp.clone(),
            descriptor_set_layout_search: descriptor_set_layout_icp.clone(),
            d_buf_partial_h: None,
            d_buf_partial_b: None,
            staging_buf_partial_h: None,
            staging_buf_partial_b: None,
            capacity_groups: 0,
        })
    }

    pub fn compute_gicp(
        &mut self,
        static_bufs: &GicpStaticBuffers,
        neighbor_search_ctx: &SearchGpuContext,
        source_pts_num: usize,
        target_pts_num: usize,
        max_dist_sq: f32,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        let device = &self.vulkan_context.device;
        let queue = &self.vulkan_context.queue;
        let memory_allocator = &self.vulkan_context.memory_allocator;
        let descriptor_set_allocator = &self.vulkan_context.descriptor_set_allocator;
        let command_buffer_allocator = &self.vulkan_context.command_buffer_allocator;
        let pipeline_layout = &self.pipeline_layout_search;
        let compute_pipeline = &self.compute_pipeline_search;

        let gicp_params = GicpParams {
            num_source: source_pts_num as i32,
            num_target: target_pts_num as i32,
            max_dist_sq,
        };

        const LOCAL_SIZE: usize = 64;
        let num_groups = (source_pts_num + LOCAL_SIZE - 1) / LOCAL_SIZE;

        // partial バッファが不足していれば再確保
        if num_groups > self.capacity_groups {
            let new_cap = (num_groups as f64 * 1.5) as usize + 1;
            debug!("Allocating partial H/b buffers for {} groups", new_cap);

            self.d_buf_partial_h = Some(Buffer::new_slice::<f32>(
                memory_allocator.clone(),
                BufferCreateInfo {
                    usage: BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_SRC,
                    ..Default::default()
                },
                AllocationCreateInfo {
                    memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                    ..Default::default()
                },
                (new_cap * 36) as u64,
            )?);

            self.d_buf_partial_b = Some(Buffer::new_slice::<f32>(
                memory_allocator.clone(),
                BufferCreateInfo {
                    usage: BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_SRC,
                    ..Default::default()
                },
                AllocationCreateInfo {
                    memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                    ..Default::default()
                },
                (new_cap * 6) as u64,
            )?);

            self.staging_buf_partial_h = Some(Buffer::new_slice::<f32>(
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
                (new_cap * 36) as u64,
            )?);

            self.staging_buf_partial_b = Some(Buffer::new_slice::<f32>(
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
                (new_cap * 6) as u64,
            )?);

            self.capacity_groups = new_cap;
        }

        let descriptor_set = DescriptorSet::new(
            descriptor_set_allocator.clone(),
            self.descriptor_set_layout_search.clone(),
            [
                vulkano::descriptor_set::WriteDescriptorSet::buffer(
                    0,
                    static_bufs.d_source_pts.clone(),
                ),
                vulkano::descriptor_set::WriteDescriptorSet::buffer(
                    1,
                    static_bufs.d_source_covs.clone(),
                ),
                vulkano::descriptor_set::WriteDescriptorSet::buffer(
                    2,
                    static_bufs.d_target_pts.clone(),
                ),
                vulkano::descriptor_set::WriteDescriptorSet::buffer(
                    3,
                    static_bufs.d_target_covs.clone(),
                ),
                vulkano::descriptor_set::WriteDescriptorSet::buffer(
                    4,
                    neighbor_search_ctx
                        .d_buf_indices
                        .as_ref()
                        .context("Failed to get neighbor indices buffer")?
                        .clone(),
                ),
                vulkano::descriptor_set::WriteDescriptorSet::buffer(
                    5,
                    neighbor_search_ctx
                        .d_buf_dists_sq
                        .as_ref()
                        .context("Failed to get neighbor dists buffer")?
                        .clone(),
                ),
                vulkano::descriptor_set::WriteDescriptorSet::buffer(
                    6,
                    self.d_buf_partial_h
                        .as_ref()
                        .context("Failed to get partial H buffer")?
                        .clone()
                        .slice(0..(num_groups * 36) as u64),
                ),
                vulkano::descriptor_set::WriteDescriptorSet::buffer(
                    7,
                    self.d_buf_partial_b
                        .as_ref()
                        .context("Failed to get partial b buffer")?
                        .clone()
                        .slice(0..(num_groups * 6) as u64),
                ),
            ],
            [],
        )
        .context("Failed to create descriptor set for search neighbor")?;

        let mut command_buffer_builder = AutoCommandBufferBuilder::primary(
            command_buffer_allocator.clone(),
            queue.queue_family_index().clone(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .context("Failed to create command buffer builder")?;

        // partial バッファは毎回上書きするため fill_buffer 不要
        let work_group_count = [num_groups as u32, 1, 1];

        unsafe {
            command_buffer_builder
                .bind_pipeline_compute(compute_pipeline.clone())
                .context("Failed to bind compute pipeline")?
                .push_constants(pipeline_layout.clone(), 0, gicp_params)
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

        // partial H/b を staging へコピー（num_groups 分のみ）
        command_buffer_builder
            .copy_buffer(CopyBufferInfo::buffers(
                self.d_buf_partial_h
                    .as_ref()
                    .context("Failed to get partial H buffer")?
                    .clone()
                    .slice(0..(num_groups * 36) as u64),
                self.staging_buf_partial_h
                    .as_ref()
                    .context("Failed to get staging H buffer")?
                    .clone()
                    .slice(0..(num_groups * 36) as u64),
            ))
            .context("Failed to copy partial H to staging")?;
        command_buffer_builder
            .copy_buffer(CopyBufferInfo::buffers(
                self.d_buf_partial_b
                    .as_ref()
                    .context("Failed to get partial b buffer")?
                    .clone()
                    .slice(0..(num_groups * 6) as u64),
                self.staging_buf_partial_b
                    .as_ref()
                    .context("Failed to get staging b buffer")?
                    .clone()
                    .slice(0..(num_groups * 6) as u64),
            ))
            .context("Failed to copy partial b to staging")?;

        let command_buffer = command_buffer_builder.build()?;

        let compute_start_time = Instant::now();
        let future = sync::now(device.clone())
            .then_execute(queue.clone(), command_buffer)?
            .then_signal_fence_and_flush()?;

        future.wait(None)?;

        let compute_end_time = compute_start_time.elapsed();
        debug!("Compute icp shader execution time: {:?}", compute_end_time);

        // partial バッファを CPU で sum して H(36), b(6) を得る
        let h_partials = self
            .staging_buf_partial_h
            .as_ref()
            .context("Failed to get staging partial H buffer")?
            .read()?;
        let b_partials = self
            .staging_buf_partial_b
            .as_ref()
            .context("Failed to get staging partial b buffer")?
            .read()?;

        let mut output_h = vec![0.0f32; 36];
        let mut output_b = vec![0.0f32; 6];
        for g in 0..num_groups {
            for i in 0..36 {
                output_h[i] += h_partials[g * 36 + i];
            }
            for i in 0..6 {
                output_b[i] += b_partials[g * 6 + i];
            }
        }

        Ok((output_h, output_b))
    }
}

pub fn solve_gicp(system: (Matrix6<f32>, Vector6<f32>), damping: f32) -> Option<Matrix4<f64>> {
    let (mut h, b) = system;
    for i in 0..6 {
        h[(i, i)] += damping;
    }

    let delta = h.lu().solve(&b)?;

    let rot_vec = Vector3::new(delta[0], delta[1], delta[2]);
    let trans_vec = Vector3::new(delta[3], delta[4], delta[5]);

    let angle = rot_vec.norm();
    let rotation = if angle < 1.0e-10 {
        UnitQuaternion::identity()
    } else {
        UnitQuaternion::from_axis_angle(&nalgebra::Unit::new_normalize(rot_vec), angle)
    };

    let mut mat = rotation.to_homogeneous().cast::<f64>();
    mat[(0, 3)] = trans_vec.x as f64;
    mat[(1, 3)] = trans_vec.y as f64;
    mat[(2, 3)] = trans_vec.z as f64;

    Some(mat)
}
