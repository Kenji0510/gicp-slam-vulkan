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

    pub d_buf_h: Option<Subbuffer<[f32]>>,
    pub d_buf_b: Option<Subbuffer<[f32]>>,

    pub staging_buf_h: Option<Subbuffer<[f32]>>,
    pub staging_buf_b: Option<Subbuffer<[f32]>>,

    is_initialized: bool,
}

impl GicpGpuContext {
    pub fn new(vulkan_context: VulkanContext) -> Result<Self> {
        mod cs_icp {
            vulkano_shaders::shader! {
                ty: "compute",
                path: "src/kernels/gicp/gicp.glsl",
            }
        }

        let shader_icp =
            cs_icp::load(vulkan_context.device.clone()).context("Failed to load ICP shader")?;

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
            d_buf_h: None,
            d_buf_b: None,
            staging_buf_h: None,
            staging_buf_b: None,
            is_initialized: false,
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

        if self.is_initialized == false {
            debug!("Allocating buffers for H and b matrix");

            self.d_buf_h = Some(Buffer::new_slice::<f32>(
                memory_allocator.clone(),
                BufferCreateInfo {
                    usage: BufferUsage::STORAGE_BUFFER
                        | BufferUsage::TRANSFER_SRC
                        | BufferUsage::TRANSFER_DST,
                    ..Default::default()
                },
                AllocationCreateInfo {
                    memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                    ..Default::default()
                },
                6 * 6,
            )?);

            self.d_buf_b = Some(Buffer::new_slice::<f32>(
                memory_allocator.clone(),
                BufferCreateInfo {
                    usage: BufferUsage::STORAGE_BUFFER
                        | BufferUsage::TRANSFER_SRC
                        | BufferUsage::TRANSFER_DST,
                    ..Default::default()
                },
                AllocationCreateInfo {
                    memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                    ..Default::default()
                },
                6,
            )?);

            self.staging_buf_h = Some(Buffer::new_slice::<f32>(
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
                6 * 6,
            )?);

            self.staging_buf_b = Some(Buffer::new_slice::<f32>(
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
                6,
            )?);

            self.is_initialized = true;
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
                    self.d_buf_h
                        .as_ref()
                        .context("Failed to get H buffer")?
                        .clone(),
                ),
                vulkano::descriptor_set::WriteDescriptorSet::buffer(
                    7,
                    self.d_buf_b
                        .as_ref()
                        .context("Failed to get b buffer")?
                        .clone(),
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

        command_buffer_builder
            .fill_buffer(
                self.d_buf_h
                    .as_ref()
                    .context("Failed to get H buffer")?
                    .clone()
                    .reinterpret::<[u32]>(),
                0u32,
            )
            .context("Failed to fill H buffer with zeros")?
            .fill_buffer(
                self.d_buf_b
                    .as_ref()
                    .context("Failed to get b buffer")?
                    .clone()
                    .reinterpret::<[u32]>(),
                0u32,
            )
            .context("Failed to fill b buffer with zeros")?;

        const LOCAL_SIZE: u32 = 64; // Match the local size to BLOCK_SIZE in the shader
        let group_count_x = (source_pts_num as u32 + LOCAL_SIZE - 1) / LOCAL_SIZE;
        let work_group_count = [group_count_x, 1, 1];

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

        // <!--- Copy H from GPU to staging buffer --->
        let copy_output_h_src = self
            .d_buf_h
            .as_ref()
            .context("Failed to get output H buffer for copy")?
            .clone()
            .slice(0..(6 * 6) as u64);
        let copy_output_h_dst = self
            .staging_buf_h
            .as_ref()
            .context("Failed to get staging buffer for H copy")?
            .clone()
            .slice(0..(6 * 6) as u64);
        command_buffer_builder
            .copy_buffer(CopyBufferInfo::buffers(
                copy_output_h_src,
                copy_output_h_dst,
            ))
            .context("Failed to copy output H to staging buffer")?;
        // <!--- Copy H from GPU to staging buffer --->

        // <!--- Copy b from GPU to staging buffer --->
        let copy_output_b_src = self
            .d_buf_b
            .as_ref()
            .context("Failed to get output b buffer for copy")?
            .clone()
            .slice(0..6 as u64);
        let copy_output_b_dst = self
            .staging_buf_b
            .as_ref()
            .context("Failed to get staging buffer for b copy")?
            .clone()
            .slice(0..6 as u64);
        command_buffer_builder
            .copy_buffer(CopyBufferInfo::buffers(
                copy_output_b_src,
                copy_output_b_dst,
            ))
            .context("Failed to copy output b to staging buffer")?;
        // <!--- Copy b from GPU to staging buffer --->

        let command_buffer = command_buffer_builder.build()?;

        let compute_start_time = Instant::now();
        let future = sync::now(device.clone())
            .then_execute(queue.clone(), command_buffer)?
            .then_signal_fence_and_flush()?;

        future.wait(None)?;

        let compute_end_time = compute_start_time.elapsed();
        debug!("Compute icp shader execution time: {:?}", compute_end_time);

        // <!--- Copy results from staging buffer to CPU --->
        let h_content = self
            .staging_buf_h
            .as_ref()
            .context("Failed to get staging buffer for H read")?
            .read()?;
        let output_h: Vec<f32> = h_content.iter().take(6 * 6 as usize).copied().collect();

        let distances_content = self
            .staging_buf_b
            .as_ref()
            .context("Failed to get staging buffer for b read")?
            .read()?;
        let output_b: Vec<f32> = distances_content.iter().take(6 as usize).copied().collect();
        // <!--- Copy results from staging buffer to CPU --->

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
