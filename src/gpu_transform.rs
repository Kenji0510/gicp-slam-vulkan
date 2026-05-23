use std::{sync::Arc, time::Instant};

use anyhow::{Context, Result};
use log::debug;
use vulkano::{
    buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer},
    command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage},
    descriptor_set::{DescriptorSet, layout::DescriptorSetLayout},
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter},
    pipeline::{
        ComputePipeline, Pipeline, PipelineLayout, PipelineShaderStageCreateInfo,
        compute::ComputePipelineCreateInfo, layout::PipelineDescriptorSetLayoutCreateInfo,
    },
    shader::{ShaderModule, ShaderModuleCreateInfo},
    sync::{self, GpuFuture},
};

use crate::{
    gpu_covariances::CovarianceGpuContext, gpu_voxel::VoxelGpuContext, init_gpu::VulkanContext,
};

#[derive(bytemuck::Pod, bytemuck::Zeroable, Clone, Copy)]
#[repr(C)]
pub struct TransformParams {
    pub r00: f32,
    pub r01: f32,
    pub r02: f32,
    pub t0: f32,
    pub r10: f32,
    pub r11: f32,
    pub r12: f32,
    pub t1: f32,
    pub r20: f32,
    pub r21: f32,
    pub r22: f32,
    pub t2: f32,
    pub num_points: u32,
}

pub struct TransformGpuContext {
    vulkan_context: VulkanContext,

    compute_pipeline_transform: Arc<ComputePipeline>,
    pipeline_layout_transform: Arc<PipelineLayout>,
    descriptor_set_layout_transform: Arc<DescriptorSetLayout>,

    pub d_buf_input_pts: Option<Subbuffer<[f32]>>,
    pub d_buf_input_covs: Option<Subbuffer<[f32]>>,
    pub d_buf_output_pts: Option<Subbuffer<[f32]>>,
    pub d_buf_output_covs: Option<Subbuffer<[f32]>>,
    pub staging_buf_output_pts: Option<Subbuffer<[f32]>>,
    pub staging_buf_output_covs: Option<Subbuffer<[f32]>>,
    pub current_capacity_pts: usize,

    pub num_points: usize,
    pub table_size: i32,
    pub voxel_size: f32,
}

impl TransformGpuContext {
    pub fn new(vulkan_context: VulkanContext) -> Result<Self> {
        let shader_transform = unsafe {
            ShaderModule::new(
                vulkan_context.device.clone(),
                ShaderModuleCreateInfo::new(&bytemuck::pod_collect_to_vec::<u8, u32>(
                    include_bytes!(concat!(env!("OUT_DIR"), "/shaders/transform.spv")),
                )),
            )
        }
        .context("Failed to load transform shader")?;

        let cs_transform = shader_transform
            .entry_point("main")
            .context("Failed to find entry point in transform shader")?;

        let stage_transform = PipelineShaderStageCreateInfo::new(cs_transform);

        let layout_transform = PipelineLayout::new(
            vulkan_context.device.clone(),
            PipelineDescriptorSetLayoutCreateInfo::from_stages([&stage_transform])
                .into_pipeline_layout_create_info(vulkan_context.device.clone())
                .context("Failed to create pipeline layout")?,
        )
        .context("Failed to create pipeline layout")?;

        let compute_pipeline_transform = ComputePipeline::new(
            vulkan_context.device.clone(),
            None,
            ComputePipelineCreateInfo::stage_layout(stage_transform, layout_transform),
        )
        .context("Failed to create compute pipeline transform")?;

        let pipeline_layout_transform = compute_pipeline_transform.layout();

        let descriptor_set_layout_transform = pipeline_layout_transform
            .set_layouts()
            .get(0)
            .context("Failed to get descriptor set layout for transform")?;

        Ok(Self {
            vulkan_context: vulkan_context.clone(),
            compute_pipeline_transform: compute_pipeline_transform.clone(),
            pipeline_layout_transform: pipeline_layout_transform.clone(),
            descriptor_set_layout_transform: descriptor_set_layout_transform.clone(),
            d_buf_input_pts: None,
            d_buf_input_covs: None,
            d_buf_output_pts: None,
            d_buf_output_covs: None,
            staging_buf_output_pts: None,
            staging_buf_output_covs: None,
            current_capacity_pts: 0,
            num_points: 0,
            table_size: 0,
            voxel_size: 0.0,
        })
    }

    // This function should be called after computing the voxelization and covariance, and before the transformation, to set up the necessary buffers and parameters.
    pub fn transform(
        &mut self,
        voxel_gpu_context: &VoxelGpuContext,
        cov_gpu_context: &CovarianceGpuContext,
        transform_params: TransformParams,
    ) -> Result<()> {
        let device = &self.vulkan_context.device;
        let queue = &self.vulkan_context.queue;
        let memory_allocator = &self.vulkan_context.memory_allocator;
        let descriptor_set_allocator = &self.vulkan_context.descriptor_set_allocator;
        let command_buffer_allocator = &self.vulkan_context.command_buffer_allocator;
        let pipeline_layout = &self.pipeline_layout_transform;
        let compute_pipeline = &self.compute_pipeline_transform;

        if self.current_capacity_pts < voxel_gpu_context.h_downsampled_pts_num {
            debug!(
                "Reallocating buffers for {} points",
                voxel_gpu_context.h_downsampled_pts_num
            );

            let new_capacity = (voxel_gpu_context.h_downsampled_pts_num as f64 * 1.5) as usize;
            self.current_capacity_pts = new_capacity;

            self.d_buf_output_pts = Some(Buffer::new_slice::<f32>(
                memory_allocator.clone(),
                BufferCreateInfo {
                    usage: BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_SRC,
                    ..Default::default()
                },
                AllocationCreateInfo {
                    memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                    ..Default::default()
                },
                (new_capacity * 3) as u64,
            )?);

            self.d_buf_output_covs = Some(Buffer::new_slice::<f32>(
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

            self.staging_buf_output_pts = Some(Buffer::new_slice::<f32>(
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
                (new_capacity * 3) as u64,
            )?);

            self.staging_buf_output_covs = Some(Buffer::new_slice::<f32>(
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
            self.descriptor_set_layout_transform.clone(),
            [
                vulkano::descriptor_set::WriteDescriptorSet::buffer(
                    0,
                    voxel_gpu_context
                        .d_buf_out_pts
                        .as_ref()
                        .context("Failed to get output points buffer")?
                        .clone(),
                ),
                vulkano::descriptor_set::WriteDescriptorSet::buffer(
                    1,
                    cov_gpu_context
                        .d_buf_covariances
                        .as_ref()
                        .context("Failed to get output covariance buffer")?
                        .clone(),
                ),
                vulkano::descriptor_set::WriteDescriptorSet::buffer(
                    2,
                    self.d_buf_output_pts
                        .as_ref()
                        .context("Failed to get output points buffer")?
                        .clone(),
                ),
                vulkano::descriptor_set::WriteDescriptorSet::buffer(
                    3,
                    self.d_buf_output_covs
                        .as_ref()
                        .context("Failed to get output covariance buffer")?
                        .clone(),
                ),
            ],
            [],
        )
        .context("Failed to create descriptor set for transform")?;

        let mut command_buffer_builder = AutoCommandBufferBuilder::primary(
            command_buffer_allocator.clone(),
            queue.queue_family_index().clone(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .context("Failed to create command buffer builder")?;

        const LOCAL_SIZE: u32 = 256;
        let group_count_x =
            (voxel_gpu_context.h_downsampled_pts_num as u32 + LOCAL_SIZE - 1) / LOCAL_SIZE;
        let work_group_count = [group_count_x, 1, 1];

        unsafe {
            command_buffer_builder
                .bind_pipeline_compute(compute_pipeline.clone())
                .context("Failed to bind compute pipeline")?
                .push_constants(pipeline_layout.clone(), 0, transform_params)
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

        // <!--- DEBUG --->
        // let copy_output_pts_src = self
        //     .d_buf_output_pts
        //     .as_ref()
        //     .context("Failed to get output points buffer for copy")?
        //     .clone()
        //     .slice(0..(voxel_gpu_context.h_downsampled_pts_num * 3) as u64);
        // let copy_output_pts_dst = self
        //     .staging_buf_output_pts
        //     .as_ref()
        //     .context("Failed to get staging output points buffer")?
        //     .clone()
        //     .slice(0..(voxel_gpu_context.h_downsampled_pts_num * 3) as u64);

        // command_buffer_builder.copy_buffer(CopyBufferInfo::buffers(
        //     copy_output_pts_src,
        //     copy_output_pts_dst,
        // ))?;

        // let copy_output_covs_src = self
        //     .d_buf_output_covs
        //     .as_ref()
        //     .context("Failed to get output covariances buffer for copy")?
        //     .clone()
        //     .slice(0..(voxel_gpu_context.h_downsampled_pts_num * 9) as u64);
        // let copy_output_covs_dst = self
        //     .staging_buf_output_covs
        //     .as_ref()
        //     .context("Failed to get staging output covariances buffer")?
        //     .clone()
        //     .slice(0..(voxel_gpu_context.h_downsampled_pts_num * 9) as u64);

        // command_buffer_builder.copy_buffer(CopyBufferInfo::buffers(
        //     copy_output_covs_src,
        //     copy_output_covs_dst,
        // ))?;
        // <!--- DEBUG --->

        let command_buffer = command_buffer_builder.build()?;

        let compute_start_time = Instant::now();
        let future = sync::now(device.clone())
            .then_execute(queue.clone(), command_buffer)?
            .then_signal_fence_and_flush()?;

        future.wait(None)?;

        let compute_end_time = compute_start_time.elapsed();
        debug!(
            "Compute transform shader execution time: {:?}",
            compute_end_time
        );

        self.num_points = voxel_gpu_context.h_downsampled_pts_num;
        self.table_size = voxel_gpu_context.table_size;
        self.voxel_size = voxel_gpu_context.voxel_size;

        // <!--- DEBUG --->
        // let out_pts_content = self
        //     .staging_buf_output_pts
        //     .as_ref()
        //     .context("Failed to get staging output points buffer")?
        //     .read()?;
        // let output_points: Vec<[f32; 3]> = out_pts_content
        //     .chunks_exact(3)
        //     .take(voxel_gpu_context.h_downsampled_pts_num)
        //     .map(|chunk| -> Result<[f32; 3]> {
        //         chunk.try_into().context("Failed to map for output points")
        //     })
        //     .collect::<Result<Vec<_>, _>>()?;

        // let out_covs_content = self
        //     .staging_buf_output_covs
        //     .as_ref()
        //     .context("Failed to get staging output covariances buffer")?
        //     .read()?;
        // let output_covs: Vec<[f32; 9]> = out_covs_content
        //     .chunks_exact(9)
        //     .take(voxel_gpu_context.h_downsampled_pts_num)
        //     .map(|chunk| -> Result<[f32; 9]> {
        //         chunk
        //             .try_into()
        //             .context("Failed to map for output covariances")
        //     })
        //     .collect::<Result<Vec<_>, _>>()?;
        // <!--- DEBUG --->

        Ok(())
    }
}
