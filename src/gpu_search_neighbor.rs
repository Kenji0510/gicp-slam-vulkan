use std::{sync::Arc, time::Instant};

use anyhow::{Context, Result};
use log::debug;
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

use crate::{
    gpu_transform::TransformGpuContext, gpu_voxel::VoxelGpuContext, init_gpu::VulkanContext,
};

const MAX_DIST_SQ: f32 = 1.0; // In case of voxel size 0.5

#[derive(bytemuck::Pod, bytemuck::Zeroable, Clone, Copy)]
#[repr(C)]
pub struct SearchNeighborParams {
    pub num_source: u32,
    pub num_target: u32,
    pub table_size: u32,
    pub search_range: i32,
    pub voxel_size: f32,
    pub max_dist_sq: f32,
}

pub struct SearchGpuContext {
    vulkan_context: VulkanContext,

    compute_pipeline_search: Arc<ComputePipeline>,
    pipeline_layout_search: Arc<PipelineLayout>,
    descriptor_set_layout_search: Arc<DescriptorSetLayout>,

    // d_buf_source_pts: Option<Subbuffer<[f32]>>,
    // d_buf_target_pts: Option<Subbuffer<[f32]>>,
    pub d_buf_indices: Option<Subbuffer<[i32]>>,
    pub d_buf_dists_sq: Option<Subbuffer<[f32]>>,

    pub staging_buf_indices: Option<Subbuffer<[i32]>>,
    pub staging_buf_dists_sq: Option<Subbuffer<[f32]>>,

    pub current_capacity_pts: usize,
}

impl SearchGpuContext {
    pub fn new(vulkan_context: VulkanContext) -> Result<Self> {
        let shader_search = unsafe {
            ShaderModule::new(
                vulkan_context.device.clone(),
                ShaderModuleCreateInfo::new(&bytemuck::pod_collect_to_vec::<u8, u32>(
                    include_bytes!(concat!(env!("OUT_DIR"), "/shaders/search_neighbor.spv")),
                )),
            )
        }
        .context("Failed to load search shader")?;

        let cs_search = shader_search
            .entry_point("main")
            .context("Failed to find entry point in search shader")?;

        let stage_search = PipelineShaderStageCreateInfo::new(cs_search);

        let layout_search = PipelineLayout::new(
            vulkan_context.device.clone(),
            PipelineDescriptorSetLayoutCreateInfo::from_stages([&stage_search])
                .into_pipeline_layout_create_info(vulkan_context.device.clone())
                .context("Failed to create pipeline layout")?,
        )
        .context("Failed to create pipeline layout")?;

        let compute_pipeline_search = ComputePipeline::new(
            vulkan_context.device.clone(),
            None,
            ComputePipelineCreateInfo::stage_layout(stage_search, layout_search),
        )
        .context("Failed to create compute pipeline search")?;

        let pipeline_layout_search = compute_pipeline_search.layout();

        let descriptor_set_layout_search = pipeline_layout_search
            .set_layouts()
            .get(0)
            .context("Failed to get descriptor set layout for search")?;

        Ok(Self {
            vulkan_context: vulkan_context.clone(),
            compute_pipeline_search: compute_pipeline_search.clone(),
            pipeline_layout_search: pipeline_layout_search.clone(),
            descriptor_set_layout_search: descriptor_set_layout_search.clone(),
            // d_buf_source_pts: None,
            // d_buf_target_pts: None,
            d_buf_indices: None,
            d_buf_dists_sq: None,
            staging_buf_indices: None,
            staging_buf_dists_sq: None,
            current_capacity_pts: 0,
        })
    }

    pub fn search_neighbor(
        &mut self,
        source_transform_gpu_context: &TransformGpuContext,
        downsampled_source_pts_num: usize,
        downsample_target_gpu_context: &VoxelGpuContext,
        downsampled_target_pts_num: usize,
        search_range: i32,
        max_dist_sq: f32,
    ) -> Result<(Vec<i32>, Vec<f32>)> {
        let device = &self.vulkan_context.device;
        let queue = &self.vulkan_context.queue;
        let memory_allocator = &self.vulkan_context.memory_allocator;
        let descriptor_set_allocator = &self.vulkan_context.descriptor_set_allocator;
        let command_buffer_allocator = &self.vulkan_context.command_buffer_allocator;
        let pipeline_layout = &self.pipeline_layout_search;
        let compute_pipeline = &self.compute_pipeline_search;

        let search_params = SearchNeighborParams {
            num_source: downsampled_source_pts_num as u32,
            num_target: downsampled_target_pts_num as u32,
            table_size: downsample_target_gpu_context.table_size as u32,
            search_range,
            voxel_size: downsample_target_gpu_context.voxel_size,
            max_dist_sq,
        };

        if self.current_capacity_pts < downsampled_source_pts_num {
            debug!(
                "Reallocating buffers for {} points",
                downsampled_source_pts_num
            );

            let new_capacity = (downsampled_source_pts_num as f32 * 1.5) as usize;
            self.current_capacity_pts = new_capacity;

            self.d_buf_indices = Some(Buffer::new_slice::<i32>(
                memory_allocator.clone(),
                BufferCreateInfo {
                    usage: BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_SRC,
                    ..Default::default()
                },
                AllocationCreateInfo {
                    memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                    ..Default::default()
                },
                new_capacity as u64,
            )?);

            self.d_buf_dists_sq = Some(Buffer::new_slice::<f32>(
                memory_allocator.clone(),
                BufferCreateInfo {
                    usage: BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_SRC,
                    ..Default::default()
                },
                AllocationCreateInfo {
                    memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                    ..Default::default()
                },
                new_capacity as u64,
            )?);

            self.staging_buf_indices = Some(Buffer::new_slice::<i32>(
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
                new_capacity as u64,
            )?);

            self.staging_buf_dists_sq = Some(Buffer::new_slice::<f32>(
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
                new_capacity as u64,
            )?);
        }

        let descriptor_set = DescriptorSet::new(
            descriptor_set_allocator.clone(),
            self.descriptor_set_layout_search.clone(),
            [
                vulkano::descriptor_set::WriteDescriptorSet::buffer(
                    0,
                    source_transform_gpu_context
                        .d_buf_output_pts
                        .as_ref()
                        .context("Missing transformed source points")?
                        .clone(),
                ),
                vulkano::descriptor_set::WriteDescriptorSet::buffer(
                    1,
                    downsample_target_gpu_context
                        .d_buf_out_pts
                        .as_ref()
                        .context("Missing target points")?
                        .clone(),
                ),
                vulkano::descriptor_set::WriteDescriptorSet::buffer(
                    2,
                    downsample_target_gpu_context
                        .d_buf_keys
                        .as_ref()
                        .context("Missing target table keys")?
                        .clone(),
                ),
                vulkano::descriptor_set::WriteDescriptorSet::buffer(
                    3,
                    downsample_target_gpu_context
                        .d_buf_table_voxel_indices
                        .as_ref()
                        .context("Missing target table voxel indices")?
                        .clone(),
                ),
                vulkano::descriptor_set::WriteDescriptorSet::buffer(
                    4,
                    self.d_buf_indices
                        .as_ref()
                        .context("Missing output indices buffer")?
                        .clone(),
                ),
                vulkano::descriptor_set::WriteDescriptorSet::buffer(
                    5,
                    self.d_buf_dists_sq
                        .as_ref()
                        .context("Missing output distances buffer")?
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

        const LOCAL_SIZE: u32 = 256;
        let group_count_x = (downsampled_source_pts_num as u32 + LOCAL_SIZE - 1) / LOCAL_SIZE;
        let work_group_count = [group_count_x, 1, 1];

        unsafe {
            command_buffer_builder
                .bind_pipeline_compute(compute_pipeline.clone())
                .context("Failed to bind compute pipeline")?
                .push_constants(pipeline_layout.clone(), 0, search_params)
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

        // <!--- Copy source neighbor indices from GPU to staging buffer --->
        let copy_output_indices_src = self
            .d_buf_indices
            .as_ref()
            .context("Failed to get output indices buffer for copy")?
            .clone()
            .slice(0..search_params.num_source as u64);
        let copy_output_indices_dst = self
            .staging_buf_indices
            .as_ref()
            .context("Failed to get staging buffer for indices copy")?
            .clone()
            .slice(0..search_params.num_source as u64);

        command_buffer_builder
            .copy_buffer(CopyBufferInfo::buffers(
                copy_output_indices_src,
                copy_output_indices_dst,
            ))
            .context("Failed to copy output indices to staging buffer")?;
        // <!--- Copy source neighbor indices from GPU to staging buffer --->

        // <!--- Copy source neighbor distances from GPU to staging buffer --->
        let copy_output_distances_src = self
            .d_buf_dists_sq
            .as_ref()
            .context("Failed to get output distances buffer for copy")?
            .clone()
            .slice(0..search_params.num_source as u64);
        let copy_output_distances_dst = self
            .staging_buf_dists_sq
            .as_ref()
            .context("Failed to get staging buffer for distances copy")?
            .clone()
            .slice(0..search_params.num_source as u64);

        command_buffer_builder
            .copy_buffer(CopyBufferInfo::buffers(
                copy_output_distances_src,
                copy_output_distances_dst,
            ))
            .context("Failed to copy output distances to staging buffer")?;
        // <!--- Copy source neighbor distances from GPU to staging buffer --->

        let command_buffer = command_buffer_builder.build()?;

        let compute_start_time = Instant::now();
        let future = sync::now(device.clone())
            .then_execute(queue.clone(), command_buffer)?
            .then_signal_fence_and_flush()?;

        future.wait(None)?;

        let compute_end_time = compute_start_time.elapsed();
        debug!(
            "Compute neighbor search shader execution time: {:?}",
            compute_end_time
        );

        // <!--- Copy results from staging buffer to CPU --->
        let indices_content = self
            .staging_buf_indices
            .as_ref()
            .context("Failed to get staging buffer for indices read")?
            .read()?;
        let output_indices: Vec<i32> = indices_content
            .iter()
            .take(search_params.num_source as usize)
            .copied()
            .collect();

        let distances_content = self
            .staging_buf_dists_sq
            .as_ref()
            .context("Failed to get staging buffer for distances read")?
            .read()?;
        let output_distances: Vec<f32> = distances_content
            .iter()
            .take(search_params.num_source as usize)
            .copied()
            .collect();
        // <!--- Copy results from staging buffer to CPU --->

        Ok((output_indices, output_distances))
    }
}
