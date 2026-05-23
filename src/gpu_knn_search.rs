use std::{sync::Arc, time::Instant};

use foldhash::{HashMap, HashMapExt};

use anyhow::{Context, Result};
use log::debug;
use vulkano::shader::{ShaderModule, ShaderModuleCreateInfo, SpecializationConstant};
use vulkano::{
    buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer},
    command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage},
    descriptor_set::{DescriptorSet, layout::DescriptorSetLayout},
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter},
    pipeline::{
        ComputePipeline, Pipeline, PipelineLayout, PipelineShaderStageCreateInfo,
        compute::ComputePipelineCreateInfo, layout::PipelineDescriptorSetLayoutCreateInfo,
    },
    sync::{self, GpuFuture},
};

use crate::{gpu_voxel::VoxelGpuContext, init_gpu::VulkanContext};

const K_NEIGHBORS: usize = 8;

#[derive(bytemuck::Pod, bytemuck::Zeroable, Clone, Copy)]
#[repr(C)]
pub struct KnnSearchConsts {
    pub num_points: u32,
}

pub struct KnnSearchGpuContext {
    vulkan_context: VulkanContext,

    compute_pipeline_search: Arc<ComputePipeline>,
    pipeline_layout_search: Arc<PipelineLayout>,
    descriptor_set_layout_search: Arc<DescriptorSetLayout>,

    pub d_buf_indices: Option<Subbuffer<[i32]>>,
    pub d_buf_dists_sq: Option<Subbuffer<[f32]>>,

    pub staging_buf_indices: Option<Subbuffer<[i32]>>,
    pub staging_buf_dists_sq: Option<Subbuffer<[f32]>>,

    pub current_capacity_pts: usize,
}

impl KnnSearchGpuContext {
    pub fn new(vulkan_context: VulkanContext) -> Result<Self> {
        let shader_search = unsafe {
            ShaderModule::new(
                vulkan_context.device.clone(),
                ShaderModuleCreateInfo::new(&bytemuck::pod_collect_to_vec::<u8, u32>(
                    include_bytes!(concat!(env!("OUT_DIR"), "/shaders/knn_search.spv")),
                )),
            )
        }
        .context("Failed to load search shader")?;

        let mut spec = HashMap::new();
        spec.insert(0u32, SpecializationConstant::U32(K_NEIGHBORS as u32));

        let cs_search = shader_search
            .specialize(spec)
            .context("Failed to specialize search shader with constants")?
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
            d_buf_indices: None,
            d_buf_dists_sq: None,
            staging_buf_indices: None,
            staging_buf_dists_sq: None,
            current_capacity_pts: 0,
        })
    }

    pub fn knn_search_neighbors(
        &mut self,
        target_voxel_gpu_context: &VoxelGpuContext,
    ) -> Result<()> {
        let device = &self.vulkan_context.device;
        let queue = &self.vulkan_context.queue;
        let memory_allocator = &self.vulkan_context.memory_allocator;
        let descriptor_set_allocator = &self.vulkan_context.descriptor_set_allocator;
        let command_buffer_allocator = &self.vulkan_context.command_buffer_allocator;
        let pipeline_layout = &self.pipeline_layout_search;
        let compute_pipeline = &self.compute_pipeline_search;

        let target_pts_num = target_voxel_gpu_context.h_downsampled_pts_num;

        let knn_search_params = KnnSearchConsts {
            num_points: target_pts_num as u32,
        };

        if self.current_capacity_pts < target_pts_num {
            debug!("Reallocating buffers for {} points", target_pts_num);

            let new_capacity = (target_pts_num as f32 * 1.5) as usize;
            self.current_capacity_pts = new_capacity;

            self.d_buf_indices = Some(Buffer::new_slice::<i32>(
                memory_allocator.clone(),
                BufferCreateInfo {
                    usage: BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_DST,
                    ..Default::default()
                },
                AllocationCreateInfo {
                    memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                    ..Default::default()
                },
                (new_capacity * K_NEIGHBORS) as u64,
            )?);

            self.d_buf_dists_sq = Some(Buffer::new_slice::<f32>(
                memory_allocator.clone(),
                BufferCreateInfo {
                    usage: BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_DST,
                    ..Default::default()
                },
                AllocationCreateInfo {
                    memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                    ..Default::default()
                },
                (new_capacity * K_NEIGHBORS) as u64,
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
                    self.d_buf_indices
                        .as_ref()
                        .context("Failed to get indices buffer")?
                        .clone(),
                ),
                vulkano::descriptor_set::WriteDescriptorSet::buffer(
                    2,
                    self.d_buf_dists_sq
                        .as_ref()
                        .context("Failed to get distances buffer")?
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
        let group_count_x = (target_pts_num as u32 + LOCAL_SIZE - 1) / LOCAL_SIZE;
        let work_group_count = [group_count_x, 1, 1];

        unsafe {
            command_buffer_builder
                .bind_pipeline_compute(compute_pipeline.clone())
                .context("Failed to bind compute pipeline")?
                .push_constants(pipeline_layout.clone(), 0, knn_search_params)
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

        let command_buffer = command_buffer_builder.build()?;

        let compute_start_time = Instant::now();
        let future = sync::now(device.clone())
            .then_execute(queue.clone(), command_buffer)?
            .then_signal_fence_and_flush()?;

        future.wait(None)?;

        let compute_end_time = compute_start_time.elapsed();
        debug!(
            "Compute knn search shader execution time: {:?}",
            compute_end_time
        );

        Ok(())
    }
}
