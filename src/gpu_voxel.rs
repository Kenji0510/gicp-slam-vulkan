use anyhow::{Context, Result};
use core::num;
use log::debug;
use std::{sync::Arc, time::Instant};
use vulkano::{
    buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer},
    command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage, CopyBufferInfo},
    descriptor_set::{DescriptorSet, WriteDescriptorSet, layout::DescriptorSetLayout},
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter},
    pipeline::{
        ComputePipeline, Pipeline, PipelineLayout, PipelineShaderStageCreateInfo,
        compute::ComputePipelineCreateInfo, layout::PipelineDescriptorSetLayoutCreateInfo,
    },
    shader::{ShaderModule, ShaderModuleCreateInfo},
    sync::{self, GpuFuture},
};

use crate::{gpu_copy::GpuTransferDataContext, init_gpu::VulkanContext};

#[derive(bytemuck::Pod, bytemuck::Zeroable, Clone, Copy)]
#[repr(C)]
pub struct PushConsts {
    pub num_points: i32,
    pub table_size: i32,
    pub voxel_size: f32,
    pub _pad: i32,
}

pub struct VoxelGpuContext {
    vulkan_context: VulkanContext,

    compute_pipeline_init: Arc<ComputePipeline>,
    compute_pipeline_insert: Arc<ComputePipeline>,
    compute_pipeline_compact: Arc<ComputePipeline>,
    pipeline_layout_init: Arc<PipelineLayout>,
    pipeline_layout_insert: Arc<PipelineLayout>,
    pipeline_layout_compact: Arc<PipelineLayout>,
    descriptor_set_layout_init: Arc<DescriptorSetLayout>,
    descriptor_set_layout_insert: Arc<DescriptorSetLayout>,
    descriptor_set_layout_compact: Arc<DescriptorSetLayout>,

    pub d_buf_input_pts: Option<Subbuffer<[f32]>>,
    pub d_buf_keys: Option<Subbuffer<[u32]>>,
    pub d_buf_centroids: Option<Subbuffer<[u32]>>,
    pub d_buf_counts: Option<Subbuffer<[u32]>>,
    pub d_buf_out_pts: Option<Subbuffer<[f32]>>,
    pub d_buf_counter: Option<Subbuffer<[u32]>>,
    pub d_buf_table_voxel_indices: Option<Subbuffer<[u32]>>,
    pub d_buf_cluster_ids: Option<Subbuffer<[u32]>>,

    // staging_buf_input_pts: Option<Subbuffer<[f32]>>,
    pub staging_buf_output_pts: Option<Subbuffer<[f32]>>,
    pub staging_buf_output_counter: Option<Subbuffer<[u32]>>,

    pub current_capacity_pts: usize,

    pub num_points: i32,
    pub table_size: i32,
    pub voxel_size: f32,

    pub h_downsampled_pts: Option<Vec<[f32; 3]>>,
    pub h_downsampled_pts_num: usize,
}

impl VoxelGpuContext {
    pub fn new(vulkan_context: VulkanContext) -> Result<Self> {
        let shader_init = unsafe {
            ShaderModule::new(
                vulkan_context.device.clone(),
                ShaderModuleCreateInfo::new(&bytemuck::pod_collect_to_vec::<u8, u32>(
                    include_bytes!(concat!(env!("OUT_DIR"), "/shaders/voxel_init.spv")),
                )),
            )
        }
        .context("Failed to load init shader")?;
        let shader_insert = unsafe {
            ShaderModule::new(
                vulkan_context.device.clone(),
                ShaderModuleCreateInfo::new(&bytemuck::pod_collect_to_vec::<u8, u32>(
                    include_bytes!(concat!(env!("OUT_DIR"), "/shaders/voxel_insert.spv")),
                )),
            )
        }
        .context("Failed to load insert shader")?;
        let shader_compact = unsafe {
            ShaderModule::new(
                vulkan_context.device.clone(),
                ShaderModuleCreateInfo::new(&bytemuck::pod_collect_to_vec::<u8, u32>(
                    include_bytes!(concat!(env!("OUT_DIR"), "/shaders/voxel_compact.spv")),
                )),
            )
        }
        .context("Failed to load compact shader")?;;

        let cs_init = shader_init
            .entry_point("main")
            .context("Failed to find entry point in init shader")?;
        let cs_insert = shader_insert
            .entry_point("main")
            .context("Failed to find entry point in insert shader")?;
        let cs_compact = shader_compact
            .entry_point("main")
            .context("Failed to find entry point in compact shader")?;

        let stage_init = PipelineShaderStageCreateInfo::new(cs_init);
        let stage_insert = PipelineShaderStageCreateInfo::new(cs_insert);
        let stage_compact = PipelineShaderStageCreateInfo::new(cs_compact);

        let layout_init = PipelineLayout::new(
            vulkan_context.device.clone(),
            PipelineDescriptorSetLayoutCreateInfo::from_stages([&stage_init])
                .into_pipeline_layout_create_info(vulkan_context.device.clone())
                .context("Failed to create pipeline layout")?,
        )
        .context("Failed to create pipeline layout")?;
        let layout_insert = PipelineLayout::new(
            vulkan_context.device.clone(),
            PipelineDescriptorSetLayoutCreateInfo::from_stages([&stage_insert])
                .into_pipeline_layout_create_info(vulkan_context.device.clone())
                .context("Failed to create pipeline layout")?,
        )
        .context("Failed to create pipeline layout")?;
        let layout_compact = PipelineLayout::new(
            vulkan_context.device.clone(),
            PipelineDescriptorSetLayoutCreateInfo::from_stages([&stage_compact])
                .into_pipeline_layout_create_info(vulkan_context.device.clone())
                .context("Failed to create pipeline layout")?,
        )
        .context("Failed to create pipeline layout")?;

        let compute_pipeline_init = ComputePipeline::new(
            vulkan_context.device.clone(),
            None,
            ComputePipelineCreateInfo::stage_layout(stage_init, layout_init),
        )
        .context("Failed to create compute pipeline init")?;
        let compute_pipeline_insert = ComputePipeline::new(
            vulkan_context.device.clone(),
            None,
            ComputePipelineCreateInfo::stage_layout(stage_insert, layout_insert),
        )
        .context("Failed to create compute pipeline insert")?;
        let compute_pipeline_compact = ComputePipeline::new(
            vulkan_context.device.clone(),
            None,
            ComputePipelineCreateInfo::stage_layout(stage_compact, layout_compact),
        )
        .context("Failed to create compute pipeline compact")?;

        let pipeline_layout_init = compute_pipeline_init.layout();
        let pipeline_layout_insert = compute_pipeline_insert.layout();
        let pipeline_layout_compact = compute_pipeline_compact.layout();

        let descriptor_set_layout_init = pipeline_layout_init
            .set_layouts()
            .get(0)
            .context("Failed to get descriptor set layout for init")?;
        let descriptor_set_layout_insert = pipeline_layout_insert
            .set_layouts()
            .get(0)
            .context("Failed to get descriptor set layout for insert")?;
        let descriptor_set_layout_compact = pipeline_layout_compact
            .set_layouts()
            .get(0)
            .context("Failed to get descriptor set layout for compact")?;

        Ok(Self {
            vulkan_context: vulkan_context.clone(),
            compute_pipeline_init: compute_pipeline_init.clone(),
            compute_pipeline_insert: compute_pipeline_insert.clone(),
            compute_pipeline_compact: compute_pipeline_compact.clone(),
            pipeline_layout_init: pipeline_layout_init.clone(),
            pipeline_layout_insert: pipeline_layout_insert.clone(),
            pipeline_layout_compact: pipeline_layout_compact.clone(),
            descriptor_set_layout_init: descriptor_set_layout_init.clone(),
            descriptor_set_layout_insert: descriptor_set_layout_insert.clone(),
            descriptor_set_layout_compact: descriptor_set_layout_compact.clone(),
            // staging_h_buf_input_pts: None,
            d_buf_input_pts: None,
            d_buf_keys: None,
            d_buf_centroids: None,
            d_buf_counts: None,
            d_buf_out_pts: None,
            d_buf_counter: None,
            // staging_buf_input_pts: None,
            staging_buf_output_pts: None,
            staging_buf_output_counter: None,
            current_capacity_pts: 0,
            num_points: 0,
            table_size: 0,
            voxel_size: 0.0,
            d_buf_table_voxel_indices: None,
            d_buf_cluster_ids: None,
            h_downsampled_pts: None,
            h_downsampled_pts_num: 0,
        })
    }

    pub fn voxelization(
        &mut self,
        data_on_gpu: &GpuTransferDataContext,
        voxel_size: f32,
    ) -> Result<Vec<[f32; 3]>> {
        let num_pts = data_on_gpu.num_points as usize;
        if num_pts == 0 {
            return Ok(vec![]);
        }

        let device = &self.vulkan_context.device;
        let queue = &self.vulkan_context.queue;
        let memory_allocator = &self.vulkan_context.memory_allocator;
        let descriptor_set_allocator = &self.vulkan_context.descriptor_set_allocator;
        let command_buffer_allocator = &self.vulkan_context.command_buffer_allocator;
        let pipeline_layout_init = &self.pipeline_layout_init;
        let pipeline_layout_insert = &self.pipeline_layout_insert;
        let pipeline_layout_compact = &self.pipeline_layout_compact;
        let compute_pipeline_init = &self.compute_pipeline_init;
        let compute_pipeline_insert = &self.compute_pipeline_insert;
        let compute_pipeline_compact = &self.compute_pipeline_compact;

        let table_size = num_pts * 4;
        self.num_points = num_pts as i32;
        self.table_size = table_size as i32;
        self.voxel_size = voxel_size;
        let consts_data = PushConsts {
            num_points: self.num_points,
            table_size: self.table_size,
            voxel_size: self.voxel_size,
            _pad: 0,
        };

        self.d_buf_input_pts = data_on_gpu.d_buf_input_pts.clone();

        if self.current_capacity_pts < num_pts {
            debug!("Reallocating buffers for {} points", num_pts);

            let new_capacity = (num_pts as f64 * 1.5) as usize;
            self.current_capacity_pts = new_capacity;
            let new_table_size = new_capacity * 4;

            // For input points
            // self.staging_buf_input_pts = Some(Buffer::new_slice::<f32>(
            //     memory_allocator.clone(),
            //     BufferCreateInfo {
            //         usage: BufferUsage::TRANSFER_SRC,
            //         ..Default::default()
            //     },
            //     AllocationCreateInfo {
            //         memory_type_filter: MemoryTypeFilter::PREFER_HOST
            //             | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            //         ..Default::default()
            //     },
            //     (new_capacity * 3) as u64,
            // )?);

            // self.d_buf_input_pts = Some(Buffer::new_slice::<f32>(
            //     memory_allocator.clone(),
            //     BufferCreateInfo {
            //         usage: BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_DST,
            //         ..Default::default()
            //     },
            //     AllocationCreateInfo {
            //         memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
            //         ..Default::default()
            //     },
            //     (new_capacity * 3) as u64,
            // )?);

            self.d_buf_keys = Some(Buffer::new_slice::<u32>(
                memory_allocator.clone(),
                BufferCreateInfo {
                    usage: BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_DST,
                    ..Default::default()
                },
                AllocationCreateInfo {
                    memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                    ..Default::default()
                },
                new_table_size as u64,
            )?);

            self.d_buf_centroids = Some(Buffer::new_slice::<u32>(
                memory_allocator.clone(),
                BufferCreateInfo {
                    usage: BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_DST,
                    ..Default::default()
                },
                AllocationCreateInfo {
                    memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                    ..Default::default()
                },
                (new_table_size * 3) as u64,
            )?);

            self.d_buf_counts = Some(Buffer::new_slice::<u32>(
                memory_allocator.clone(),
                BufferCreateInfo {
                    usage: BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_DST,
                    ..Default::default()
                },
                AllocationCreateInfo {
                    memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                    ..Default::default()
                },
                new_table_size as u64,
            )?);

            self.d_buf_out_pts = Some(Buffer::new_slice::<f32>(
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

            self.d_buf_counter = Some(Buffer::new_slice::<u32>(
                memory_allocator.clone(),
                BufferCreateInfo {
                    usage: BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_SRC,
                    ..Default::default()
                },
                AllocationCreateInfo {
                    memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                    ..Default::default()
                },
                1,
            )?);

            self.d_buf_table_voxel_indices = Some(Buffer::new_slice::<u32>(
                memory_allocator.clone(),
                BufferCreateInfo {
                    usage: BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_SRC,
                    ..Default::default()
                },
                AllocationCreateInfo {
                    memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                    ..Default::default()
                },
                new_table_size as u64,
            )?);

            self.d_buf_cluster_ids = Some(Buffer::new_slice::<u32>(
                memory_allocator.clone(),
                BufferCreateInfo {
                    usage: BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_SRC,
                    ..Default::default()
                },
                AllocationCreateInfo {
                    memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                    ..Default::default()
                },
                new_table_size as u64,
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

            self.staging_buf_output_counter = Some(Buffer::new_slice::<u32>(
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
                1,
            )?);
        }

        // input points
        // if let Some(staging_buf) = &self.staging_buf_input_pts {
        //     let mut mapping = staging_buf.write()?;
        //     for (chunk, pt) in mapping.chunks_exact_mut(3).zip(pts.iter()) {
        //         chunk.copy_from_slice(pt);
        //     }
        // }

        let descriptor_set_init = DescriptorSet::new(
            descriptor_set_allocator.clone(),
            self.descriptor_set_layout_init.clone(),
            [
                // WriteDescriptorSet::buffer(0, buf_input_pts.clone()),
                WriteDescriptorSet::buffer(
                    1,
                    self.d_buf_keys
                        .as_ref()
                        .context("d_buf_keys is None")?
                        .clone(),
                ),
                WriteDescriptorSet::buffer(
                    2,
                    self.d_buf_centroids
                        .as_ref()
                        .context("d_buf_centroids is None")?
                        .clone(),
                ),
                WriteDescriptorSet::buffer(
                    3,
                    self.d_buf_counts
                        .as_ref()
                        .context("d_buf_counts is None")?
                        .clone(),
                ),
                // WriteDescriptorSet::buffer(4, buf_out_pts.clone()),
                WriteDescriptorSet::buffer(
                    5,
                    self.d_buf_counter
                        .as_ref()
                        .context("d_buf_counter is None")?
                        .clone(),
                ),
                WriteDescriptorSet::buffer(
                    6,
                    self.d_buf_table_voxel_indices
                        .as_ref()
                        .context("d_buf_table_voxel_indices is None")?
                        .clone(),
                ),
            ],
            [],
        )
        .expect("Failed to set descriptor_set_init");

        let descriptor_set_insert = DescriptorSet::new(
            descriptor_set_allocator.clone(),
            self.descriptor_set_layout_insert.clone(),
            [
                WriteDescriptorSet::buffer(
                    0,
                    self.d_buf_input_pts
                        .as_ref()
                        .context("d_buf_input_pts is None")?
                        .clone(),
                ),
                WriteDescriptorSet::buffer(
                    1,
                    self.d_buf_keys
                        .as_ref()
                        .context("d_buf_keys is None")?
                        .clone(),
                ),
                WriteDescriptorSet::buffer(
                    2,
                    self.d_buf_centroids
                        .as_ref()
                        .context("d_buf_centroids is None")?
                        .clone(),
                ),
                WriteDescriptorSet::buffer(
                    3,
                    self.d_buf_counts
                        .as_ref()
                        .context("d_buf_counts is None")?
                        .clone(),
                ),
                // WriteDescriptorSet::buffer(4, buf_out_pts.clone()),
                // WriteDescriptorSet::buffer(5, buf_counter.clone()),
            ],
            [],
        )
        .expect("Failed to set descriptor_set_insert");

        let descriptor_set_compact = DescriptorSet::new(
            descriptor_set_allocator.clone(),
            self.descriptor_set_layout_compact.clone(),
            [
                // WriteDescriptorSet::buffer(0, buf_input_pts.clone()),
                WriteDescriptorSet::buffer(
                    1,
                    self.d_buf_keys
                        .as_ref()
                        .context("d_buf_keys is None")?
                        .clone(),
                ),
                WriteDescriptorSet::buffer(
                    2,
                    self.d_buf_centroids
                        .as_ref()
                        .context("d_buf_centroids is None")?
                        .clone(),
                ),
                WriteDescriptorSet::buffer(
                    3,
                    self.d_buf_counts
                        .as_ref()
                        .context("d_buf_counts is None")?
                        .clone(),
                ),
                WriteDescriptorSet::buffer(
                    4,
                    self.d_buf_out_pts
                        .as_ref()
                        .context("d_buf_out_pts is None")?
                        .clone(),
                ),
                WriteDescriptorSet::buffer(
                    5,
                    self.d_buf_counter
                        .as_ref()
                        .context("d_buf_counter is None")?
                        .clone(),
                ),
                WriteDescriptorSet::buffer(
                    6,
                    self.d_buf_table_voxel_indices
                        .as_ref()
                        .context("d_buf_table_voxel_indices is None")?
                        .clone(),
                ),
                WriteDescriptorSet::buffer(
                    7,
                    self.d_buf_cluster_ids
                        .as_ref()
                        .context("d_buf_cluster_ids is None")?
                        .clone(),
                ),
            ],
            [],
        )
        .expect("Failed to set descriptor_set_compact");

        let mut command_buffer_builder = AutoCommandBufferBuilder::primary(
            command_buffer_allocator.clone(),
            queue.queue_family_index().clone(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .context("Failed to create command buffer builder")?;

        // let copy_src = self
        //     .staging_buf_input_pts
        //     .as_ref()
        //     .context("staging_buf_input_pts is None")?
        //     .clone()
        //     .slice(0..(num_pts * 3) as u64);
        // let copy_dst = self
        //     .d_buf_input_pts
        //     .as_ref()
        //     .context("d_buf_input_pts is None")?
        //     .clone()
        //     .slice(0..(num_pts * 3) as u64);
        // command_buffer_builder.copy_buffer(CopyBufferInfo::buffers(copy_src, copy_dst))?;

        const LOCAL_SIZE: u32 = 256;
        let group_count_x = (table_size as u32 + LOCAL_SIZE - 1) / LOCAL_SIZE;
        let work_group_count = [group_count_x, 1, 1];

        unsafe {
            command_buffer_builder
                .bind_pipeline_compute(compute_pipeline_init.clone())?
                .push_constants(pipeline_layout_init.clone(), 0, consts_data)?
                .bind_descriptor_sets(
                    vulkano::pipeline::PipelineBindPoint::Compute,
                    pipeline_layout_init.clone(),
                    0,
                    descriptor_set_init.clone(),
                )?
                .dispatch(work_group_count)?;
            command_buffer_builder
                .bind_pipeline_compute(compute_pipeline_insert.clone())?
                .push_constants(pipeline_layout_insert.clone(), 0, consts_data)?
                .bind_descriptor_sets(
                    vulkano::pipeline::PipelineBindPoint::Compute,
                    pipeline_layout_insert.clone(),
                    0,
                    descriptor_set_insert.clone(),
                )?
                .dispatch(work_group_count)?;
            command_buffer_builder
                .bind_pipeline_compute(compute_pipeline_compact.clone())?
                .push_constants(pipeline_layout_compact.clone(), 0, consts_data)?
                .bind_descriptor_sets(
                    vulkano::pipeline::PipelineBindPoint::Compute,
                    pipeline_layout_compact.clone(),
                    0,
                    descriptor_set_compact.clone(),
                )?
                .dispatch(work_group_count)?;
        }

        let copy_out_src = self
            .d_buf_out_pts
            .as_ref()
            .context("d_buf_out_pts is None")?
            .clone()
            .slice(0..(num_pts * 3) as u64);
        let copy_out_dst = self
            .staging_buf_output_pts
            .as_ref()
            .context("staging_buf_output_pts is None")?
            .clone()
            .slice(0..(num_pts * 3) as u64);
        command_buffer_builder.copy_buffer(CopyBufferInfo::buffers(copy_out_src, copy_out_dst))?;
        command_buffer_builder.copy_buffer(CopyBufferInfo::buffers(
            self.d_buf_counter
                .as_ref()
                .context("d_buf_counter is None")?
                .clone(),
            self.staging_buf_output_counter
                .as_ref()
                .context("staging_buf_output_counter is None")?
                .clone(),
        ))?;

        let command_buffer = command_buffer_builder.build()?;

        let compute_start_time = Instant::now();
        let future = sync::now(device.clone())
            .then_execute(queue.clone(), command_buffer)?
            .then_signal_fence_and_flush()?;

        future.wait(None)?;

        let compute_end_time = compute_start_time.elapsed();
        debug!(
            "Compute voxelization shader execution time: {:?}",
            compute_end_time
        );

        let counter_content = self
            .staging_buf_output_counter
            .as_ref()
            .context("staging_buf_output_counter is None")?
            .read()?;
        let num_output_points = counter_content[0] as usize;
        debug!("Number of output points: {}", num_output_points);

        let out_pts_content = self
            .staging_buf_output_pts
            .as_ref()
            .context("staging_buf_output_pts is None")?
            .read()?;
        let output_points: Vec<[f32; 3]> = out_pts_content
            .chunks_exact(3)
            .take(num_output_points)
            .map(|chunk| -> Result<[f32; 3]> {
                chunk.try_into().context("Failed to map for output points")
            })
            .collect::<Result<Vec<_>, _>>()?;

        self.h_downsampled_pts = Some(output_points.clone());
        self.h_downsampled_pts_num = num_output_points;

        Ok(output_points)
    }
}
