use std::sync::Arc;

use log::debug;
use vulkano::{
    buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer},
    command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage, CopyBufferInfo},
    descriptor_set::layout::DescriptorSetLayout,
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter},
    pipeline::{ComputePipeline, PipelineLayout},
    sync::{self, GpuFuture},
};

use crate::init_gpu::VulkanContext;
use anyhow::{Context, Result};

pub struct GpuTransferDataContext {
    vulkan_context: VulkanContext,

    // compute_pipeline: Arc<ComputePipeline>,
    // pipeline_layout: Arc<PipelineLayout>,
    // descriptor_set_layout_init: Arc<DescriptorSetLayout>,
    staging_buf_input_pts: Option<Subbuffer<[f32]>>,
    pub d_buf_input_pts: Option<Subbuffer<[f32]>>,

    pub num_points: usize,
    current_capacity: usize,
}

impl GpuTransferDataContext {
    pub fn new(vulkan_context: VulkanContext) -> Result<Self> {
        Ok(Self {
            vulkan_context: vulkan_context.clone(),
            staging_buf_input_pts: None,
            d_buf_input_pts: None,
            num_points: 0,
            current_capacity: 0,
        })
    }

    pub fn copy_data_to_gpu(&mut self, pts: &[[f32; 3]], num_pts: usize) -> Result<()> {
        if num_pts == 0 {
            return Ok(());
        }

        let device = &self.vulkan_context.device;
        let queue = &self.vulkan_context.queue;
        let memory_allocator = &self.vulkan_context.memory_allocator;
        let descriptor_set_allocator = &self.vulkan_context.descriptor_set_allocator;
        let command_buffer_allocator = &self.vulkan_context.command_buffer_allocator;

        self.num_points = num_pts;

        if self.current_capacity < num_pts {
            debug!(
                "Allocating GPU buffers for {} points (previous capacity: {})",
                num_pts, self.current_capacity
            );

            let new_capacity = (num_pts as f64 * 1.5) as usize;
            self.current_capacity = new_capacity;

            // input points buffer
            self.staging_buf_input_pts = Some(Buffer::new_slice::<f32>(
                memory_allocator.clone(),
                BufferCreateInfo {
                    usage: BufferUsage::TRANSFER_SRC,
                    ..Default::default()
                },
                AllocationCreateInfo {
                    memory_type_filter: MemoryTypeFilter::PREFER_HOST
                        | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                    ..Default::default()
                },
                (new_capacity * 3) as u64,
            )?);

            self.d_buf_input_pts = Some(Buffer::new_slice::<f32>(
                memory_allocator.clone(),
                BufferCreateInfo {
                    usage: BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_DST,
                    ..Default::default()
                },
                AllocationCreateInfo {
                    memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                    ..Default::default()
                },
                (new_capacity * 3) as u64,
            )?);
        }

        if let Some(staging_buf) = &self.staging_buf_input_pts {
            let mut mapping = staging_buf.write()?;
            debug_assert!(pts.len() >= num_pts, "pts.len() < num_pts");
            for (chunk, pt) in mapping.chunks_exact_mut(3).zip(pts.iter()) {
                chunk.copy_from_slice(pt);
            }
        }

        let mut command_buffer_builder = AutoCommandBufferBuilder::primary(
            command_buffer_allocator.clone(),
            queue.queue_family_index().clone(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .context("Failed to create command buffer builder")?;

        let copy_src = self
            .staging_buf_input_pts
            .as_ref()
            .context("staging_buf_input_pts is None")?
            .clone()
            .slice(0..(num_pts * 3) as u64);
        let copy_dst = self
            .d_buf_input_pts
            .as_ref()
            .context("d_buf_input_pts is None")?
            .clone()
            .slice(0..(num_pts * 3) as u64);
        command_buffer_builder.copy_buffer(CopyBufferInfo::buffers(copy_src, copy_dst))?;

        let command_buffer = command_buffer_builder.build()?;

        let future = sync::now(device.clone())
            .then_execute(queue.clone(), command_buffer)?
            .then_signal_fence_and_flush()?;

        future.wait(None)?;

        debug!("Copied {} points to GPU buffer", num_pts);

        Ok(())
    }
}
