use anyhow::Result;
use log::{debug, info};
use std::sync::Arc;

use vulkano::{
    Version, VulkanLibrary,
    buffer::Subbuffer,
    command_buffer::allocator::StandardCommandBufferAllocator,
    descriptor_set::allocator::StandardDescriptorSetAllocator,
    device::{
        Device, DeviceCreateInfo, DeviceExtensions, DeviceFeatures, Queue, QueueCreateInfo,
        QueueFlags,
    },
    instance::{Instance, InstanceCreateFlags, InstanceCreateInfo, InstanceExtensions},
    memory::allocator::StandardMemoryAllocator,
};

pub struct GpuBuffer {
    pub voxel_d_buf_source_pts: Option<Subbuffer<[f32]>>,
    pub voxel_d_buf_source_pts_num: usize,
    pub voxel_d_buf_source_covs: Option<Subbuffer<[f32]>>,
    pub voxel_d_buf_target_pts: Option<Subbuffer<[f32]>>,
    pub voxel_d_buf_target_pts_num: usize,
    pub voxel_d_buf_target_covs: Option<Subbuffer<[f32]>>,
}

#[derive(Clone)]
pub struct VulkanContext {
    // _library: Arc<VulkanLibrary>,
    // instance: Arc<Instance>,
    // physical_device: Arc<PhysicalDevice>,
    pub device: Arc<Device>,
    pub queue: Arc<Queue>,
    pub memory_allocator: Arc<StandardMemoryAllocator>,
    pub command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    pub descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
}

impl VulkanContext {
    pub fn new() -> Result<Self> {
        let library = VulkanLibrary::new().expect("Failed to load vulkan library");
        let mut required_extensions = InstanceExtensions::empty();

        if required_extensions.khr_get_physical_device_properties2 {
            required_extensions.khr_get_physical_device_properties2 = true;
        }
        let mut flags = InstanceCreateFlags::empty();
        if required_extensions.khr_portability_enumeration {
            required_extensions.khr_portability_enumeration = true;
            flags |= InstanceCreateFlags::ENUMERATE_PORTABILITY;
        }

        let instance = Instance::new(
            library,
            InstanceCreateInfo {
                enabled_extensions: required_extensions,
                flags,
                max_api_version: Some(Version::V1_3),
                ..Default::default()
            },
        )
        .expect("Failed to create instance");

        // Enumerate all available devices and print their information
        let available_devices: Vec<_> = instance
            .enumerate_physical_devices()
            .expect("Could not enumerate devices!")
            .collect();

        if available_devices.is_empty() {
            anyhow::bail!("No Vulkan devices available");
        }

        // println!("\n=== Available Vulkan Devices ===");
        debug!("=== Available Vulkan Devices ===");
        for (i, device) in available_devices.iter().enumerate() {
            debug!(
                "Device {}: {} ({:?})",
                i,
                device.properties().device_name,
                device.properties().device_type
            );
        }
        debug!("===============================\n");

        // Select the best GPU device - prioritize discrete GPU
        let physical_device = available_devices
            .iter()
            .find(|d| {
                d.properties().device_type
                    == vulkano::device::physical::PhysicalDeviceType::DiscreteGpu
            })
            .or_else(|| {
                // Fall back to integrated GPU if no discrete GPU found
                available_devices.iter().find(|d| {
                    d.properties().device_type
                        == vulkano::device::physical::PhysicalDeviceType::IntegratedGpu
                })
            })
            .or_else(|| {
                // Fall back to any other GPU type
                available_devices.iter().find(|d| {
                    d.properties().device_type
                        == vulkano::device::physical::PhysicalDeviceType::VirtualGpu
                })
            })
            .unwrap_or(&available_devices[0])
            .clone();

        debug!(
            "Selected device: {} ({:?})\n",
            physical_device.properties().device_name,
            physical_device.properties().device_type
        );

        let queue_family_index = physical_device
            .queue_family_properties()
            .iter()
            .enumerate()
            .position(|(_, queue_family_properties)| {
                queue_family_properties
                    .queue_flags
                    .contains(QueueFlags::COMPUTE | QueueFlags::TRANSFER)
            })
            .expect("Could not find a compute queue family!")
            as u32;

        let supported_extensions = physical_device.supported_extensions();
        if !supported_extensions.ext_shader_atomic_float {
            info!("Warning: Device does not support ext_shader_atomic_float extension.");
            info!("Using fallback implementation with integer atomics.");
        }

        let features = physical_device.supported_features();
        if !features.shader_buffer_float32_atomic_add {
            info!("Warning: Device does not support float32 atomic add feature.");
            info!("Using fallback implementation with integer atomics.");
        }

        // if !features.shader_shared_float32_atomic_add {
        //     eprintln!("Error: Device does not support shader_shared_float32_atomic_add!");
        //     anyhow::bail!("Missing required feature for shared memory atomics");
        // }

        let (device, mut queues) = Device::new(
            physical_device.clone(),
            DeviceCreateInfo {
                enabled_extensions: DeviceExtensions {
                    ext_shader_atomic_float: supported_extensions.ext_shader_atomic_float,
                    khr_portability_subset: supported_extensions.khr_portability_subset,
                    ..Default::default()
                },
                enabled_features: DeviceFeatures {
                    shader_buffer_float32_atomic_add: features.shader_buffer_float32_atomic_add,
                    // shader_shared_float32_atomic_add: true,
                    ..Default::default()
                },
                queue_create_infos: vec![QueueCreateInfo {
                    queue_family_index,
                    ..Default::default()
                }],
                ..Default::default()
            },
        )
        .expect("Failed to create device!");

        let queue = queues.next().unwrap();

        let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));

        let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(
            device.clone(),
            Default::default(),
        ));

        let descriptor_set_allocator = Arc::new(StandardDescriptorSetAllocator::new(
            device.clone(),
            Default::default(),
        ));

        // println!("=== Vulkan Device Information ===");
        debug!("=== Vulkan Device Information ===");
        debug!("Device Name: {}", physical_device.properties().device_name);
        debug!(
            "Device Type: {:?}",
            physical_device.properties().device_type
        );
        debug!("Vulkan context initialized successfully.");
        debug!("===============================\n");

        Ok(VulkanContext {
            device,
            queue,
            memory_allocator,
            command_buffer_allocator,
            descriptor_set_allocator,
        })
    }
}
