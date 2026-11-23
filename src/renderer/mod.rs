use std::collections::HashMap;
use std::num::NonZeroU64;

use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3};
use wgpu::util::DeviceExt;

use crate::assets::{AssetBank, MaterialRole, ModelId, SceneVertex};
use crate::world::riders::{
    build_rider_draw_parts, AiRider, RiderPartDraw, RiderPoseMode, RiderProfile,
};
use crate::world::terrain::StaticWorldGeometry;
use crate::world::{BiomeId, RouteSpline};

pub const SHADOW_MAP_SIZE: u32 = 2048;

#[derive(Debug, Clone, Copy)]
pub struct GraphicsSettings {
    pub outline_scale: f32,
    pub prop_draw_distance_m: f32,
    pub fov_offset_deg: f32,
    pub shadows_enabled: bool,
}

impl Default for GraphicsSettings {
    fn default() -> Self {
        Self {
            outline_scale: 1.0,
            prop_draw_distance_m: 560.0,
            fov_offset_deg: 0.0,
            shadows_enabled: true,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CameraMode {
    MainMenu,
    Chase,
    Close,
    Wide,
    Side,
    FirstPerson,
    TurnaroundBack,
    TurnaroundBack34,
    TurnaroundFront34,
}

impl CameraMode {
    pub fn name(self) -> &'static str {
        match self {
            CameraMode::MainMenu => "Main Menu",
            CameraMode::Chase => "Chase",
            CameraMode::Close => "Close",
            CameraMode::Wide => "Wide",
            CameraMode::Side => "Side",
            CameraMode::FirstPerson => "First Person",
            CameraMode::TurnaroundBack => "Back",
            CameraMode::TurnaroundBack34 => "3/4 Back",
            CameraMode::TurnaroundFront34 => "3/4 Front",
        }
    }

    pub fn next(self) -> Self {
        match self {
            CameraMode::MainMenu | CameraMode::Chase => CameraMode::Close,
            CameraMode::Close => CameraMode::Wide,
            CameraMode::Wide => CameraMode::Side,
            CameraMode::Side => CameraMode::TurnaroundBack34,
            CameraMode::TurnaroundBack34 => CameraMode::TurnaroundFront34,
            CameraMode::TurnaroundFront34 => CameraMode::FirstPerson,
            CameraMode::FirstPerson | CameraMode::TurnaroundBack => CameraMode::Chase,
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct FrameUniforms {
    pub view_proj: [[f32; 4]; 4],
    pub inv_view_proj: [[f32; 4]; 4],
    pub view_mat: [[f32; 4]; 4],
    pub light_view_proj: [[f32; 4]; 4],
    pub camera_pos: [f32; 4],
    pub sun_dir: [f32; 4],
    pub sun_color: [f32; 4],
    pub sky_zenith: [f32; 4],
    pub sky_horizon: [f32; 4],
    pub fog_color: [f32; 4],
    pub screen_params: [f32; 4],
    pub outline_params: [f32; 4],
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct ObjectUniforms {
    pub model: [[f32; 4]; 4],
    pub normal_mat: [[f32; 4]; 4],
    pub tint: [f32; 4],
    pub meta: [f32; 4],
    pub skin_pose_r: [f32; 4],
    pub skin_pose_l: [f32; 4],
}

struct GpuMeshSection {
    vertex_buf: wgpu::Buffer,
    index_buf: wgpu::Buffer,
    index_count: u32,
    role: MaterialRole,
    #[allow(dead_code)]
    base_color: [f32; 4],
}

struct GpuModel {
    sections: Vec<GpuMeshSection>,
}

struct GpuWorldBuffers {
    road: Option<(wgpu::Buffer, wgpu::Buffer, u32)>,
    terrain: Option<(wgpu::Buffer, wgpu::Buffer, u32)>,
    wire: Option<(wgpu::Buffer, wgpu::Buffer, u32)>,
    water: Option<(wgpu::Buffer, wgpu::Buffer, u32)>,
}

struct GBufferTargets {
    width: u32,
    height: u32,
    lit_tex: wgpu::Texture,
    lit_view: wgpu::TextureView,
    normal_tex: wgpu::Texture,
    normal_view: wgpu::TextureView,
    depth_meta_tex: wgpu::Texture,
    depth_meta_view: wgpu::TextureView,
    scene_depth_tex: wgpu::Texture,
    scene_depth_view: wgpu::TextureView,
    post_bind_group: wgpu::BindGroup,
}

#[derive(Clone)]
struct DrawItem {
    offset_bytes: u32,
    model_id: Option<ModelId>,
    section_idx: usize,
    world_kind: u8, // 0=model, 1=road, 2=terrain, 3=wire, 4=water
}

pub struct LigneClaireRenderer {
    pub output_format: wgpu::TextureFormat,
    models: HashMap<ModelId, GpuModel>,
    world_bufs: GpuWorldBuffers,

    frame_buf: wgpu::Buffer,
    obj_align: usize,
    obj_buf: wgpu::Buffer,
    obj_capacity: usize,
    obj_staging: Vec<u8>,

    shadow_tex: wgpu::Texture,
    shadow_view: wgpu::TextureView,
    #[allow(dead_code)]
    shadow_sampler: wgpu::Sampler,
    gbuf_sampler: wgpu::Sampler,

    frame_bgl: wgpu::BindGroupLayout,
    scene_frame_bgl: wgpu::BindGroupLayout,
    post_bgl: wgpu::BindGroupLayout,
    obj_bgl: wgpu::BindGroupLayout,

    frame_bg: wgpu::BindGroup,
    scene_frame_bg: wgpu::BindGroup,
    obj_bg: wgpu::BindGroup,

    shadow_pipeline: wgpu::RenderPipeline,
    sky_pipeline: wgpu::RenderPipeline,
    scene_pipeline: wgpu::RenderPipeline,
    water_pipeline: wgpu::RenderPipeline,
    post_pipeline: wgpu::RenderPipeline,

    gbuf: GBufferTargets,
    pub egui_renderer: egui_wgpu::Renderer,
}

fn upload_mesh(
    device: &wgpu::Device,
    label: &str,
    verts: &[SceneVertex],
    indices: &[u32],
) -> Option<(wgpu::Buffer, wgpu::Buffer, u32)> {
    if verts.is_empty() || indices.is_empty() {
        return None;
    }
    let vb = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(&format!("{label}_vb")),
        contents: bytemuck::cast_slice(verts),
        usage: wgpu::BufferUsages::VERTEX,
    });
    let ib = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(&format!("{label}_ib")),
        contents: bytemuck::cast_slice(indices),
        usage: wgpu::BufferUsages::INDEX,
    });
    Some((vb, ib, indices.len() as u32))
}

impl LigneClaireRenderer {
    pub fn new(
        device: &wgpu::Device,
        output_format: wgpu::TextureFormat,
        width: u32,
        height: u32,
        assets: &AssetBank,
    ) -> Self {
        let width = width.max(1);
        let height = height.max(1);

        let mut models = HashMap::new();
        for (&id, model) in &assets.models {
            let mut gpu_secs = Vec::new();
            for (si, sec) in model.sections.iter().enumerate() {
                if let Some((vb, ib, count)) = upload_mesh(
                    device,
                    &format!("{id:?}_{si}"),
                    &sec.vertices,
                    &sec.indices,
                ) {
                    gpu_secs.push(GpuMeshSection {
                        vertex_buf: vb,
                        index_buf: ib,
                        index_count: count,
                        role: sec.role,
                        base_color: sec.base_color,
                    });
                }
            }
            models.insert(id, GpuModel { sections: gpu_secs });
        }

        let frame_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("frame_uniforms"),
            size: std::mem::size_of::<FrameUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let obj_align = device.limits().min_uniform_buffer_offset_alignment.max(256) as usize;
        let obj_capacity = 4096usize;
        let obj_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("object_uniforms"),
            size: (obj_align * obj_capacity) as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let shadow_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("shadow_map"),
            size: wgpu::Extent3d {
                width: SHADOW_MAP_SIZE,
                height: SHADOW_MAP_SIZE,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let shadow_view = shadow_tex.create_view(&wgpu::TextureViewDescriptor::default());

        let shadow_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("shadow_cmp_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            compare: Some(wgpu::CompareFunction::LessEqual),
            ..Default::default()
        });

        let gbuf_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("gbuf_nearest_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        let frame_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("frame_bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: NonZeroU64::new(std::mem::size_of::<FrameUniforms>() as u64),
                },
                count: None,
            }],
        });

        let scene_frame_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("scene_frame_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: NonZeroU64::new(std::mem::size_of::<FrameUniforms>() as u64),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Comparison),
                    count: None,
                },
            ],
        });

        let post_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("post_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: NonZeroU64::new(std::mem::size_of::<FrameUniforms>() as u64),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
            ],
        });

        let obj_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("obj_bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: true,
                    min_binding_size: NonZeroU64::new(std::mem::size_of::<ObjectUniforms>() as u64),
                },
                count: None,
            }],
        });

        let frame_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("frame_bg"),
            layout: &frame_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: frame_buf.as_entire_binding(),
            }],
        });

        let scene_frame_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("scene_frame_bg"),
            layout: &scene_frame_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: frame_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&shadow_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&shadow_sampler),
                },
            ],
        });

        let obj_bg = Self::make_obj_bg(device, &obj_bgl, &obj_buf);

        let shadow_mod = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("shadow_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/shadow.wgsl").into()),
        });
        let sky_mod = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("sky_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/sky.wgsl").into()),
        });
        let scene_mod = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("scene_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/scene.wgsl").into()),
        });
        let water_mod = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("water_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/water.wgsl").into()),
        });
        let post_mod = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("post_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/post.wgsl").into()),
        });

        let shadow_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("shadow_pl"),
            bind_group_layouts: &[Some(&frame_bgl), Some(&obj_bgl)],
            immediate_size: 0,
        });
        let shadow_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("shadow_pipeline"),
            layout: Some(&shadow_pl),
            vertex: wgpu::VertexState {
                module: &shadow_mod,
                entry_point: Some("vs_shadow"),
                compilation_options: Default::default(),
                buffers: &[Some(SceneVertex::desc())],
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::LessEqual),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState {
                    constant: 2,
                    slope_scale: 1.5,
                    clamp: 0.0,
                },
            }),
            multisample: wgpu::MultisampleState::default(),
            fragment: None,
            multiview_mask: None,
            cache: None,
        });

        let gbuf_targets = [
            Some(wgpu::ColorTargetState {
                format: wgpu::TextureFormat::Rgba8Unorm,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            }),
            Some(wgpu::ColorTargetState {
                format: wgpu::TextureFormat::Rgba8Unorm,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            }),
            Some(wgpu::ColorTargetState {
                format: wgpu::TextureFormat::Rgba8Unorm,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            }),
        ];

        let sky_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("sky_pl"),
            bind_group_layouts: &[Some(&frame_bgl)],
            immediate_size: 0,
        });
        let sky_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("sky_pipeline"),
            layout: Some(&sky_pl),
            vertex: wgpu::VertexState {
                module: &sky_mod,
                entry_point: Some("vs_sky"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: Some(false),
                depth_compare: Some(wgpu::CompareFunction::Always),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &sky_mod,
                entry_point: Some("fs_sky"),
                compilation_options: Default::default(),
                targets: &gbuf_targets,
            }),
            multiview_mask: None,
            cache: None,
        });

        let scene_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("scene_pl"),
            bind_group_layouts: &[Some(&scene_frame_bgl), Some(&obj_bgl)],
            immediate_size: 0,
        });
        let scene_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("scene_pipeline"),
            layout: Some(&scene_pl),
            vertex: wgpu::VertexState {
                module: &scene_mod,
                entry_point: Some("vs_scene"),
                compilation_options: Default::default(),
                buffers: &[Some(SceneVertex::desc())],
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::LessEqual),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &scene_mod,
                entry_point: Some("fs_scene"),
                compilation_options: Default::default(),
                targets: &gbuf_targets,
            }),
            multiview_mask: None,
            cache: None,
        });

        let water_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("water_pl"),
            bind_group_layouts: &[Some(&frame_bgl), Some(&obj_bgl)],
            immediate_size: 0,
        });
        let water_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("water_pipeline"),
            layout: Some(&water_pl),
            vertex: wgpu::VertexState {
                module: &water_mod,
                entry_point: Some("vs_water"),
                compilation_options: Default::default(),
                buffers: &[Some(SceneVertex::desc())],
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::LessEqual),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &water_mod,
                entry_point: Some("fs_water"),
                compilation_options: Default::default(),
                targets: &gbuf_targets,
            }),
            multiview_mask: None,
            cache: None,
        });

        let post_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("post_pl"),
            bind_group_layouts: &[Some(&post_bgl)],
            immediate_size: 0,
        });
        let post_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("post_pipeline"),
            layout: Some(&post_pl),
            vertex: wgpu::VertexState {
                module: &post_mod,
                entry_point: Some("vs_post"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &post_mod,
                entry_point: Some("fs_post"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: output_format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });

        let gbuf = Self::create_gbuffer(device, width, height, &post_bgl, &frame_buf, &gbuf_sampler);
        let egui_renderer = egui_wgpu::Renderer::new(
            device,
            output_format,
            egui_wgpu::RendererOptions {
                msaa_samples: 1,
                depth_stencil_format: None,
                dithering: true,
                predictable_texture_filtering: false,
            },
        );

        Self {
            output_format,
            models,
            world_bufs: GpuWorldBuffers {
                road: None,
                terrain: None,
                wire: None,
                water: None,
            },
            frame_buf,
            obj_align,
            obj_buf,
            obj_capacity,
            obj_staging: vec![0u8; obj_align * obj_capacity],
            shadow_tex,
            shadow_view,
            shadow_sampler,
            gbuf_sampler,
            frame_bgl,
            scene_frame_bgl,
            post_bgl,
            obj_bgl,
            frame_bg,
            scene_frame_bg,
            obj_bg,
            shadow_pipeline,
            sky_pipeline,
            scene_pipeline,
            water_pipeline,
            post_pipeline,
            gbuf,
            egui_renderer,
        }
    }

    fn make_obj_bg(device: &wgpu::Device, bgl: &wgpu::BindGroupLayout, buf: &wgpu::Buffer) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("obj_bg"),
            layout: bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: buf,
                    offset: 0,
                    size: NonZeroU64::new(std::mem::size_of::<ObjectUniforms>() as u64),
                }),
            }],
        })
    }

    fn create_gbuffer(
        device: &wgpu::Device,
        width: u32,
        height: u32,
        post_bgl: &wgpu::BindGroupLayout,
        frame_buf: &wgpu::Buffer,
        gbuf_sampler: &wgpu::Sampler,
    ) -> GBufferTargets {
        let mk_tex = |label: &str, fmt: wgpu::TextureFormat| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: fmt,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            })
        };
        let lit_tex = mk_tex("gbuf_lit", wgpu::TextureFormat::Rgba8Unorm);
        let normal_tex = mk_tex("gbuf_norm", wgpu::TextureFormat::Rgba8Unorm);
        let depth_meta_tex = mk_tex("gbuf_depth_meta", wgpu::TextureFormat::Rgba8Unorm);
        let scene_depth_tex = mk_tex("gbuf_z", wgpu::TextureFormat::Depth32Float);

        let lit_view = lit_tex.create_view(&Default::default());
        let normal_view = normal_tex.create_view(&Default::default());
        let depth_meta_view = depth_meta_tex.create_view(&Default::default());
        let scene_depth_view = scene_depth_tex.create_view(&Default::default());

        let post_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("post_bg"),
            layout: post_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: frame_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&lit_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&normal_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&depth_meta_view),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::Sampler(gbuf_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::TextureView(&scene_depth_view),
                },
            ],
        });

        GBufferTargets {
            width,
            height,
            lit_tex,
            lit_view,
            normal_tex,
            normal_view,
            depth_meta_tex,
            depth_meta_view,
            scene_depth_tex,
            scene_depth_view,
            post_bind_group,
        }
    }

    pub fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        let w = width.max(1);
        let h = height.max(1);
        if self.gbuf.width != w || self.gbuf.height != h {
            self.gbuf = Self::create_gbuffer(device, w, h, &self.post_bgl, &self.frame_buf, &self.gbuf_sampler);
        }
    }

    pub fn upload_world(&mut self, device: &wgpu::Device, geo: &StaticWorldGeometry) {
        self.world_bufs = GpuWorldBuffers {
            road: upload_mesh(device, "world_road", &geo.road_vertices, &geo.road_indices),
            terrain: upload_mesh(device, "world_terrain", &geo.terrain_vertices, &geo.terrain_indices),
            wire: upload_mesh(device, "world_wire", &geo.wire_vertices, &geo.wire_indices),
            water: upload_mesh(device, "world_water", &geo.water_vertices, &geo.water_indices),
        };
    }

    fn push_obj_uniform(
        &mut self,
        device: &wgpu::Device,
        idx: usize,
        model: Mat4,
        tint: [f32; 4],
        object_id: u32,
        outline_weight: f32,
        crease_sensitivity: f32,
    ) -> u32 {
        self.push_obj_uniform_skinned(
            device,
            idx,
            model,
            tint,
            object_id,
            outline_weight,
            crease_sensitivity,
            [0.0; 4],
            [0.0; 4],
        )
    }

    fn push_obj_uniform_skinned(
        &mut self,
        device: &wgpu::Device,
        idx: usize,
        model: Mat4,
        tint: [f32; 4],
        object_id: u32,
        outline_weight: f32,
        crease_sensitivity: f32,
        skin_pose_r: [f32; 4],
        skin_pose_l: [f32; 4],
    ) -> u32 {
        if idx >= self.obj_capacity {
            let new_cap = (idx + 512).next_power_of_two();
            self.obj_buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("object_uniforms"),
                size: (self.obj_align * new_cap) as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.obj_bg = Self::make_obj_bg(device, &self.obj_bgl, &self.obj_buf);
            self.obj_staging.resize(self.obj_align * new_cap, 0);
            self.obj_capacity = new_cap;
        }
        let normal_mat = model.inverse().transpose();
        let u = ObjectUniforms {
            model: model.to_cols_array_2d(),
            normal_mat: normal_mat.to_cols_array_2d(),
            tint,
            meta: [
                (object_id.min(255) as f32) / 255.0,
                outline_weight,
                crease_sensitivity,
                0.0,
            ],
            skin_pose_r,
            skin_pose_l,
        };
        let off = idx * self.obj_align;
        let bytes = bytemuck::bytes_of(&u);
        self.obj_staging[off..off + bytes.len()].copy_from_slice(bytes);
        off as u32
    }

    pub fn render_frame(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        target_view: &wgpu::TextureView,
        target_width: u32,
        target_height: u32,
        biome: BiomeId,
        spline: &RouteSpline,
        geo: &StaticWorldGeometry,
        player_dist_m: f32,
        player_lat_m: f32,
        player_wheel_rad: f32,
        player_crank_rad: f32,
        player_lean_rad: f32,
        player_profile: &RiderProfile,
        ai_riders: &[AiRider],
        camera_mode: CameraMode,
        graphics: &GraphicsSettings,
        time_s: f32,
        egui_jobs: Option<(&[egui::epaint::ClippedPrimitive], &egui_wgpu::ScreenDescriptor)>,
    ) {
        self.resize(device, target_width, target_height);

        let is_menu = camera_mode == CameraMode::MainMenu;
        let eff_player_lat_m = if is_menu {
            match biome {
                BiomeId::CoastalPromenade => 2.15,
                BiomeId::MountainClimb => 0.72,
                BiomeId::RollingRuralGravel => 0.78,
            }
        } else {
            player_lat_m
        };

        let player_sample = spline.sample(player_dist_m);
        let player_pos = player_sample.pos + player_sample.right * eff_player_lat_m;
        let forward = player_sample.tangent;
        let right = player_sample.right;
        let up = player_sample.up;

        let (cam_pos, cam_target, base_fov_deg) = match camera_mode {
            CameraMode::MainMenu => (
                player_pos - forward * 4.55 - right * 0.42 + up * 1.62,
                player_pos + forward * 19.5 - right * 0.48 + up * 1.46,
                47.0_f32,
            ),
            CameraMode::Chase => (
                player_pos - forward * 4.45 + up * 1.94 + right * 0.02,
                player_pos + forward * 15.5 + up * 0.85,
                51.0_f32,
            ),
            CameraMode::Close => (
                player_pos - forward * 2.55 + up * 1.48 + right * 0.45,
                player_pos + forward * 8.0 + up * 1.02,
                52.0,
            ),
            CameraMode::Wide => (
                player_pos - forward * 5.6 + up * 2.75 - right * 0.5,
                player_pos + forward * 22.0 + up * 1.2,
                56.0,
            ),
            CameraMode::Side => (
                player_pos + right * 2.75 + up * 1.16 - forward * 0.10,
                player_pos + up * 0.96,
                46.0,
            ),
            CameraMode::TurnaroundBack => (
                player_pos - forward * 3.10 + up * 1.15,
                player_pos + up * 0.94,
                44.0,
            ),
            CameraMode::TurnaroundBack34 => (
                player_pos - forward * 2.25 + right * 2.15 + up * 1.25,
                player_pos + up * 0.94,
                45.0,
            ),
            CameraMode::TurnaroundFront34 => (
                player_pos + forward * 2.35 + right * 1.75 + up * 1.22,
                player_pos + up * 0.94,
                45.0,
            ),
            CameraMode::FirstPerson => (
                player_pos + up * 1.58 + forward * 0.25,
                player_pos + forward * 22.0 + up * 1.25,
                64.0,
            ),
        };
        let fov_deg = (base_fov_deg + graphics.fov_offset_deg).clamp(32.0, 80.0);

        let aspect = target_width.max(1) as f32 / target_height.max(1) as f32;
        let near = 0.25_f32;
        let far = 1900.0_f32;
        let view_mat = Mat4::look_at_rh(cam_pos, cam_target, Vec3::Y);
        let proj_mat = Mat4::perspective_rh(fov_deg.to_radians(), aspect, near, far);
        let view_proj = proj_mat * view_mat;
        let inv_view_proj = view_proj.inverse();

        // Tight directional shadow frustum around player (casts crisp spoked wheel shadows!)
        let sun_dir = geo.hero_sun_dir;
        let shadow_center = if is_menu {
            player_pos
        } else {
            player_pos + forward * 4.0
        };
        let light_pos = shadow_center + sun_dir * 45.0;
        let light_view = Mat4::look_at_rh(light_pos, shadow_center, Vec3::Y);
        let r_ortho = 24.0_f32;
        let light_proj = Mat4::orthographic_rh(-r_ortho, r_ortho, -r_ortho, r_ortho, 1.0, 110.0);
        let light_view_proj = light_proj * light_view;

        // Biome atmospheric palette
        let (sun_col, sky_zen, sky_hor, fog_col, fog_dens, disc_str) = match biome {
            BiomeId::MountainClimb => (
                [1.00, 0.91, 0.72, near],
                [0.64, 0.72, 0.79, far],
                [0.95, 0.88, 0.75, if self.output_format.is_srgb() { 1.0 } else { 0.0 }],
                [0.86, 0.84, 0.77, 0.0024],
                0.0024,
                0.0,
            ),
            BiomeId::CoastalPromenade => (
                [1.00, 0.86, 0.60, near],
                [0.64, 0.69, 0.76, far],
                [0.99, 0.80, 0.56, if self.output_format.is_srgb() { 1.0 } else { 0.0 }],
                [0.94, 0.80, 0.63, 0.0016],
                0.0016,
                1.0,
            ),
            BiomeId::RollingRuralGravel => (
                [1.00, 0.90, 0.70, near],
                [0.62, 0.73, 0.81, far],
                [0.95, 0.89, 0.76, if self.output_format.is_srgb() { 1.0 } else { 0.0 }],
                [0.88, 0.85, 0.76, 0.0020],
                0.0020,
                0.0,
            ),
        };
        let _ = fog_dens;

        let frame_u = FrameUniforms {
            view_proj: view_proj.to_cols_array_2d(),
            inv_view_proj: inv_view_proj.to_cols_array_2d(),
            view_mat: view_mat.to_cols_array_2d(),
            light_view_proj: light_view_proj.to_cols_array_2d(),
            camera_pos: [cam_pos.x, cam_pos.y, cam_pos.z, time_s],
            sun_dir: [sun_dir.x, sun_dir.y, sun_dir.z, disc_str],
            sun_color: sun_col,
            sky_zenith: sky_zen,
            sky_horizon: sky_hor,
            fog_color: fog_col,
            screen_params: [
                target_width as f32,
                target_height as f32,
                1.0 / target_width.max(1) as f32,
                1.0 / target_height.max(1) as f32,
            ],
            outline_params: [
                0.036,
                0.85 * graphics.outline_scale.clamp(0.0, 2.0),
                0.80,
                0.003,
            ],
        };
        queue.write_buffer(&self.frame_buf, 0, bytemuck::bytes_of(&frame_u));

        // Gather draw list (static world, visible props, riders)
        let mut draw_count = 0usize;
        let mut scene_draws = Vec::with_capacity(2048);
        let mut shadow_draws = Vec::with_capacity(256);

        // Road, terrain, wires
        if self.world_bufs.road.is_some() {
            let off = self.push_obj_uniform(device, draw_count, Mat4::IDENTITY, [1.0; 4], 1, 0.35, 0.0);
            draw_count += 1;
            scene_draws.push(DrawItem { offset_bytes: off, model_id: None, section_idx: 0, world_kind: 1 });
        }
        if self.world_bufs.terrain.is_some() {
            let off = self.push_obj_uniform(device, draw_count, Mat4::IDENTITY, [1.0; 4], 2, 0.68, 0.0);
            draw_count += 1;
            scene_draws.push(DrawItem { offset_bytes: off, model_id: None, section_idx: 0, world_kind: 2 });
        }
        if self.world_bufs.wire.is_some() {
            let off = self.push_obj_uniform(device, draw_count, Mat4::IDENTITY, [1.0; 4], 3, 0.55, 0.0);
            draw_count += 1;
            scene_draws.push(DrawItem { offset_bytes: off, model_id: None, section_idx: 0, world_kind: 3 });
        }
        let water_off = if self.world_bufs.water.is_some() {
            let off = self.push_obj_uniform(device, draw_count, Mat4::IDENTITY, [1.0; 4], 4, 0.0, 0.0);
            draw_count += 1;
            Some(off)
        } else {
            None
        };

        // Visible props (distance-culled)
        let max_prop_dist = graphics.prop_draw_distance_m.clamp(150.0, 1200.0);
        for p in &geo.props {
            let pos = p.transform.w_axis.truncate();
            let dist_cam = pos.distance(cam_pos);
            if dist_cam > max_prop_dist {
                continue;
            }
            let num_secs = self.models.get(&p.model).map(|m| m.sections.len()).unwrap_or(0);
            // Keep internal crease lines strictly off organic/rock/foliage geometry (silhouette-only!)
            // and allow subtle architectural corner creases only on crisp man-made structures.
            let crease_s = match p.model {
                ModelId::ColonialTownhouse
                | ModelId::CoastalBuildingTall
                | ModelId::CoastalBuildingMid
                | ModelId::RanchHouse => 0.45,
                _ => 0.0,
            };
            let off = self.push_obj_uniform(
                device,
                draw_count,
                p.transform,
                p.tint,
                p.object_id,
                p.outline_weight,
                crease_s,
            );
            draw_count += 1;
            let in_shadow = graphics.shadows_enabled && p.cast_shadow && pos.distance(shadow_center) < 34.0;
            for si in 0..num_secs {
                let item = DrawItem {
                    offset_bytes: off,
                    model_id: Some(p.model),
                    section_idx: si,
                    world_kind: 0,
                };
                if in_shadow {
                    shadow_draws.push(item.clone());
                }
                scene_draws.push(item);
            }
        }

        // Riders (Player + AI peloton when in game)
        let mut rider_parts: Vec<RiderPartDraw> = Vec::with_capacity(96);
        let player_pose = if is_menu {
            RiderPoseMode::MenuStopped
        } else {
            RiderPoseMode::Pedaling
        };
        build_rider_draw_parts(
            spline,
            player_dist_m,
            eff_player_lat_m,
            player_wheel_rad,
            player_crank_rad,
            player_lean_rad,
            time_s,
            player_pose,
            player_profile,
            230,
            &mut rider_parts,
        );
        if !is_menu {
            for (ai_idx, ai) in ai_riders.iter().enumerate() {
                let s = spline.sample(ai.distance_m);
                if s.pos.distance(cam_pos) < 180.0 {
                    let base_id = 210 - (ai_idx as u32) * 5;
                    build_rider_draw_parts(
                        spline,
                        ai.distance_m,
                        ai.lateral_m,
                        ai.wheel_angle_rad,
                        ai.crank_angle_rad,
                        ai.lean_angle_rad,
                        time_s,
                        RiderPoseMode::Pedaling,
                        &ai.profile,
                        base_id,
                        &mut rider_parts,
                    );
                }
            }
        }

        for rp in &rider_parts {
            let sec_roles: Vec<MaterialRole> = self
                .models
                .get(&rp.model)
                .map(|m| m.sections.iter().map(|s| s.role).collect())
                .unwrap_or_default();
            let pos = rp.transform.w_axis.truncate();
            let in_shadow = graphics.shadows_enabled && pos.distance(shadow_center) < 34.0;
            let (outline_w, crease_s) = match rp.model {
                ModelId::BikeWheel => (0.72, 0.0),
                ModelId::BikeFrame | ModelId::BikeCrank => (0.86, 0.0),
                _ => (0.92, 0.0),
            };
            for (si, role) in sec_roles.into_iter().enumerate() {
                let tint = match role {
                    MaterialRole::BikeFrame => rp.frame_color,
                    MaterialRole::JerseyPrimary => rp.jersey_primary,
                    MaterialRole::JerseySecondary => rp.jersey_secondary,
                    MaterialRole::JerseyAccent => rp.jersey_accent,
                    MaterialRole::Helmet => rp.helmet_color,
                    MaterialRole::HelmetAccent => rp.helmet_accent,
                    _ => [1.0, 1.0, 1.0, 1.0],
                };
                let off = self.push_obj_uniform_skinned(
                    device,
                    draw_count,
                    rp.transform,
                    tint,
                    rp.object_id,
                    outline_w,
                    crease_s,
                    rp.skin_pose_r,
                    rp.skin_pose_l,
                );
                draw_count += 1;
                let item = DrawItem {
                    offset_bytes: off,
                    model_id: Some(rp.model),
                    section_idx: si,
                    world_kind: 0,
                };
                if in_shadow {
                    shadow_draws.push(item.clone());
                }
                scene_draws.push(item);
            }
        }

        let write_bytes = draw_count * self.obj_align;
        if write_bytes > 0 {
            queue.write_buffer(&self.obj_buf, 0, &self.obj_staging[..write_bytes]);
        }

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("cadence_encoder"),
        });

        // Pass 1: Shadow Map (2048x2048)
        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("shadow_pass"),
                color_attachments: &[],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.shadow_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            rpass.set_pipeline(&self.shadow_pipeline);
            rpass.set_bind_group(0, &self.frame_bg, &[]);
            for item in &shadow_draws {
                if let Some(mid) = item.model_id {
                    let sec = &self.models[&mid].sections[item.section_idx];
                    rpass.set_bind_group(1, &self.obj_bg, &[item.offset_bytes]);
                    rpass.set_vertex_buffer(0, sec.vertex_buf.slice(..));
                    rpass.set_index_buffer(sec.index_buf.slice(..), wgpu::IndexFormat::Uint32);
                    rpass.draw_indexed(0..sec.index_count, 0, 0..1);
                }
            }
        }

        // Pass 2 + 3 + 4: Sky -> Scene -> Water into MRT G-Buffer
        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("gbuffer_pass"),
                color_attachments: &[
                    Some(wgpu::RenderPassColorAttachment {
                        view: &self.gbuf.lit_view,
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: wgpu::StoreOp::Store,
                        },
                    }),
                    Some(wgpu::RenderPassColorAttachment {
                        view: &self.gbuf.normal_view,
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: wgpu::StoreOp::Store,
                        },
                    }),
                    Some(wgpu::RenderPassColorAttachment {
                        view: &self.gbuf.depth_meta_view,
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: wgpu::StoreOp::Store,
                        },
                    }),
                ],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.gbuf.scene_depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });

            // Sky fullscreen
            rpass.set_pipeline(&self.sky_pipeline);
            rpass.set_bind_group(0, &self.frame_bg, &[]);
            rpass.draw(0..3, 0..1);

            // Scene geometry (road, terrain, wire, props, riders)
            rpass.set_pipeline(&self.scene_pipeline);
            rpass.set_bind_group(0, &self.scene_frame_bg, &[]);
            for item in &scene_draws {
                rpass.set_bind_group(1, &self.obj_bg, &[item.offset_bytes]);
                match item.world_kind {
                    1 => {
                        if let Some((vb, ib, count)) = &self.world_bufs.road {
                            rpass.set_vertex_buffer(0, vb.slice(..));
                            rpass.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
                            rpass.draw_indexed(0..*count, 0, 0..1);
                        }
                    }
                    2 => {
                        if let Some((vb, ib, count)) = &self.world_bufs.terrain {
                            rpass.set_vertex_buffer(0, vb.slice(..));
                            rpass.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
                            rpass.draw_indexed(0..*count, 0, 0..1);
                        }
                    }
                    3 => {
                        if let Some((vb, ib, count)) = &self.world_bufs.wire {
                            rpass.set_vertex_buffer(0, vb.slice(..));
                            rpass.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
                            rpass.draw_indexed(0..*count, 0, 0..1);
                        }
                    }
                    _ => {
                        if let Some(mid) = item.model_id {
                            let sec = &self.models[&mid].sections[item.section_idx];
                            rpass.set_vertex_buffer(0, sec.vertex_buf.slice(..));
                            rpass.set_index_buffer(sec.index_buf.slice(..), wgpu::IndexFormat::Uint32);
                            rpass.draw_indexed(0..sec.index_count, 0, 0..1);
                        }
                    }
                }
            }

            // Water pass
            if let (Some(w_off), Some((vb, ib, count))) = (water_off, &self.world_bufs.water) {
                rpass.set_pipeline(&self.water_pipeline);
                rpass.set_bind_group(0, &self.frame_bg, &[]);
                rpass.set_bind_group(1, &self.obj_bg, &[w_off]);
                rpass.set_vertex_buffer(0, vb.slice(..));
                rpass.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
                rpass.draw_indexed(0..*count, 0, 0..1);
            }
        }

        // Pass 5: Fullscreen Ligne Claire Ink Outline Post-Process -> Target Surface
        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("post_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            rpass.set_pipeline(&self.post_pipeline);
            rpass.set_bind_group(0, &self.gbuf.post_bind_group, &[]);
            rpass.draw(0..3, 0..1);
        }

        // Pass 6: Stylized egui HUD Overlay
        if let Some((paint_jobs, screen_desc)) = egui_jobs {
            let mut extra_cmds = self.egui_renderer.update_buffers(
                device,
                queue,
                &mut encoder,
                paint_jobs,
                screen_desc,
            );
            {
                let rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("egui_hud_pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: target_view,
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });
                let mut rpass_static = rpass.forget_lifetime();
                self.egui_renderer.render(&mut rpass_static, paint_jobs, screen_desc);
            }
            extra_cmds.push(encoder.finish());
            queue.submit(extra_cmds);
        } else {
            queue.submit(std::iter::once(encoder.finish()));
        }

        let _ = (
            &self.shadow_tex,
            &self.gbuf.lit_tex,
            &self.gbuf.normal_tex,
            &self.gbuf.depth_meta_tex,
            &self.gbuf.scene_depth_tex,
            &self.frame_bgl,
            &self.scene_frame_bgl,
        );
    }
}
