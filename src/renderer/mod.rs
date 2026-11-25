use std::collections::HashMap;
use std::num::NonZeroU64;

use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3};
use wgpu::util::DeviceExt;

use crate::assets::{f32_to_f16, AssetBank, MaterialRole, ModelId, SceneVertex, SkeletonInfo, SkinWeightVertex};
use crate::world::riders::{
    build_animal_draw_parts, build_rider_draw_parts, AiRider, DynamicAnimal, RiderPartDraw,
    RiderPoseMode, RiderProfile, MAX_CYCLIST_BONES,
};
use crate::world::terrain::{Frustum, StaticWorldGeometry};
use crate::world::{BiomeId, RouteSpline};

pub const SHADOW_MAP_SIZE: u32 = 2048;
pub const NUM_SHADOW_CASCADES: usize = 3;
pub const MAX_SKINNED_VERTS: usize = 524288;
pub const MAX_SKIN_JOBS: usize = 512;
pub const MAX_SKIN_BONES: usize = 2048;
pub const MAX_INDIRECT_COMMANDS: usize = 16384;

#[derive(Debug, Clone, Copy)]
pub struct GraphicsSettings {
    pub outline_scale: f32,
    pub prop_draw_distance_m: f32,
    pub fov_offset_deg: f32,
    pub shadows_enabled: bool,
    pub taa_enabled: bool,
}

impl Default for GraphicsSettings {
    fn default() -> Self {
        Self {
            outline_scale: 1.0,
            prop_draw_distance_m: 560.0,
            fov_offset_deg: 0.0,
            shadows_enabled: true,
            taa_enabled: true,
        }
    }
}

fn halton_value(mut index: u32, base: u32) -> f32 {
    let mut f = 1.0_f32;
    let mut r = 0.0_f32;
    let b = base as f32;
    while index > 0 {
        f /= b;
        r += f * ((index % base) as f32);
        index /= base;
    }
    r
}

/// 8-sample Halton(2, 3) sub-pixel camera jitter offset in pixels `[-0.38, +0.38]`.
pub fn halton_2_3_jitter_px(frame_index: u32) -> [f32; 2] {
    let idx = (frame_index % 8) + 1;
    [
        (halton_value(idx, 2) - 0.5) * 0.76,
        (halton_value(idx, 3) - 0.5) * 0.76,
    ]
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
    pub light_view_proj_c1: [[f32; 4]; 4],
    pub light_view_proj_c2: [[f32; 4]; 4],
    pub cascade_splits: [f32; 4],
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct TaaUniforms {
    pub inv_view_proj: [[f32; 4]; 4],
    pub prev_view_proj: [[f32; 4]; 4],
    pub screen_params: [f32; 4],
    pub jitter_params: [f32; 4],
    pub taa_params: [f32; 4],
    pub camera_params: [f32; 4],
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

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct SkinDispatchParams {
    pub job_count: u32,
    pub max_verts_per_job: u32,
    pub _pad0: u32,
    pub _pad1: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct SkinJob {
    pub skin_pose_r: [f32; 4],
    pub skin_pose_l: [f32; 4],
    pub src_offset: u32,
    pub dst_offset: u32,
    pub vertex_count: u32,
    pub bone_offset: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct SinglePassCullUniforms {
    pub frustum_planes: [[f32; 4]; 6],
    pub filter_params: [f32; 4], // color pass: [cam.x, cam.y, cam.z, max_draw_dist]; shadow pass: [min_caster_r, max_caster_r, shadows_enabled, 0.0]
    pub sector_mask: [u32; 4],
    pub cmd_info: [u32; 4],      // [command_count, is_shadow_pass (0 = color view, 1 = CSM cascade), 0, 0]
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct GpuDrawDesc {
    pub pos_radius: [f32; 4],
    pub index_count: u32,
    pub first_index: u32,
    pub base_vertex: i32,
    pub flags: u32, // bits 0..4: sector_id (0..15 or 31 for riders), bit 8: cast_shadow, bit 9: is_sector_surface
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct DrawIndexedIndirectCmd {
    pub index_count: u32,
    pub instance_count: u32,
    pub first_index: u32,
    pub base_vertex: i32,
    pub first_instance: u32,
}

#[derive(Clone, Copy)]
struct GpuMeshSection {
    base_vertex: i32,
    first_index: u32,
    index_count: u32,
    vertex_count: u32,
    skin_src_offset: u32,
    role: MaterialRole,
    #[allow(dead_code)]
    base_color: [f32; 4],
}

struct GpuModel {
    sections: Vec<GpuMeshSection>,
    bounding_radius: f32,
}

struct GpuWorldOffsets {
    water_base_vertex: i32,
    water_first_index: u32,
    water_index_count: u32,
    skinned_base_vertex: u32,
}

struct GBufferTargets {
    width: u32,
    height: u32,
    lit_tex: wgpu::Texture,
    lit_view: wgpu::TextureView,
    normal_tex: wgpu::Texture,
    normal_view: wgpu::TextureView,
    scene_depth_tex: wgpu::Texture,
    scene_depth_view: wgpu::TextureView,
    post_color_tex: wgpu::Texture,
    post_color_view: wgpu::TextureView,
    taa_history_tex: [wgpu::Texture; 2],
    taa_history_views: [wgpu::TextureView; 2],
    post_bind_group: wgpu::BindGroup,
    taa_bind_groups: [wgpu::BindGroup; 2],
}

#[derive(Clone, Copy)]
struct EnqueuedCommandRef {
    cmd_slot: usize,
    offset_bytes: u32,
    shadow_candidate: bool,
}

#[derive(Clone, Copy)]
struct StaticSectorCmd {
    sector_id: usize,
    cmd_slot: usize,
    uniform_idx: usize, // 0 = road, 1 = terrain, 2 = wire, 3 = baked_prop_tier
    shadow_candidate: bool,
    caster_tier: u8,    // 0 = none, 1 = C0 only, 2 = C0..C1, 3 = C0..C2
}

pub struct LigneClaireRenderer {
    pub output_format: wgpu::TextureFormat,
    models: HashMap<ModelId, GpuModel>,
    static_model_verts: Vec<SceneVertex>,
    static_model_indices: Vec<u32>,
    static_draw_descs: Vec<GpuDrawDesc>,
    static_sector_cmds: Vec<StaticSectorCmd>,
    static_descs_dirty: bool,

    mega_vb: wgpu::Buffer,
    mega_ib: wgpu::Buffer,
    world_offsets: GpuWorldOffsets,

    // Pre-render Compute Skinning resources (procedural quadrupeds/bipeds + true 4-bone glTF skeletal cyclists)
    skin_src_vb: wgpu::Buffer,
    skin_weights_buf: wgpu::Buffer,
    skin_bones_buf: wgpu::Buffer,
    cyclist_skeleton: Option<SkeletonInfo>,
    skin_params_buf: wgpu::Buffer,
    skin_jobs_buf: wgpu::Buffer,
    skin_bgl: wgpu::BindGroupLayout,
    skin_bg: wgpu::BindGroup,
    skin_pipeline: wgpu::ComputePipeline,

    // Sector Queue & Multi-Dispatch GPU Indirect Command Generation resources
    // (1 dispatch + indirect draw arg buffer for Color View, 1 dispatch + indirect draw arg buffer per CSM Cascade)
    cam_cull_uniform_buf: wgpu::Buffer,
    csm_cull_uniform_bufs: [wgpu::Buffer; NUM_SHADOW_CASCADES],
    draw_desc_buf: wgpu::Buffer,
    cam_indirect_buf: wgpu::Buffer,
    csm_indirect_bufs: [wgpu::Buffer; NUM_SHADOW_CASCADES],
    cull_bgl: wgpu::BindGroupLayout,
    cam_cull_bg: wgpu::BindGroup,
    csm_cull_bgs: [wgpu::Buffer; 0], // placeholder alignment
    csm_cull_bind_groups: [wgpu::BindGroup; NUM_SHADOW_CASCADES],
    cull_pipeline: wgpu::ComputePipeline,

    frame_buf: wgpu::Buffer,
    shadow_frame_bufs: [wgpu::Buffer; NUM_SHADOW_CASCADES],
    obj_align: usize,
    obj_buf: wgpu::Buffer,
    obj_capacity: usize,
    obj_staging: Vec<u8>,

    shadow_tex: wgpu::Texture,
    shadow_array_view: wgpu::TextureView,
    shadow_cascade_views: [wgpu::TextureView; NUM_SHADOW_CASCADES],
    shadow_sampler: wgpu::Sampler,
    gbuf_sampler: wgpu::Sampler,
    taa_linear_sampler: wgpu::Sampler,
    taa_buf: wgpu::Buffer,

    frame_bgl: wgpu::BindGroupLayout,
    post_bgl: wgpu::BindGroupLayout,
    taa_bgl: wgpu::BindGroupLayout,
    obj_bgl: wgpu::BindGroupLayout,

    frame_bg: wgpu::BindGroup,
    shadow_frame_bgs: [wgpu::BindGroup; NUM_SHADOW_CASCADES],
    obj_bg: wgpu::BindGroup,

    shadow_pipeline: wgpu::RenderPipeline,
    sky_pipeline: wgpu::RenderPipeline,
    gbuf_road_tarmac_pipeline: wgpu::RenderPipeline,
    gbuf_road_gravel_pipeline: wgpu::RenderPipeline,
    gbuf_terrain_natural_pipeline: wgpu::RenderPipeline,
    gbuf_terrain_coastal_pipeline: wgpu::RenderPipeline,
    gbuf_prop_pipeline: wgpu::RenderPipeline,
    gbuf_skinned_pipeline: wgpu::RenderPipeline,
    gbuf_water_pipeline: wgpu::RenderPipeline,
    post_pipeline: wgpu::RenderPipeline,
    taa_pipeline: wgpu::RenderPipeline,

    taa_frame_index: u32,
    taa_history_valid: bool,
    prev_unjittered_view_proj: Mat4,
    prev_jitter_uv: [f32; 2],
    prev_cam_pos: Vec3,
    last_camera_mode: Option<CameraMode>,
    last_biome: Option<BiomeId>,

    gbuf: GBufferTargets,
    pub egui_renderer: egui_wgpu::Renderer,
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
        let mut static_model_verts: Vec<SceneVertex> = Vec::new();
        let mut static_model_weights: Vec<SkinWeightVertex> = Vec::new();
        let mut static_model_indices: Vec<u32> = Vec::new();
        let cyclist_skeleton = assets
            .models
            .get(&ModelId::RiderTorso)
            .and_then(|m| m.skeleton.clone());

        for (&id, model) in &assets.models {
            let mut gpu_secs = Vec::new();
            for sec in &model.sections {
                if sec.vertices.is_empty() || sec.indices.is_empty() {
                    continue;
                }
                let base_vertex = static_model_verts.len() as i32;
                let skin_src_offset = static_model_verts.len() as u32;
                let vertex_count = sec.vertices.len() as u32;
                let first_index = static_model_indices.len() as u32;
                let index_count = sec.indices.len() as u32;

                static_model_verts.extend_from_slice(&sec.vertices);
                static_model_weights.extend_from_slice(&sec.skin_weights);
                static_model_indices.extend_from_slice(&sec.indices);

                gpu_secs.push(GpuMeshSection {
                    base_vertex,
                    first_index,
                    index_count,
                    vertex_count,
                    skin_src_offset,
                    role: sec.role,
                    base_color: sec.base_color,
                });
            }
            models.insert(
                id,
                GpuModel {
                    sections: gpu_secs,
                    bounding_radius: model.bounding_radius,
                },
            );
        }

        let skin_src_vb = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("skin_src_vb"),
            contents: bytemuck::cast_slice(&static_model_verts),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        let skin_weights_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("skin_weights_buf"),
            contents: bytemuck::cast_slice(&static_model_weights),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        let skin_bones_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("skin_bones_buf"),
            size: (std::mem::size_of::<[[f32; 4]; 4]>() * MAX_SKIN_BONES) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let initial_skinned_base = static_model_verts.len() as u32;
        let mut initial_vb_data = static_model_verts.clone();
        initial_vb_data.resize(static_model_verts.len() + MAX_SKINNED_VERTS, SceneVertex::zeroed());

        let mega_vb = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("mega_vb"),
            contents: bytemuck::cast_slice(&initial_vb_data),
            usage: wgpu::BufferUsages::VERTEX
                | wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST,
        });
        let mega_ib = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("mega_ib"),
            contents: bytemuck::cast_slice(&static_model_indices),
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
        });

        let world_offsets = GpuWorldOffsets {
            water_base_vertex: 0,
            water_first_index: 0,
            water_index_count: 0,
            skinned_base_vertex: initial_skinned_base,
        };

        // Pre-render Compute Skinning pipeline
        let skin_params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("skin_params_buf"),
            size: std::mem::size_of::<SkinDispatchParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let skin_jobs_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("skin_jobs_buf"),
            size: (std::mem::size_of::<SkinJob>() * MAX_SKIN_JOBS) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let skin_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("skin_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: NonZeroU64::new(
                            std::mem::size_of::<SkinDispatchParams>() as u64,
                        ),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: NonZeroU64::new(std::mem::size_of::<SkinJob>() as u64),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: NonZeroU64::new(
                            std::mem::size_of::<SceneVertex>() as u64
                        ),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: NonZeroU64::new(
                            std::mem::size_of::<SceneVertex>() as u64
                        ),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: NonZeroU64::new(
                            std::mem::size_of::<SkinWeightVertex>() as u64
                        ),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: NonZeroU64::new(
                            std::mem::size_of::<[[f32; 4]; 4]>() as u64
                        ),
                    },
                    count: None,
                },
            ],
        });
        let skin_bg = Self::make_skin_bg(
            device,
            &skin_bgl,
            &skin_params_buf,
            &skin_jobs_buf,
            &skin_src_vb,
            &mega_vb,
            &skin_weights_buf,
            &skin_bones_buf,
        );
        let skin_mod = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("skinning_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/skinning.wgsl").into()),
        });
        let skin_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("skin_pl"),
            bind_group_layouts: &[Some(&skin_bgl)],
            immediate_size: 0,
        });
        let skin_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("skin_pipeline"),
            layout: Some(&skin_pl),
            module: &skin_mod,
            entry_point: Some("cs_skin_vertices"),
            compilation_options: Default::default(),
            cache: None,
        });

        // Sector Queue & Multi-Dispatch GPU Indirect Command Generation compute pipeline
        // Separate uniform buffers and separate indirect draw argument buffers for Color View and each CSM Cascade
        let single_cull_u_size = std::mem::size_of::<SinglePassCullUniforms>() as u64;
        let single_indirect_buf_size =
            (std::mem::size_of::<DrawIndexedIndirectCmd>() * MAX_INDIRECT_COMMANDS) as u64;

        let cam_cull_uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cam_cull_uniform_buf"),
            size: single_cull_u_size,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let csm_cull_uniform_bufs = std::array::from_fn(|ci| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("csm_cull_uniform_buf_c{ci}")),
                size: single_cull_u_size,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        });
        let draw_desc_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("draw_desc_buf"),
            size: (std::mem::size_of::<GpuDrawDesc>() * MAX_INDIRECT_COMMANDS) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let cam_indirect_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cam_indirect_draw_buf"),
            size: single_indirect_buf_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::INDIRECT
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let csm_indirect_bufs = std::array::from_fn(|ci| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("csm_indirect_draw_buf_c{ci}")),
                size: single_indirect_buf_size,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::INDIRECT
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        });
        let cull_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("cull_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: NonZeroU64::new(single_cull_u_size),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: NonZeroU64::new(
                            std::mem::size_of::<GpuDrawDesc>() as u64
                        ),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: NonZeroU64::new(
                            std::mem::size_of::<DrawIndexedIndirectCmd>() as u64,
                        ),
                    },
                    count: None,
                },
            ],
        });
        let make_cull_bg = |label: &str, u_buf: &wgpu::Buffer, ind_buf: &wgpu::Buffer| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(label),
                layout: &cull_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: u_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: draw_desc_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: ind_buf.as_entire_binding(),
                    },
                ],
            })
        };
        let cam_cull_bg = make_cull_bg("cam_cull_bg", &cam_cull_uniform_buf, &cam_indirect_buf);
        let csm_cull_bind_groups = std::array::from_fn(|ci| {
            make_cull_bg(
                &format!("csm_cull_bg_c{ci}"),
                &csm_cull_uniform_bufs[ci],
                &csm_indirect_bufs[ci],
            )
        });
        let cull_mod = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("cull_indirect_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/cull_indirect.wgsl").into()),
        });
        let cull_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("cull_pl"),
            bind_group_layouts: &[Some(&cull_bgl)],
            immediate_size: 0,
        });
        let cull_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("cull_indirect_pipeline"),
            layout: Some(&cull_pl),
            module: &cull_mod,
            entry_point: Some("cs_build_indirect_commands"),
            compilation_options: Default::default(),
            cache: None,
        });

        let frame_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("frame_uniforms"),
            size: std::mem::size_of::<FrameUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let shadow_frame_bufs = std::array::from_fn(|ci| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("shadow_frame_uniforms_c{ci}")),
                size: std::mem::size_of::<FrameUniforms>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        });

        let obj_align = device.limits().min_uniform_buffer_offset_alignment.max(256) as usize;
        let obj_capacity = MAX_INDIRECT_COMMANDS;
        let obj_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("object_uniforms"),
            size: (obj_align * obj_capacity) as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let shadow_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("shadow_csm_array"),
            size: wgpu::Extent3d {
                width: SHADOW_MAP_SIZE,
                height: SHADOW_MAP_SIZE,
                depth_or_array_layers: NUM_SHADOW_CASCADES as u32,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let shadow_array_view = shadow_tex.create_view(&wgpu::TextureViewDescriptor {
            label: Some("shadow_csm_array_view"),
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            base_array_layer: 0,
            array_layer_count: Some(NUM_SHADOW_CASCADES as u32),
            ..Default::default()
        });
        let shadow_cascade_views = std::array::from_fn(|ci| {
            shadow_tex.create_view(&wgpu::TextureViewDescriptor {
                label: Some(&format!("shadow_cascade_view_{ci}")),
                dimension: Some(wgpu::TextureViewDimension::D2),
                base_array_layer: ci as u32,
                array_layer_count: Some(1),
                ..Default::default()
            })
        });

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
        let taa_linear_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("taa_linear_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let taa_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("taa_uniforms"),
            size: std::mem::size_of::<TaaUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
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

        // Deferred Lighting & Ligne Claire Outline Pass bind group layout:
        // Binds FrameUniforms, unlit G-Buffer Albedo+OutlineWeight, G-Buffer Normal+ID, Hardware Depth32Float, and 3-Cascade CSM Shadow Map!
        let post_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("post_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: NonZeroU64::new(
                            std::mem::size_of::<FrameUniforms>() as u64
                        ),
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
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2Array,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Comparison),
                    count: None,
                },
            ],
        });

        let taa_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("taa_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: NonZeroU64::new(
                            std::mem::size_of::<TaaUniforms>() as u64
                        ),
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
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
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
        let shadow_frame_bgs = std::array::from_fn(|ci| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(&format!("shadow_frame_bg_c{ci}")),
                layout: &frame_bgl,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: shadow_frame_bufs[ci].as_entire_binding(),
                }],
            })
        });

        let obj_bg = Self::make_obj_bg(device, &obj_bgl, &obj_buf);

        let common_wgsl = include_str!("../shaders/common.wgsl");
        let make_shader_with_common = |label: &str, domain_src: &str| {
            let full_src = format!("{common_wgsl}\n{domain_src}");
            device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(label),
                source: wgpu::ShaderSource::Wgsl(full_src.into()),
            })
        };

        let shadow_mod =
            make_shader_with_common("shadow_shader", include_str!("../shaders/shadow.wgsl"));
        let sky_mod = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("sky_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/sky.wgsl").into()),
        });
        let gbuf_road_mod =
            make_shader_with_common("gbuf_road_shader", include_str!("../shaders/gbuf_road.wgsl"));
        let gbuf_terrain_mod = make_shader_with_common(
            "gbuf_terrain_shader",
            include_str!("../shaders/gbuf_terrain.wgsl"),
        );
        let gbuf_prop_mod =
            make_shader_with_common("gbuf_prop_shader", include_str!("../shaders/gbuf_prop.wgsl"));
        let gbuf_skinned_mod = make_shader_with_common(
            "gbuf_skinned_shader",
            include_str!("../shaders/gbuf_skinned.wgsl"),
        );
        let gbuf_water_mod = make_shader_with_common(
            "gbuf_water_shader",
            include_str!("../shaders/gbuf_water.wgsl"),
        );
        let post_mod =
            make_shader_with_common("post_lighting_shader", include_str!("../shaders/post.wgsl"));
        let taa_mod = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("taa_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/taa.wgsl").into()),
        });

        let gbuf_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("gbuf_pl"),
            bind_group_layouts: &[Some(&frame_bgl), Some(&obj_bgl)],
            immediate_size: 0,
        });
        let shadow_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("shadow_pipeline"),
            layout: Some(&gbuf_pl),
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
                    constant: 1,
                    slope_scale: 0.35,
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

        let make_gbuf_domain_pipeline =
            |label: &str, module: &wgpu::ShaderModule, vs_entry: &str, fs_entry: &str| {
                device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some(label),
                    layout: Some(&gbuf_pl),
                    vertex: wgpu::VertexState {
                        module,
                        entry_point: Some(vs_entry),
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
                        module,
                        entry_point: Some(fs_entry),
                        compilation_options: Default::default(),
                        targets: &gbuf_targets,
                    }),
                    multiview_mask: None,
                    cache: None,
                })
            };

        let gbuf_road_tarmac_pipeline = make_gbuf_domain_pipeline(
            "gbuf_road_tarmac_pipeline",
            &gbuf_road_mod,
            "vs_gbuf_road",
            "fs_gbuf_road_tarmac",
        );
        let gbuf_road_gravel_pipeline = make_gbuf_domain_pipeline(
            "gbuf_road_gravel_pipeline",
            &gbuf_road_mod,
            "vs_gbuf_road",
            "fs_gbuf_road_gravel",
        );
        let gbuf_terrain_natural_pipeline = make_gbuf_domain_pipeline(
            "gbuf_terrain_natural_pipeline",
            &gbuf_terrain_mod,
            "vs_gbuf_terrain",
            "fs_gbuf_terrain_natural",
        );
        let gbuf_terrain_coastal_pipeline = make_gbuf_domain_pipeline(
            "gbuf_terrain_coastal_pipeline",
            &gbuf_terrain_mod,
            "vs_gbuf_terrain",
            "fs_gbuf_terrain_coastal",
        );
        let gbuf_prop_pipeline = make_gbuf_domain_pipeline(
            "gbuf_prop_pipeline",
            &gbuf_prop_mod,
            "vs_gbuf_prop",
            "fs_gbuf_prop",
        );
        let gbuf_skinned_pipeline = make_gbuf_domain_pipeline(
            "gbuf_skinned_pipeline",
            &gbuf_skinned_mod,
            "vs_gbuf_skinned",
            "fs_gbuf_skinned",
        );
        let gbuf_water_pipeline = make_gbuf_domain_pipeline(
            "gbuf_water_pipeline",
            &gbuf_water_mod,
            "vs_gbuf_water",
            "fs_gbuf_water",
        );

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
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });

        let taa_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("taa_pl"),
            bind_group_layouts: &[Some(&taa_bgl)],
            immediate_size: 0,
        });
        let taa_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("taa_pipeline"),
            layout: Some(&taa_pl),
            vertex: wgpu::VertexState {
                module: &taa_mod,
                entry_point: Some("vs_taa"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &taa_mod,
                entry_point: Some("fs_taa"),
                compilation_options: Default::default(),
                targets: &[
                    Some(wgpu::ColorTargetState {
                        format: output_format,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                    Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rgba8Unorm,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                ],
            }),
            multiview_mask: None,
            cache: None,
        });

        let gbuf = Self::create_gbuffer(
            device,
            width,
            height,
            &post_bgl,
            &taa_bgl,
            &frame_buf,
            &taa_buf,
            &gbuf_sampler,
            &taa_linear_sampler,
            &shadow_array_view,
            &shadow_sampler,
        );
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
            static_model_verts,
            static_model_indices,
            static_draw_descs: Vec::new(),
            static_sector_cmds: Vec::new(),
            static_descs_dirty: true,
            mega_vb,
            mega_ib,
            world_offsets,
            skin_src_vb,
            skin_weights_buf,
            skin_bones_buf,
            cyclist_skeleton,
            skin_params_buf,
            skin_jobs_buf,
            skin_bgl,
            skin_bg,
            skin_pipeline,
            cam_cull_uniform_buf,
            csm_cull_uniform_bufs,
            draw_desc_buf,
            cam_indirect_buf,
            csm_indirect_bufs,
            cull_bgl,
            cam_cull_bg,
            csm_cull_bgs: [],
            csm_cull_bind_groups,
            cull_pipeline,
            frame_buf,
            shadow_frame_bufs,
            obj_align,
            obj_buf,
            obj_capacity,
            obj_staging: vec![0u8; obj_align * obj_capacity],
            shadow_tex,
            shadow_array_view,
            shadow_cascade_views,
            shadow_sampler,
            gbuf_sampler,
            taa_linear_sampler,
            taa_buf,
            frame_bgl,
            post_bgl,
            taa_bgl,
            obj_bgl,
            frame_bg,
            shadow_frame_bgs,
            obj_bg,
            shadow_pipeline,
            sky_pipeline,
            gbuf_road_tarmac_pipeline,
            gbuf_road_gravel_pipeline,
            gbuf_terrain_natural_pipeline,
            gbuf_terrain_coastal_pipeline,
            gbuf_prop_pipeline,
            gbuf_skinned_pipeline,
            gbuf_water_pipeline,
            post_pipeline,
            taa_pipeline,
            taa_frame_index: 0,
            taa_history_valid: false,
            prev_unjittered_view_proj: Mat4::IDENTITY,
            prev_jitter_uv: [0.0; 2],
            prev_cam_pos: Vec3::ZERO,
            last_camera_mode: None,
            last_biome: None,
            gbuf,
            egui_renderer,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn make_skin_bg(
        device: &wgpu::Device,
        bgl: &wgpu::BindGroupLayout,
        params_buf: &wgpu::Buffer,
        jobs_buf: &wgpu::Buffer,
        src_vb: &wgpu::Buffer,
        dst_vb: &wgpu::Buffer,
        weights_buf: &wgpu::Buffer,
        bones_buf: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("skin_bg"),
            layout: bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: jobs_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: src_vb.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: dst_vb.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: weights_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: bones_buf.as_entire_binding(),
                },
            ],
        })
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

    #[allow(clippy::too_many_arguments)]
    fn create_gbuffer(
        device: &wgpu::Device,
        width: u32,
        height: u32,
        post_bgl: &wgpu::BindGroupLayout,
        taa_bgl: &wgpu::BindGroupLayout,
        frame_buf: &wgpu::Buffer,
        taa_buf: &wgpu::Buffer,
        gbuf_sampler: &wgpu::Sampler,
        taa_linear_sampler: &wgpu::Sampler,
        shadow_array_view: &wgpu::TextureView,
        shadow_sampler: &wgpu::Sampler,
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
        let lit_tex = mk_tex("gbuf_albedo", wgpu::TextureFormat::Rgba8Unorm);
        let normal_tex = mk_tex("gbuf_norm", wgpu::TextureFormat::Rgba8Unorm);
        let scene_depth_tex = mk_tex("gbuf_z", wgpu::TextureFormat::Depth32Float);
        let post_color_tex = mk_tex("post_color", wgpu::TextureFormat::Rgba8Unorm);
        let taa_history_tex = [
            mk_tex("taa_history_0", wgpu::TextureFormat::Rgba8Unorm),
            mk_tex("taa_history_1", wgpu::TextureFormat::Rgba8Unorm),
        ];

        let lit_view = lit_tex.create_view(&Default::default());
        let normal_view = normal_tex.create_view(&Default::default());
        let scene_depth_view = scene_depth_tex.create_view(&Default::default());
        let post_color_view = post_color_tex.create_view(&Default::default());
        let taa_history_views = [
            taa_history_tex[0].create_view(&Default::default()),
            taa_history_tex[1].create_view(&Default::default()),
        ];

        let post_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("post_lighting_bg"),
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
                    resource: wgpu::BindingResource::Sampler(gbuf_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(&scene_depth_view),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::TextureView(shadow_array_view),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: wgpu::BindingResource::Sampler(shadow_sampler),
                },
            ],
        });

        // Ping-pong TAA bind groups: pass i reads taa_history_views[1 - i] and renders into taa_history_views[i]
        let taa_bind_groups = std::array::from_fn(|curr_idx| {
            let prev_idx = 1 - curr_idx;
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(&format!("taa_bg_{curr_idx}")),
                layout: taa_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: taa_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&post_color_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(&taa_history_views[prev_idx]),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::TextureView(&scene_depth_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: wgpu::BindingResource::TextureView(&normal_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: wgpu::BindingResource::Sampler(taa_linear_sampler),
                    },
                ],
            })
        });

        GBufferTargets {
            width,
            height,
            lit_tex,
            lit_view,
            normal_tex,
            normal_view,
            scene_depth_tex,
            scene_depth_view,
            post_color_tex,
            post_color_view,
            taa_history_tex,
            taa_history_views,
            post_bind_group,
            taa_bind_groups,
        }
    }

    pub fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        let w = width.max(1);
        let h = height.max(1);
        if self.gbuf.width != w || self.gbuf.height != h {
            self.gbuf = Self::create_gbuffer(
                device,
                w,
                h,
                &self.post_bgl,
                &self.taa_bgl,
                &self.frame_buf,
                &self.taa_buf,
                &self.gbuf_sampler,
                &self.taa_linear_sampler,
                &self.shadow_array_view,
                &self.shadow_sampler,
            );
            self.taa_history_valid = false;
        }
    }

    pub fn upload_world(&mut self, device: &wgpu::Device, geo: &StaticWorldGeometry) {
        let mut all_verts = Vec::with_capacity(
            self.static_model_verts.len()
                + geo.road_vertices.len()
                + geo.terrain_vertices.len()
                + geo.wire_vertices.len()
                + geo.water_vertices.len()
                + geo.props.len() * 96
                + MAX_SKINNED_VERTS,
        );
        let mut all_indices = Vec::with_capacity(
            self.static_model_indices.len()
                + geo.road_indices.len()
                + geo.terrain_indices.len()
                + geo.wire_indices.len()
                + geo.water_indices.len()
                + geo.props.len() * 160,
        );

        all_verts.extend_from_slice(&self.static_model_verts);
        all_indices.extend_from_slice(&self.static_model_indices);

        let road_base_vertex = all_verts.len() as i32;
        let road_first_index = all_indices.len() as u32;
        all_verts.extend_from_slice(&geo.road_vertices);
        all_indices.extend_from_slice(&geo.road_indices);

        let terrain_base_vertex = all_verts.len() as i32;
        let terrain_first_index = all_indices.len() as u32;
        all_verts.extend_from_slice(&geo.terrain_vertices);
        all_indices.extend_from_slice(&geo.terrain_indices);

        let wire_base_vertex = all_verts.len() as i32;
        let wire_first_index = all_indices.len() as u32;
        all_verts.extend_from_slice(&geo.wire_vertices);
        all_indices.extend_from_slice(&geo.wire_indices);

        let water_base_vertex = all_verts.len() as i32;
        let water_first_index = all_indices.len() as u32;
        let water_index_count = geo.water_indices.len() as u32;
        all_verts.extend_from_slice(&geo.water_vertices);
        all_indices.extend_from_slice(&geo.water_indices);

        // Pre-bake static sector draw descriptors and per-sector CSM-size-tiered prop meshes
        let mut static_draw_descs: Vec<GpuDrawDesc> = Vec::with_capacity(geo.sectors.len() * 7);
        let mut static_sector_cmds: Vec<StaticSectorCmd> = Vec::with_capacity(geo.sectors.len() * 7);

        for (sid, sec) in geo.sectors.iter().enumerate() {
            let sec_flag = (sid as u32) | 512u32; // bit 9 = is_sector_surface
            if sec.road_index_count > 0 && static_draw_descs.len() < MAX_INDIRECT_COMMANDS {
                let slot = static_draw_descs.len();
                static_draw_descs.push(GpuDrawDesc {
                    pos_radius: [sec.center.x, sec.center.y, sec.center.z, sec.bounding_radius],
                    index_count: sec.road_index_count,
                    first_index: road_first_index + sec.road_index_offset,
                    base_vertex: road_base_vertex,
                    flags: sec_flag,
                });
                static_sector_cmds.push(StaticSectorCmd {
                    sector_id: sid,
                    cmd_slot: slot,
                    uniform_idx: 0,
                    shadow_candidate: false,
                    caster_tier: 0,
                });
            }
            if sec.terrain_index_count > 0 && static_draw_descs.len() < MAX_INDIRECT_COMMANDS {
                let slot = static_draw_descs.len();
                static_draw_descs.push(GpuDrawDesc {
                    pos_radius: [sec.center.x, sec.center.y, sec.center.z, sec.bounding_radius],
                    index_count: sec.terrain_index_count,
                    first_index: terrain_first_index + sec.terrain_index_offset,
                    base_vertex: terrain_base_vertex,
                    flags: sec_flag,
                });
                static_sector_cmds.push(StaticSectorCmd {
                    sector_id: sid,
                    cmd_slot: slot,
                    uniform_idx: 1,
                    shadow_candidate: false,
                    caster_tier: 0,
                });
            }
            if sec.wire_index_count > 0 && static_draw_descs.len() < MAX_INDIRECT_COMMANDS {
                let slot = static_draw_descs.len();
                static_draw_descs.push(GpuDrawDesc {
                    pos_radius: [sec.center.x, sec.center.y, sec.center.z, sec.bounding_radius],
                    index_count: sec.wire_index_count,
                    first_index: wire_first_index + sec.wire_index_offset,
                    base_vertex: wire_base_vertex,
                    flags: sec_flag,
                });
                static_sector_cmds.push(StaticSectorCmd {
                    sector_id: sid,
                    cmd_slot: slot,
                    uniform_idx: 2,
                    shadow_candidate: false,
                    caster_tier: 0,
                });
            }

            if !sec.prop_indices.is_empty() {
                let mut tier_verts: [Vec<SceneVertex>; 4] =
                    std::array::from_fn(|_| Vec::new());
                let mut tier_indices: [Vec<u32>; 4] =
                    std::array::from_fn(|_| Vec::new());
                let mut tier_min = [Vec3::splat(f32::INFINITY); 4];
                let mut tier_max = [Vec3::splat(f32::NEG_INFINITY); 4];

                for &pi in &sec.prop_indices {
                    let p = &geo.props[pi];
                    let Some(gpu_model) = self.models.get(&p.model) else {
                        continue;
                    };
                    let sx = p.transform.x_axis.truncate().length();
                    let sy = p.transform.y_axis.truncate().length();
                    let sz = p.transform.z_axis.truncate().length();
                    let max_scale = sx.max(sy).max(sz).max(0.01);
                    let world_radius = gpu_model.bounding_radius * max_scale;

                    let tier_idx = if !p.cast_shadow {
                        0
                    } else if world_radius < 0.90 {
                        1
                    } else if world_radius < 2.50 {
                        2
                    } else {
                        3
                    };
                    let has_crease = matches!(
                        p.model,
                        ModelId::ColonialTownhouse
                            | ModelId::CoastalBuildingTall
                            | ModelId::CoastalBuildingMid
                            | ModelId::RamblaBuilding(_)
                            | ModelId::RanchHouse
                    );
                    let normal_mat = glam::Mat3::from_mat4(p.transform).inverse().transpose();
                    let base_pos = p.transform.w_axis.truncate();
                    // Unique non-periodic per-instance seed in [0.5, 250.0] derived from prop index & world coordinates
                    let raw_seed = ((pi as u32).wrapping_mul(747796405).wrapping_add(2891336453))
                        ^ (base_pos.x.to_bits().wrapping_mul(1597334677))
                        ^ (base_pos.z.to_bits().wrapping_mul(3812015801));
                    let inst_seed_f32 =
                        0.5 + ((raw_seed & 0x00FF_FFFF) as f32 / 16_777_215.0) * 249.5;
                    let inst_seed_f16 = f32_to_f16(inst_seed_f32);
                    let scale_factor = max_scale.sqrt().clamp(0.65, 1.65);

                    for ms in &gpu_model.sections {
                        let local_v_base = tier_verts[tier_idx].len() as u32;
                        let v_start = ms.base_vertex as usize;
                        let v_end = v_start + (ms.vertex_count as usize);
                        for sv in &self.static_model_verts[v_start..v_end] {
                            let wp = p.transform.transform_point3(Vec3::from_array(sv.pos));
                            tier_min[tier_idx] = tier_min[tier_idx].min(wp);
                            tier_max[tier_idx] = tier_max[tier_idx].max(wp);
                            let ln = Vec3::new(
                                (sv.normal_mat[0] as f32) / 127.0,
                                (sv.normal_mat[1] as f32) / 127.0,
                                (sv.normal_mat[2] as f32) / 127.0,
                            );
                            let wn = (normal_mat * ln).normalize_or_zero();
                            let mat_type = (sv.normal_mat[3].max(0) as u8) & 7;
                            let rgb = [
                                ((sv.color[0] as f32) * p.tint[0]).clamp(0.0, 255.0).round() as u8,
                                ((sv.color[1] as f32) * p.tint[1]).clamp(0.0, 255.0).round() as u8,
                                ((sv.color[2] as f32) * p.tint[2]).clamp(0.0, 255.0).round() as u8,
                            ];

                            let local_y = sv.pos[1].max(0.0);
                            let r_xz = (sv.pos[0] * sv.pos[0] + sv.pos[2] * sv.pos[2]).sqrt();
                            let _ = inst_seed_f16;
                            // Separate biomechanical wind response:
                            // - stem_flex (uv.x): Coherent whole-plant cantilever sway in the general wind direction (depends only on height local_y)
                            // - leaf_turb (uv.y): High-frequency leaf/blade turbulence (0 on stiff trunks, stems, and Cortaderia reed stalks/plumes; >0 on leaves/blades)
                            let (stem_flex, leaf_turb) = match p.model {
                                ModelId::TreeCanopy
                                | ModelId::TreePine
                                | ModelId::TreeEucalyptus(_)
                                | ModelId::TreeOmbu(_)
                                | ModelId::TreeCeibo(_) => {
                                    let h_norm = ((local_y - 0.30).max(0.0) / 4.5).clamp(0.0, 1.35);
                                    let stem = 0.075 * h_norm.powf(1.35) * scale_factor;
                                    let leaf = if mat_type == 2 {
                                        let out = (r_xz / 1.8).clamp(0.0, 1.2);
                                        0.042 * h_norm * (0.35 + 0.65 * out) * scale_factor
                                    } else {
                                        0.0
                                    };
                                    (stem, leaf)
                                }
                                ModelId::PalmTree | ModelId::SmallPalm(_) => {
                                    let h_norm = ((local_y - 0.12).max(0.0) / 2.0).clamp(0.0, 1.35);
                                    let frond_out = ((r_xz - 0.18).max(0.0) / 1.15).clamp(0.0, 1.3);
                                    let stem = 0.055 * h_norm.powf(1.35) * scale_factor;
                                    let leaf = 0.048 * frond_out * frond_out * h_norm.sqrt() * scale_factor;
                                    (stem, leaf)
                                }
                                ModelId::PampasCortadera(_) => {
                                    // Cortaderia selloana is structurally stiff: whole plant leans gently as a cantilever beam;
                                    // only the flexible outer grass blades (mat_type == 2) have leaf turbulence!
                                    let h_norm = (local_y / 1.85).clamp(0.0, 1.25);
                                    let stem = 0.036 * h_norm.powf(1.35) * scale_factor;
                                    let leaf = if mat_type == 2 {
                                        let blade_out = ((r_xz - 0.14).max(0.0) / 0.65).clamp(0.0, 1.2);
                                        0.038 * blade_out * blade_out * scale_factor
                                    } else {
                                        0.0
                                    };
                                    (stem, leaf)
                                }
                                ModelId::ChircaBush(_)
                                | ModelId::CeiboBush(_)
                                | ModelId::BushCluster => {
                                    let h_norm = ((local_y - 0.03).max(0.0) / 1.10).clamp(0.0, 1.25);
                                    let out_norm = ((r_xz - 0.12).max(0.0) / 0.70).clamp(0.0, 1.25);
                                    let stem = 0.046 * h_norm.powf(1.35) * scale_factor;
                                    let leaf = 0.032 * out_norm * out_norm * h_norm * scale_factor;
                                    (stem, leaf)
                                }
                                ModelId::WildflowerTuft | ModelId::CampoGrass(_) => {
                                    let h_norm = (local_y / 0.34).clamp(0.0, 1.25);
                                    let out_norm = (r_xz / 0.28).clamp(0.2, 1.2);
                                    let stem = 0.030 * h_norm.powf(1.3) * scale_factor;
                                    let leaf = 0.022 * h_norm * out_norm * scale_factor;
                                    (stem, leaf)
                                }
                                ModelId::Flagpole => {
                                    if r_xz > 0.11 && local_y > 2.5 {
                                        let cloth_out = ((r_xz - 0.10) / 1.0).clamp(0.0, 1.2);
                                        (0.025 * cloth_out * scale_factor, 0.045 * cloth_out * cloth_out * scale_factor)
                                    } else {
                                        (0.0, 0.0)
                                    }
                                }
                                _ => (0.0, 0.0),
                            };
                            let packed_uv = if stem_flex > 0.0001 || leaf_turb > 0.0001 {
                                [f32_to_f16(stem_flex), f32_to_f16(leaf_turb)]
                            } else {
                                [0u16, 0u16]
                            };

                            let packed_v = SceneVertex::new_packed_prop(
                                wp.to_array(),
                                wn.to_array(),
                                packed_uv,
                                rgb,
                                mat_type,
                                p.object_id,
                                p.outline_weight,
                                has_crease,
                            );
                            tier_verts[tier_idx].push(packed_v);
                        }
                        let i_start = ms.first_index as usize;
                        let i_end = i_start + (ms.index_count as usize);
                        tier_indices[tier_idx].extend(
                            self.static_model_indices[i_start..i_end]
                                .iter()
                                .map(|&idx| local_v_base + idx),
                        );
                    }
                }

                for tier_idx in 0..4 {
                    if tier_indices[tier_idx].is_empty()
                        || static_draw_descs.len() >= MAX_INDIRECT_COMMANDS
                    {
                        continue;
                    }
                    let tier_base_vertex = all_verts.len() as i32;
                    let tier_first_index = all_indices.len() as u32;
                    let tier_index_count = tier_indices[tier_idx].len() as u32;
                    all_verts.extend_from_slice(&tier_verts[tier_idx]);
                    all_indices.extend_from_slice(&tier_indices[tier_idx]);

                    let t_center = (tier_min[tier_idx] + tier_max[tier_idx]) * 0.5;
                    let t_radius = (tier_max[tier_idx] - t_center).length().max(0.5);
                    let (cast_shadow, rep_caster_cm, caster_tier): (bool, u32, u8) = match tier_idx {
                        0 => (false, 0, 0),
                        1 => (true, 50, 1),  // 0.50 m -> Cascade 0 only
                        2 => (true, 160, 2), // 1.60 m -> Cascades 0 & 1
                        _ => (true, 600, 3), // 6.00 m -> Cascades 0, 1 & 2
                    };
                    let flags = (sid as u32)
                        | if cast_shadow { 256u32 } else { 0u32 }
                        | (rep_caster_cm << 16);
                    let slot = static_draw_descs.len();
                    static_draw_descs.push(GpuDrawDesc {
                        pos_radius: [t_center.x, t_center.y, t_center.z, t_radius],
                        index_count: tier_index_count,
                        first_index: tier_first_index,
                        base_vertex: tier_base_vertex,
                        flags,
                    });
                    static_sector_cmds.push(StaticSectorCmd {
                        sector_id: sid,
                        cmd_slot: slot,
                        uniform_idx: 3,
                        shadow_candidate: cast_shadow,
                        caster_tier,
                    });
                }
            }
        }

        // Group static sector commands by uniform_idx so each render pass binds at most 4 static uniform offsets
        static_sector_cmds.sort_by_key(|c| (c.uniform_idx, c.sector_id, c.cmd_slot));

        let skinned_base_vertex = all_verts.len() as u32;
        all_verts.resize(all_verts.len() + MAX_SKINNED_VERTS, SceneVertex::zeroed());

        self.mega_vb = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("mega_vb"),
            contents: bytemuck::cast_slice(&all_verts),
            usage: wgpu::BufferUsages::VERTEX
                | wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST,
        });
        self.mega_ib = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("mega_ib"),
            contents: bytemuck::cast_slice(&all_indices),
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
        });
        self.world_offsets = GpuWorldOffsets {
            water_base_vertex,
            water_first_index,
            water_index_count,
            skinned_base_vertex,
        };
        self.static_draw_descs = static_draw_descs;
        self.static_sector_cmds = static_sector_cmds;
        self.static_descs_dirty = true;
        self.taa_history_valid = false;
        self.skin_bg = Self::make_skin_bg(
            device,
            &self.skin_bgl,
            &self.skin_params_buf,
            &self.skin_jobs_buf,
            &self.skin_src_vb,
            &self.mega_vb,
            &self.skin_weights_buf,
            &self.skin_bones_buf,
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn push_obj_uniform(
        &mut self,
        device: &wgpu::Device,
        idx: usize,
        model: Mat4,
        tint: [f32; 4],
        object_id: u32,
        outline_weight: f32,
        crease_sensitivity: f32,
        use_vertex_meta: bool,
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
        let normal_mat = if model == Mat4::IDENTITY {
            Mat4::IDENTITY
        } else {
            model.inverse().transpose()
        };
        let u = ObjectUniforms {
            model: model.to_cols_array_2d(),
            normal_mat: normal_mat.to_cols_array_2d(),
            tint,
            meta: [
                (object_id.min(255) as f32) / 255.0,
                outline_weight,
                crease_sensitivity,
                if use_vertex_meta { 1.0 } else { 0.0 },
            ],
            skin_pose_r: [0.0; 4],
            skin_pose_l: [0.0; 4],
        };
        let off = idx * self.obj_align;
        let bytes = bytemuck::bytes_of(&u);
        self.obj_staging[off..off + bytes.len()].copy_from_slice(bytes);
        off as u32
    }

    #[allow(clippy::too_many_arguments)]
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
        animals: &[DynamicAnimal],
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
        let unjittered_proj_mat = Mat4::perspective_rh(fov_deg.to_radians(), aspect, near, far);
        let unjittered_view_proj = unjittered_proj_mat * view_mat;
        let unjittered_inv_view_proj = unjittered_view_proj.inverse();

        // Halton(2, 3) 8-phase sub-pixel camera projection jitter for Temporal Anti-Aliasing (TAA)
        let (proj_mat, curr_jitter_uv) = if graphics.taa_enabled {
            let j_px = halton_2_3_jitter_px(self.taa_frame_index);
            let w_f = target_width.max(1) as f32;
            let h_f = target_height.max(1) as f32;
            let j_ndc_x = (j_px[0] * 2.0) / w_f;
            let j_ndc_y = (j_px[1] * 2.0) / h_f;
            let jitter_mat = Mat4::from_translation(Vec3::new(j_ndc_x, j_ndc_y, 0.0));
            (
                jitter_mat * unjittered_proj_mat,
                [j_px[0] / w_f, -j_px[1] / h_f],
            )
        } else {
            (unjittered_proj_mat, [0.0, 0.0])
        };
        let view_proj = proj_mat * view_mat;
        let inv_view_proj = view_proj.inverse();

        let camera_cut = !self.taa_history_valid
            || self.last_camera_mode != Some(camera_mode)
            || self.last_biome != Some(biome)
            || cam_pos.distance(self.prev_cam_pos) > 15.0;
        let prev_vp_for_taa = if camera_cut {
            unjittered_view_proj
        } else {
            self.prev_unjittered_view_proj
        };
        let taa_u = TaaUniforms {
            inv_view_proj: unjittered_inv_view_proj.to_cols_array_2d(),
            prev_view_proj: prev_vp_for_taa.to_cols_array_2d(),
            screen_params: [
                target_width as f32,
                target_height as f32,
                1.0 / target_width.max(1) as f32,
                1.0 / target_height.max(1) as f32,
            ],
            jitter_params: [
                curr_jitter_uv[0],
                curr_jitter_uv[1],
                self.prev_jitter_uv[0],
                self.prev_jitter_uv[1],
            ],
            taa_params: [
                if graphics.taa_enabled { 1.0 } else { 0.0 },
                if camera_cut { 1.0 } else { 0.0 },
                if self.output_format.is_srgb() { 1.0 } else { 0.0 },
                0.003,
            ],
            camera_params: [near, far, 0.11, 0.22],
        };
        queue.write_buffer(&self.taa_buf, 0, bytemuck::bytes_of(&taa_u));

        // 3-Cascade Directional Shadow Map (CSM) with texel snapping & per-cascade caster size filtering
        let sun_dir = geo.hero_sun_dir;
        let shadow_sun_dir = Vec3::new(sun_dir.x, sun_dir.y.max(0.46), sun_dir.z).normalize();
        let view_fwd = (cam_target - cam_pos).normalize_or_zero();
        let csm_centers: [Vec3; NUM_SHADOW_CASCADES] = if is_menu {
            [
                player_pos + forward * 1.5,
                player_pos + forward * 24.0,
                player_pos + forward * 82.0,
            ]
        } else {
            [
                player_pos + forward * 3.0,
                cam_pos + view_fwd * 28.0,
                cam_pos + view_fwd * 86.0,
            ]
        };
        let csm_ortho_radii: [f32; NUM_SHADOW_CASCADES] = [12.0, 42.0, 128.0];
        let csm_split_depths: [f32; NUM_SHADOW_CASCADES] = [18.0, 68.0, 215.0];
        let csm_min_caster_r: [f32; NUM_SHADOW_CASCADES] = [0.0, 0.90, 2.50];
        let csm_max_caster_r: [f32; NUM_SHADOW_CASCADES] = [28.0, 45.0, 90.0];

        let light_rot_origin = Mat4::look_at_rh(shadow_sun_dir * 55.0, Vec3::ZERO, Vec3::Y);
        let cascade_vp: [Mat4; NUM_SHADOW_CASCADES] = std::array::from_fn(|ci| {
            let center = csm_centers[ci];
            let r_ortho = csm_ortho_radii[ci];
            let z_half = (r_ortho * 1.6).max(48.0);
            let texel_world = (2.0 * r_ortho) / (SHADOW_MAP_SIZE as f32);
            let c_ls = light_rot_origin.transform_point3(center);
            let dx = (c_ls.x / texel_world).round() * texel_world - c_ls.x;
            let dy = (c_ls.y / texel_world).round() * texel_world - c_ls.y;
            let light_view = Mat4::from_translation(Vec3::new(dx, dy, 0.0))
                * Mat4::look_at_rh(center + shadow_sun_dir * z_half, center, Vec3::Y);
            let light_proj =
                Mat4::orthographic_rh(-r_ortho, r_ortho, -r_ortho, r_ortho, 1.5, z_half * 2.0);
            light_proj * light_view
        });

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
            light_view_proj: cascade_vp[0].to_cols_array_2d(),
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
            light_view_proj_c1: cascade_vp[1].to_cols_array_2d(),
            light_view_proj_c2: cascade_vp[2].to_cols_array_2d(),
            cascade_splits: [
                csm_split_depths[0],
                csm_split_depths[1],
                csm_split_depths[2],
                if graphics.shadows_enabled { 1.0 } else { 0.0 },
            ],
        };
        queue.write_buffer(&self.frame_buf, 0, bytemuck::bytes_of(&frame_u));
        for ci in 0..NUM_SHADOW_CASCADES {
            let mut shadow_u = frame_u;
            shadow_u.light_view_proj = cascade_vp[ci].to_cols_array_2d();
            queue.write_buffer(&self.shadow_frame_bufs[ci], 0, bytemuck::bytes_of(&shadow_u));
        }

        // -----------------------------------------------------------------
        // CPU OCTREE FRUSTUM TRAVERSAL & SECTOR ENQUEUEING TO GPU
        // -----------------------------------------------------------------
        let max_prop_dist = graphics.prop_draw_distance_m.clamp(150.0, 1200.0);
        let cam_frustum = Frustum::from_view_proj(unjittered_view_proj);
        let csm_frustums: [Frustum; NUM_SHADOW_CASCADES] =
            std::array::from_fn(|ci| Frustum::from_view_proj(cascade_vp[ci]));

        let set_sector_bit = |mask: &mut [u32; 4], sid: usize| {
            let sid = sid.min(127);
            mask[sid >> 5] |= 1u32 << (sid & 31);
        };
        let is_sector_bit = |mask: &[u32; 4], sid: usize| -> bool {
            let sid = sid.min(127);
            ((mask[sid >> 5] >> (sid & 31)) & 1u32) != 0
        };

        // Bit 127 reserved for dynamic cyclists
        let mut cam_sector_mask = [0u32, 0u32, 0u32, 1u32 << 31];
        let mut shadow_sector_mask = [0u32, 0u32, 0u32, 1u32 << 31];
        let mut csm_sector_masks = [[0u32, 0u32, 0u32, 1u32 << 31]; NUM_SHADOW_CASCADES];

        // 1. Traverse 3D WorldOctree against Camera View Frustum on CPU
        let mut cam_vis_sectors: Vec<usize> = Vec::with_capacity(geo.sectors.len());
        geo.octree.query_frustum(&cam_frustum, &mut cam_vis_sectors);
        for sid in cam_vis_sectors {
            if sid < geo.sectors.len() {
                set_sector_bit(&mut cam_sector_mask, sid);
            }
        }

        // 2. Traverse 3D WorldOctree against CSM Shadow Frustums on CPU
        if graphics.shadows_enabled {
            let mut csm_vis_sectors: Vec<usize> = Vec::with_capacity(geo.sectors.len());
            for (ci, csm_f) in csm_frustums.iter().enumerate() {
                csm_vis_sectors.clear();
                geo.octree.query_frustum(csm_f, &mut csm_vis_sectors);
                for &sid in &csm_vis_sectors {
                    if sid < geo.sectors.len() {
                        set_sector_bit(&mut shadow_sector_mask, sid);
                        set_sector_bit(&mut csm_sector_masks[ci], sid);
                    }
                }
            }
        }

        if self.static_descs_dirty {
            if !self.static_draw_descs.is_empty() {
                queue.write_buffer(
                    &self.draw_desc_buf,
                    0,
                    bytemuck::cast_slice(&self.static_draw_descs),
                );
            }
            self.static_descs_dirty = false;
        }

        let static_cmd_base = self.static_draw_descs.len();
        let mut draw_count = 0usize;
        let mut rider_draw_descs: Vec<GpuDrawDesc> = Vec::with_capacity(96);
        let mut rider_cmds: Vec<EnqueuedCommandRef> = Vec::with_capacity(96);
        let mut skin_jobs: Vec<SkinJob> = Vec::with_capacity(64);
        let mut skin_bones: Vec<[[f32; 4]; 4]> = Vec::with_capacity(256);
        let mut max_skin_verts_in_job = 0u32;
        let mut skinned_vert_cursor = self.world_offsets.skinned_base_vertex;

        let transform_max_scale = |m: &Mat4| -> f32 {
            let sx = m.x_axis.truncate().length();
            let sy = m.y_axis.truncate().length();
            let sz = m.z_axis.truncate().length();
            sx.max(sy).max(sz).max(0.01)
        };

        // 1. Static world ObjectUniform slots (4 shared slots for the entire static world)
        let road_off =
            self.push_obj_uniform(device, draw_count, Mat4::IDENTITY, [1.0; 4], 1, 0.35, 0.0, false);
        draw_count += 1;
        let terrain_off =
            self.push_obj_uniform(device, draw_count, Mat4::IDENTITY, [1.0; 4], 2, 0.68, 0.0, false);
        draw_count += 1;
        let wire_off =
            self.push_obj_uniform(device, draw_count, Mat4::IDENTITY, [1.0; 4], 3, 0.55, 0.0, false);
        draw_count += 1;
        let prop_tier_off =
            self.push_obj_uniform(device, draw_count, Mat4::IDENTITY, [1.0; 4], 0, 0.0, 0.0, true);
        draw_count += 1;
        let static_uniform_offsets = [road_off, terrain_off, wire_off, prop_tier_off];
        let water_off = if self.world_offsets.water_index_count > 0 {
            let off = self.push_obj_uniform(
                device,
                draw_count,
                Mat4::IDENTITY,
                [1.0; 4],
                4,
                0.0,
                0.0,
                false,
            );
            draw_count += 1;
            Some(off)
        } else {
            None
        };

        // 2. Enqueue dynamic Riders & Compute-Skinned Quadruped Animals (pre-skinned via Compute Shader before rendering!)
        let mut rider_parts: Vec<RiderPartDraw> = Vec::with_capacity(192);
        let player_pose = if is_menu {
            RiderPoseMode::MenuStopped
        } else {
            RiderPoseMode::Pedaling
        };
        build_rider_draw_parts(
            spline,
            player_dist_m,
            eff_player_lat_m,
            1.0,
            player_wheel_rad,
            player_crank_rad,
            player_lean_rad,
            time_s,
            player_pose,
            player_profile,
            230,
            self.cyclist_skeleton.as_ref(),
            &mut rider_parts,
        );
        if !is_menu {
            for (ai_idx, ai) in ai_riders.iter().enumerate() {
                let s = spline.sample(ai.distance_m);
                if s.pos.distance(cam_pos) < 180.0 {
                    let base_id = 210_u32.saturating_sub((ai_idx as u32) * 5).max(140);
                    build_rider_draw_parts(
                        spline,
                        ai.distance_m,
                        ai.lateral_m,
                        ai.direction,
                        ai.wheel_angle_rad,
                        ai.crank_angle_rad,
                        ai.lean_angle_rad,
                        time_s,
                        RiderPoseMode::Pedaling,
                        &ai.profile,
                        base_id,
                        self.cyclist_skeleton.as_ref(),
                        &mut rider_parts,
                    );
                }
            }
        }
        build_animal_draw_parts(biome, spline, animals, cam_pos, &mut rider_parts);

        for rp in &rider_parts {
            let (sec_count, base_bound_r) = match self.models.get(&rp.model) {
                Some(m) => (m.sections.len(), m.bounding_radius),
                None => continue,
            };
            let pos = rp.transform.w_axis.truncate();
            let world_radius = base_bound_r.max(0.25) * transform_max_scale(&rp.transform);
            let (outline_w, crease_s) = match rp.model {
                ModelId::BikeWheel => (0.72, 0.0),
                ModelId::BikeFrame | ModelId::BikeCrank => (0.86, 0.0),
                _ => (0.92, 0.0),
            };
            let is_skinned = rp.skin_pose_r[3] > 0.5;
            let bone_offset = if let Some(mats) = &rp.bone_matrices {
                if skin_bones.len() + MAX_CYCLIST_BONES <= MAX_SKIN_BONES {
                    let off = skin_bones.len() as u32;
                    for m in mats {
                        skin_bones.push(m.to_cols_array_2d());
                    }
                    off
                } else {
                    0
                }
            } else {
                0
            };

            for sec_idx in 0..sec_count {
                if static_cmd_base + rider_draw_descs.len() >= MAX_INDIRECT_COMMANDS {
                    break;
                }
                let ms = self.models[&rp.model].sections[sec_idx];
                let tint = match ms.role {
                    MaterialRole::BikeFrame => rp.frame_color,
                    MaterialRole::JerseyPrimary => rp.jersey_primary,
                    MaterialRole::JerseySecondary => rp.jersey_secondary,
                    MaterialRole::JerseyAccent => rp.jersey_accent,
                    MaterialRole::Helmet => rp.helmet_color,
                    MaterialRole::HelmetAccent => rp.helmet_accent,
                    _ => [1.0, 1.0, 1.0, 1.0],
                };
                let off = self.push_obj_uniform(
                    device,
                    draw_count,
                    rp.transform,
                    tint,
                    rp.object_id,
                    outline_w,
                    crease_s,
                    false,
                );
                draw_count += 1;

                let eff_base_vertex = if is_skinned
                    && skin_jobs.len() < MAX_SKIN_JOBS
                    && (skinned_vert_cursor - self.world_offsets.skinned_base_vertex + ms.vertex_count)
                        <= MAX_SKINNED_VERTS as u32
                {
                    let dst_offset = skinned_vert_cursor;
                    skinned_vert_cursor += ms.vertex_count;
                    max_skin_verts_in_job = max_skin_verts_in_job.max(ms.vertex_count);
                    skin_jobs.push(SkinJob {
                        skin_pose_r: rp.skin_pose_r,
                        skin_pose_l: rp.skin_pose_l,
                        src_offset: ms.skin_src_offset,
                        dst_offset,
                        vertex_count: ms.vertex_count,
                        bone_offset,
                    });
                    dst_offset as i32
                } else {
                    ms.base_vertex
                };

                let slot = static_cmd_base + rider_draw_descs.len();
                rider_draw_descs.push(GpuDrawDesc {
                    pos_radius: [pos.x, pos.y, pos.z, world_radius],
                    index_count: ms.index_count,
                    first_index: ms.first_index,
                    base_vertex: eff_base_vertex,
                    flags: 127u32 | 256u32, // sector 127 (dynamic rider), cast_shadow = true
                });
                rider_cmds.push(EnqueuedCommandRef {
                    cmd_slot: slot,
                    offset_bytes: off,
                    shadow_candidate: true,
                });
            }
        }

        // Upload object uniforms, skinning jobs, bone matrices, and dynamic rider draw descriptors to GPU
        let write_bytes = draw_count * self.obj_align;
        if write_bytes > 0 {
            queue.write_buffer(&self.obj_buf, 0, &self.obj_staging[..write_bytes]);
        }

        let skin_params = SkinDispatchParams {
            job_count: skin_jobs.len() as u32,
            max_verts_per_job: max_skin_verts_in_job,
            _pad0: 0,
            _pad1: 0,
        };
        queue.write_buffer(&self.skin_params_buf, 0, bytemuck::bytes_of(&skin_params));
        if !skin_jobs.is_empty() {
            queue.write_buffer(&self.skin_jobs_buf, 0, bytemuck::cast_slice(&skin_jobs));
        }
        if !skin_bones.is_empty() {
            queue.write_buffer(&self.skin_bones_buf, 0, bytemuck::cast_slice(&skin_bones));
        }

        let cmd_count = (static_cmd_base + rider_draw_descs.len()) as u32;
        if !rider_draw_descs.is_empty() {
            let rider_byte_offset =
                (static_cmd_base * std::mem::size_of::<GpuDrawDesc>()) as u64;
            queue.write_buffer(
                &self.draw_desc_buf,
                rider_byte_offset,
                bytemuck::cast_slice(&rider_draw_descs),
            );
        }

        // Upload per-pass culling uniforms: 1 for Color View + 1 for each CSM Shadow Cascade
        let cam_cull_u = SinglePassCullUniforms {
            frustum_planes: cam_frustum.planes,
            filter_params: [cam_pos.x, cam_pos.y, cam_pos.z, max_prop_dist],
            sector_mask: cam_sector_mask,
            cmd_info: [cmd_count, 0, 0, 0],
        };
        queue.write_buffer(&self.cam_cull_uniform_buf, 0, bytemuck::bytes_of(&cam_cull_u));

        let shadows_flag = if graphics.shadows_enabled { 1.0_f32 } else { 0.0_f32 };
        for ci in 0..NUM_SHADOW_CASCADES {
            let csm_cull_u = SinglePassCullUniforms {
                frustum_planes: csm_frustums[ci].planes,
                filter_params: [
                    csm_min_caster_r[ci],
                    csm_max_caster_r[ci],
                    shadows_flag,
                    0.0,
                ],
                sector_mask: csm_sector_masks[ci],
                cmd_info: [cmd_count, 1, 0, 0],
            };
            queue.write_buffer(
                &self.csm_cull_uniform_bufs[ci],
                0,
                bytemuck::bytes_of(&csm_cull_u),
            );
        }

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("cadence_encoder"),
        });

        // -----------------------------------------------------------------
        // Pass 0a: Pre-Render Compute Skinning Pass (skinning.wgsl)
        // -----------------------------------------------------------------
        if !skin_jobs.is_empty() && max_skin_verts_in_job > 0 {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("skinning_compute_pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.skin_pipeline);
            cpass.set_bind_group(0, &self.skin_bg, &[]);
            cpass.dispatch_workgroups(
                max_skin_verts_in_job.div_ceil(64),
                skin_jobs.len() as u32,
                1,
            );
        }

        // -----------------------------------------------------------------
        // Pass 0b: Multi-Dispatch GPU Culling & Indirect Command Generation (cull_indirect.wgsl)
        //          - Dispatch 0: Color G-Buffer View -> cam_indirect_buf
        //          - Dispatches 1..=3: CSM Cascades C0, C1, C2 -> csm_indirect_bufs[0..3]
        // -----------------------------------------------------------------
        if cmd_count > 0 {
            let workgroups = cmd_count.div_ceil(64);
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("cull_indirect_multi_dispatch_pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.cull_pipeline);
            // Color G-Buffer View Culling Dispatch
            cpass.set_bind_group(0, &self.cam_cull_bg, &[]);
            cpass.dispatch_workgroups(workgroups, 1, 1);
            // Separate Culling Dispatch per CSM Cascade
            for ci in 0..NUM_SHADOW_CASCADES {
                cpass.set_bind_group(0, &self.csm_cull_bind_groups[ci], &[]);
                cpass.dispatch_workgroups(workgroups, 1, 1);
            }
        }

        let indirect_cmd_size = std::mem::size_of::<DrawIndexedIndirectCmd>() as u64;

        // -----------------------------------------------------------------
        // Passes 1..3: 3-Cascade Directional Shadow Maps via Dedicated Per-Cascade Indirect Buffers
        // -----------------------------------------------------------------
        for ci in 0..NUM_SHADOW_CASCADES {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some(&format!("shadow_pass_c{ci}")),
                color_attachments: &[],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.shadow_cascade_views[ci],
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
            rpass.set_bind_group(0, &self.shadow_frame_bgs[ci], &[]);
            rpass.set_vertex_buffer(0, self.mega_vb.slice(..));
            rpass.set_index_buffer(self.mega_ib.slice(..), wgpu::IndexFormat::Uint32);
            let csm_ind_buf = &self.csm_indirect_bufs[ci];
            if graphics.shadows_enabled {
                let mut last_off: Option<u32> = None;
                let csm_mask = &csm_sector_masks[ci];
                let min_tier = (ci as u8) + 1;
                for cmd in &self.static_sector_cmds {
                    if cmd.shadow_candidate
                        && cmd.caster_tier >= min_tier
                        && is_sector_bit(csm_mask, cmd.sector_id)
                    {
                        let off = static_uniform_offsets[cmd.uniform_idx];
                        if last_off != Some(off) {
                            rpass.set_bind_group(1, &self.obj_bg, &[off]);
                            last_off = Some(off);
                        }
                        let indirect_offset = (cmd.cmd_slot as u64) * indirect_cmd_size;
                        rpass.draw_indexed_indirect(csm_ind_buf, indirect_offset);
                    }
                }
                for cmd in &rider_cmds {
                    if cmd.shadow_candidate {
                        if last_off != Some(cmd.offset_bytes) {
                            rpass.set_bind_group(1, &self.obj_bg, &[cmd.offset_bytes]);
                            last_off = Some(cmd.offset_bytes);
                        }
                        let indirect_offset = (cmd.cmd_slot as u64) * indirect_cmd_size;
                        rpass.draw_indexed_indirect(csm_ind_buf, indirect_offset);
                    }
                }
            }
        }

        // -----------------------------------------------------------------
        // Pass 4: Pure Unlit G-Buffer Pass (Domain-Permuted Shaders + Opaque Water)
        // -----------------------------------------------------------------
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

            // 4a. Sky fullscreen
            rpass.set_pipeline(&self.sky_pipeline);
            rpass.set_bind_group(0, &self.frame_bg, &[]);
            rpass.draw(0..3, 0..1);

            // 4b. Static sector geometry via Domain-Specific G-Buffer Pipelines & cam_indirect_buf
            let road_pipe = if biome == BiomeId::RollingRuralGravel {
                &self.gbuf_road_gravel_pipeline
            } else {
                &self.gbuf_road_tarmac_pipeline
            };
            let terrain_pipe = if biome == BiomeId::CoastalPromenade {
                &self.gbuf_terrain_coastal_pipeline
            } else {
                &self.gbuf_terrain_natural_pipeline
            };

            rpass.set_bind_group(0, &self.frame_bg, &[]);
            rpass.set_vertex_buffer(0, self.mega_vb.slice(..));
            rpass.set_index_buffer(self.mega_ib.slice(..), wgpu::IndexFormat::Uint32);
            let mut last_off: Option<u32> = None;
            let mut last_domain: Option<usize> = None;
            for cmd in &self.static_sector_cmds {
                if is_sector_bit(&cam_sector_mask, cmd.sector_id) {
                    let domain = match cmd.uniform_idx {
                        0 => 0, // Road domain
                        1 => 1, // Terrain domain
                        _ => 2, // Prop/Wire/Foliage domain
                    };
                    if last_domain != Some(domain) {
                        match domain {
                            0 => rpass.set_pipeline(road_pipe),
                            1 => rpass.set_pipeline(terrain_pipe),
                            _ => rpass.set_pipeline(&self.gbuf_prop_pipeline),
                        }
                        last_domain = Some(domain);
                    }
                    let off = static_uniform_offsets[cmd.uniform_idx];
                    if last_off != Some(off) {
                        rpass.set_bind_group(1, &self.obj_bg, &[off]);
                        last_off = Some(off);
                    }
                    let indirect_offset = (cmd.cmd_slot as u64) * indirect_cmd_size;
                    rpass.draw_indexed_indirect(&self.cam_indirect_buf, indirect_offset);
                }
            }

            // 4c. Dynamic Riders & Compute-Skinned Quadruped Animals via Dedicated Skinned G-Buffer Pipeline
            if !rider_cmds.is_empty() {
                rpass.set_pipeline(&self.gbuf_skinned_pipeline);
                for cmd in &rider_cmds {
                    if last_off != Some(cmd.offset_bytes) {
                        rpass.set_bind_group(1, &self.obj_bg, &[cmd.offset_bytes]);
                        last_off = Some(cmd.offset_bytes);
                    }
                    let indirect_offset = (cmd.cmd_slot as u64) * indirect_cmd_size;
                    rpass.draw_indexed_indirect(&self.cam_indirect_buf, indirect_offset);
                }
            }

            // 4d. Opaque Water rendered directly into the G-Buffer (lit in the deferred lighting pass)
            if let Some(w_off) = water_off {
                rpass.set_pipeline(&self.gbuf_water_pipeline);
                rpass.set_bind_group(0, &self.frame_bg, &[]);
                rpass.set_bind_group(1, &self.obj_bg, &[w_off]);
                rpass.draw_indexed(
                    self.world_offsets.water_first_index
                        ..(self.world_offsets.water_first_index
                            + self.world_offsets.water_index_count),
                    self.world_offsets.water_base_vertex,
                    0..1,
                );
            }
        }

        // -----------------------------------------------------------------
        // Pass 5: Fullscreen Ligne Claire Ink Outline Post-Process -> Offscreen Post Color Buffer
        // -----------------------------------------------------------------
        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("post_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.gbuf.post_color_view,
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

        // -----------------------------------------------------------------
        // Pass 5b: Temporal Anti-Aliasing (TAA) Resolve & CAS Sharpening -> Target Surface + TAA History
        // -----------------------------------------------------------------
        let curr_taa_idx = (self.taa_frame_index as usize) & 1;
        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("taa_pass"),
                color_attachments: &[
                    Some(wgpu::RenderPassColorAttachment {
                        view: target_view,
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: wgpu::StoreOp::Store,
                        },
                    }),
                    Some(wgpu::RenderPassColorAttachment {
                        view: &self.gbuf.taa_history_views[curr_taa_idx],
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: wgpu::StoreOp::Store,
                        },
                    }),
                ],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            rpass.set_pipeline(&self.taa_pipeline);
            rpass.set_bind_group(0, &self.gbuf.taa_bind_groups[curr_taa_idx], &[]);
            rpass.draw(0..3, 0..1);
        }

        self.prev_unjittered_view_proj = unjittered_view_proj;
        self.prev_jitter_uv = curr_jitter_uv;
        self.prev_cam_pos = cam_pos;
        self.last_camera_mode = Some(camera_mode);
        self.last_biome = Some(biome);
        self.taa_history_valid = true;
        self.taa_frame_index = self.taa_frame_index.wrapping_add(1);

        // -----------------------------------------------------------------
        // Pass 6: Stylized egui HUD Overlay
        // -----------------------------------------------------------------
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
            &self.gbuf.scene_depth_tex,
            &self.gbuf.post_color_tex,
            &self.gbuf.taa_history_tex,
            &self.frame_bgl,
            &self.cull_bgl,
            &self.csm_cull_bgs,
        );
    }
}

