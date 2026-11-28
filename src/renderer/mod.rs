use std::collections::HashMap;
use std::num::NonZeroU64;

use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3};
use wgpu::util::DeviceExt;

use crate::assets::{f32_to_f16, AssetBank, MaterialRole, ModelId, SceneVertex, SkeletonInfo, SkinWeightVertex};
use crate::world::riders::{
    build_animal_draw_parts, build_rider_draw_parts, AiRider, DynamicAnimal, RiderPartDraw,
    RiderPoseMode, RiderProfile, BIKE_COMPUTER_SCREEN_CORNERS, MAX_CYCLIST_BONES,
};
use crate::world::terrain::{Frustum, StaticWorldGeometry, WorldOctree};
use crate::world::{BiomeId, RouteSpline};

pub const SHADOW_MAP_SIZE: u32 = 2048;
pub const NUM_SHADOW_CASCADES: usize = 3;
pub const CSM_ORTHO_RADII: [f32; NUM_SHADOW_CASCADES] = [12.0, 42.0, 128.0];
pub const CSM_SPLIT_DEPTHS: [f32; NUM_SHADOW_CASCADES] = [18.0, 68.0, 215.0];
pub const CSM_MIN_CASTER_R: [f32; NUM_SHADOW_CASCADES] = [0.0, 0.90, 2.50];
pub const CSM_MAX_CASTER_R: [f32; NUM_SHADOW_CASCADES] = [28.0, 45.0, 90.0];

pub const BUCKET_ROAD: usize = 0;
pub const BUCKET_TERRAIN: usize = 1;
pub const BUCKET_PROP: usize = 2;
pub const BUCKET_SKINNED: usize = 3;
pub const BUCKET_WATER: usize = 4;
pub const NUM_MATERIAL_BUCKETS: usize = 5;
pub const NUM_DRAW_COUNT_SLOTS: usize = NUM_MATERIAL_BUCKETS + NUM_SHADOW_CASCADES; // 5 G-Buffer buckets + 3 CSM cascades
pub const MAX_SKINNED_VERTS: usize = 524288;
pub const MAX_SKIN_JOBS: usize = 512;
pub const MAX_SKIN_BONES: usize = 2048;
pub const MAX_INDIRECT_COMMANDS: usize = 16384;
pub const MAX_TOTAL_INDIRECT_SLOTS: usize = MAX_INDIRECT_COMMANDS * (1 + NUM_SHADOW_CASCADES);
pub const MIN_PROJECTED_BBOX_PX: f32 = 8.0;

pub fn preferred_device_features(adapter: &wgpu::Adapter) -> wgpu::Features {
    let avail = adapter.features();
    let mut req = wgpu::Features::empty();
    if avail.contains(wgpu::Features::INDIRECT_FIRST_INSTANCE) {
        req |= wgpu::Features::INDIRECT_FIRST_INSTANCE;
    }
    if avail.contains(wgpu::Features::MULTI_DRAW_INDIRECT_COUNT) {
        req |= wgpu::Features::MULTI_DRAW_INDIRECT_COUNT;
    }
    req
}

/// Computes a texel-snapped orthographic light-space view-projection matrix for one CSM cascade,
/// extending the depth range upstream/downstream by `1 / sin(elevation)` so tall buildings and
/// steep canyon slopes are never clipped by the shadow near/far planes.
pub fn compute_csm_cascade_vp(center: Vec3, shadow_sun_dir: Vec3, r_ortho: f32) -> Mat4 {
    let sin_e = shadow_sun_dir.y.max(0.25);
    let cos_e = (1.0 - sin_e * sin_e).max(0.0).sqrt();
    let cot_e = cos_e / sin_e;
    let z_back = r_ortho * cot_e + 85.0 / sin_e;
    let z_front = r_ortho * cot_e + 40.0 / sin_e;
    let texel_world = (2.0 * r_ortho) / (SHADOW_MAP_SIZE as f32);
    let light_rot_origin = Mat4::look_at_rh(shadow_sun_dir * 55.0, Vec3::ZERO, Vec3::Y);
    let c_ls = light_rot_origin.transform_point3(center);
    let dx = c_ls.x - (c_ls.x / texel_world).round() * texel_world;
    let dy = c_ls.y - (c_ls.y / texel_world).round() * texel_world;
    let light_view = Mat4::from_translation(Vec3::new(dx, dy, 0.0))
        * Mat4::look_at_rh(center + shadow_sun_dir * z_back, center, Vec3::Y);
    let light_proj =
        Mat4::orthographic_rh(-r_ortho, r_ortho, -r_ortho, r_ortho, 1.5, z_back + z_front);
    light_proj * light_view
}

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
    ChestPov,
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
            CameraMode::ChestPov => "Chest POV",
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
            CameraMode::FirstPerson => CameraMode::ChestPov,
            CameraMode::ChestPov | CameraMode::TurnaroundBack => CameraMode::Chase,
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
pub struct ObjectData {
    pub model: [[f32; 4]; 4],
    pub normal_mat: [[f32; 4]; 4],
    pub tint: [f32; 4],
    pub meta: [f32; 4],
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
pub struct MultiPassCullUniforms {
    pub cam_view_proj: [[f32; 4]; 4],
    pub cam_planes: [[f32; 4]; 6],
    pub csm_planes_0: [[f32; 4]; 6],
    pub csm_planes_1: [[f32; 4]; 6],
    pub csm_planes_2: [[f32; 4]; 6],
    pub cam_pos_dist: [f32; 4],
    pub screen_params: [f32; 4],
    pub csm_min_r: [f32; 4],
    pub csm_max_r: [f32; 4],
    pub cam_sector_mask: [u32; 4],
    pub csm_sector_mask_0: [u32; 4],
    pub csm_sector_mask_1: [u32; 4],
    pub csm_sector_mask_2: [u32; 4],
    pub bucket_bases: [[u32; 4]; 2],
    pub bucket_caps: [[u32; 4]; 2],
    pub cmd_info: [u32; 4], // [cmd_count, compact_enabled (0/1), 0, 0]
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct GpuDrawDesc {
    pub pos_radius: [f32; 4],
    pub aabb_half: [f32; 4],
    pub index_count: u32,
    pub first_index: u32,
    pub base_vertex: i32,
    pub object_idx: u32,
    pub flags: u32,           // bits 0..6: sector_id (0..126 or 127), bit 8: cast_shadow, bit 9: is_sector_surface, bits 16..31: rep_caster_cm
    pub material_bucket: u32, // 0=Road, 1=Terrain, 2=Prop, 3=Skinned, 4=Water
    pub _pad0: u32,
    pub _pad1: u32,
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

const _: () = assert!(std::mem::size_of::<ObjectData>() == 160);
const _: () = assert!(std::mem::size_of::<GpuDrawDesc>() == 64);
const _: () = assert!(std::mem::size_of::<DrawIndexedIndirectCmd>() == 20);
const _: () = assert!(std::mem::size_of::<MultiPassCullUniforms>() == 656);
const _: () = assert!(std::mem::size_of::<MultiPassCullUniforms>() % 16 == 0);

pub fn transform_aabb(m: Mat4, local_center: Vec3, local_half: Vec3) -> (Vec3, Vec3) {
    let world_center = m.transform_point3(local_center);
    let rx = m.x_axis.truncate().abs();
    let ry = m.y_axis.truncate().abs();
    let rz = m.z_axis.truncate().abs();
    let world_half = rx * local_half.x + ry * local_half.y + rz * local_half.z;
    (world_center, world_half.max(Vec3::splat(0.01)))
}

pub fn projected_aabb_max_extent_px(
    view_proj: Mat4,
    center: Vec3,
    half: Vec3,
    screen_w: f32,
    screen_h: f32,
) -> f32 {
    let mut min_x = f32::INFINITY;
    let mut min_y = f32::INFINITY;
    let mut max_x = f32::NEG_INFINITY;
    let mut max_y = f32::NEG_INFINITY;
    for i in 0..8 {
        let sx = if (i & 1) != 0 { 1.0 } else { -1.0 };
        let sy = if (i & 2) != 0 { 1.0 } else { -1.0 };
        let sz = if (i & 4) != 0 { 1.0 } else { -1.0 };
        let corner = center + Vec3::new(sx * half.x, sy * half.y, sz * half.z);
        let clip = view_proj * corner.extend(1.0);
        if clip.w <= 0.001 {
            return f32::INFINITY;
        }
        let ndc_x = clip.x / clip.w;
        let ndc_y = clip.y / clip.w;
        min_x = min_x.min(ndc_x);
        min_y = min_y.min(ndc_y);
        max_x = max_x.max(ndc_x);
        max_y = max_y.max(ndc_y);
    }
    let ext_w = (max_x - min_x) * 0.5 * screen_w;
    let ext_h = (max_y - min_y) * 0.5 * screen_h;
    ext_w.max(ext_h)
}

pub fn compute_indirect_layout(
    cam_bucket_caps: [u32; NUM_MATERIAL_BUCKETS],
    csm_caps: [u32; NUM_SHADOW_CASCADES],
) -> ([u32; NUM_DRAW_COUNT_SLOTS], [u32; NUM_DRAW_COUNT_SLOTS], u32) {
    let mut bases = [0u32; NUM_DRAW_COUNT_SLOTS];
    let mut caps = [0u32; NUM_DRAW_COUNT_SLOTS];
    let mut cursor = 0u32;
    for i in 0..NUM_MATERIAL_BUCKETS {
        bases[i] = cursor;
        caps[i] = cam_bucket_caps[i];
        cursor = cursor.saturating_add(cam_bucket_caps[i]);
    }
    for ci in 0..NUM_SHADOW_CASCADES {
        let idx = NUM_MATERIAL_BUCKETS + ci;
        bases[idx] = cursor;
        caps[idx] = csm_caps[ci];
        cursor = cursor.saturating_add(csm_caps[ci]);
    }
    (bases, caps, cursor)
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
    bounds_center: Vec3,
    bounds_half_extents: Vec3,
}

struct GpuModel {
    sections: Vec<GpuMeshSection>,
    bounding_radius: f32,
}

struct GpuWorldOffsets {
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

/// Chest POV eye position relative to the rider's road contact point (metres, rider basis).
const CHEST_POV_EYE_UP: f32 = 1.36;
const CHEST_POV_EYE_FWD: f32 = 0.115;
/// Chest POV near plane: small enough that the arms passing beside the lens are not clipped open.
const CHEST_POV_NEAR_M: f32 = 0.06;

/// Projects the head unit's display corners to normalized screen UV; `None` if any corner is behind the camera.
fn project_head_unit_screen(model: Mat4, view_proj: Mat4) -> Option<[[f32; 2]; 4]> {
    let mut out = [[0.0; 2]; 4];
    for (o, c) in out.iter_mut().zip(BIKE_COMPUTER_SCREEN_CORNERS) {
        let clip = view_proj * model.transform_point3(c).extend(1.0);
        if clip.w <= 1e-4 {
            return None;
        }
        let ndc = clip.truncate() / clip.w;
        *o = [ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5];
    }
    Some(out)
}

pub struct LigneClaireRenderer {
    pub output_format: wgpu::TextureFormat,
    supports_mdi_count: bool,
    supports_indirect_first_instance: bool,
    models: HashMap<ModelId, GpuModel>,
    static_model_verts: Vec<SceneVertex>,
    static_model_indices: Vec<u32>,
    static_draw_descs: Vec<GpuDrawDesc>,
    static_bucket_counts: [u32; NUM_MATERIAL_BUCKETS],
    static_shadow_counts: [u32; NUM_SHADOW_CASCADES],
    static_descs_dirty: bool,
    world_octree: WorldOctree,

    mega_vb: wgpu::Buffer,
    instance_id_vb: wgpu::Buffer,
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

    // Single-Dispatch GPU Culling & Stream-Compacted Shared Indirect Draw Buffer resources
    cull_uniform_buf: wgpu::Buffer,
    draw_desc_buf: wgpu::Buffer,
    indirect_draw_buf: wgpu::Buffer,
    draw_counts_buf: wgpu::Buffer,
    cull_bgl: wgpu::BindGroupLayout,
    cull_bg: wgpu::BindGroup,
    cull_pipeline: wgpu::ComputePipeline,

    frame_buf: wgpu::Buffer,
    shadow_frame_bufs: [wgpu::Buffer; NUM_SHADOW_CASCADES],
    obj_buf: wgpu::Buffer,
    obj_capacity: usize,
    objects_staging: Vec<ObjectData>,

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
    /// Player head-unit display corners in normalized screen UV (0..1, top-left origin) from the
    /// last rendered frame, set only in `CameraMode::ChestPov`. Consumed by the egui HUD next frame.
    pub head_unit_screen_uv: Option<[[f32; 2]; 4]>,
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
        let features = device.features();
        let supports_indirect_first_instance =
            features.contains(wgpu::Features::INDIRECT_FIRST_INSTANCE);
        let supports_mdi_count = supports_indirect_first_instance
            && features.contains(wgpu::Features::MULTI_DRAW_INDIRECT_COUNT);

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
                    bounds_center: sec.bounds_center,
                    bounds_half_extents: sec.bounds_half_extents,
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

        // Single-Dispatch GPU Culling & Stream-Compacted Shared Indirect Draw Buffer pipeline
        let multi_cull_u_size = std::mem::size_of::<MultiPassCullUniforms>() as u64;
        let shared_indirect_buf_size =
            (std::mem::size_of::<DrawIndexedIndirectCmd>() * MAX_TOTAL_INDIRECT_SLOTS) as u64;
        let draw_counts_buf_size = (std::mem::size_of::<u32>() * NUM_DRAW_COUNT_SLOTS) as u64;

        let cull_uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("multi_cull_uniform_buf"),
            size: multi_cull_u_size,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let draw_desc_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("draw_desc_buf"),
            size: (std::mem::size_of::<GpuDrawDesc>() * MAX_INDIRECT_COMMANDS) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let indirect_draw_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("shared_indirect_draw_buf"),
            size: shared_indirect_buf_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::INDIRECT
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let draw_counts_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("shared_draw_counts_buf"),
            size: draw_counts_buf_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::INDIRECT
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
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
                        min_binding_size: NonZeroU64::new(multi_cull_u_size),
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
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: NonZeroU64::new(draw_counts_buf_size),
                    },
                    count: None,
                },
            ],
        });
        let cull_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("cull_bg"),
            layout: &cull_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: cull_uniform_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: draw_desc_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: indirect_draw_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: draw_counts_buf.as_entire_binding(),
                },
            ],
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

        let obj_capacity = MAX_INDIRECT_COMMANDS;
        let obj_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("objects_storage_buf"),
            size: (std::mem::size_of::<ObjectData>() * obj_capacity) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let instance_id_vb = Self::make_instance_id_vb(device, obj_capacity);

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
            label: Some("objects_bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: NonZeroU64::new(std::mem::size_of::<ObjectData>() as u64),
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
                buffers: &[Some(SceneVertex::desc()), Some(Self::instance_id_vb_desc())],
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
                        buffers: &[Some(SceneVertex::desc()), Some(Self::instance_id_vb_desc())],
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
            supports_mdi_count,
            supports_indirect_first_instance,
            models,
            static_model_verts,
            static_model_indices,
            static_draw_descs: Vec::new(),
            static_bucket_counts: [0; NUM_MATERIAL_BUCKETS],
            static_shadow_counts: [0; NUM_SHADOW_CASCADES],
            static_descs_dirty: true,
            world_octree: WorldOctree {
                nodes: Vec::new(),
                root: 0,
            },
            mega_vb,
            instance_id_vb,
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
            cull_uniform_buf,
            draw_desc_buf,
            indirect_draw_buf,
            draw_counts_buf,
            cull_bgl,
            cull_bg,
            cull_pipeline,
            frame_buf,
            shadow_frame_bufs,
            obj_buf,
            obj_capacity,
            objects_staging: vec![ObjectData::zeroed(); obj_capacity],
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
            head_unit_screen_uv: None,
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
            label: Some("objects_bg"),
            layout: bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: buf.as_entire_binding(),
            }],
        })
    }

    fn make_instance_id_vb(device: &wgpu::Device, cap: usize) -> wgpu::Buffer {
        let ids: Vec<u32> = (0..cap as u32).collect();
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("instance_id_vb"),
            contents: bytemuck::cast_slice(&ids),
            usage: wgpu::BufferUsages::VERTEX,
        })
    }

    fn instance_id_vb_desc() -> wgpu::VertexBufferLayout<'static> {
        const ATTRS: &[wgpu::VertexAttribute] = &[wgpu::VertexAttribute {
            offset: 0,
            shader_location: 4,
            format: wgpu::VertexFormat::Uint32,
        }];
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<u32>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: ATTRS,
        }
    }

    fn submit_indirect_bucket<'a>(
        &'a self,
        rpass: &mut wgpu::RenderPass<'a>,
        base_slot: u32,
        max_count: u32,
        count_index: u32,
    ) {
        if max_count == 0 {
            return;
        }
        let indirect_cmd_size = std::mem::size_of::<DrawIndexedIndirectCmd>() as u64;
        let offset = (base_slot as u64) * indirect_cmd_size;
        if self.supports_mdi_count {
            let count_offset = (count_index as u64) * std::mem::size_of::<u32>() as u64;
            rpass.multi_draw_indexed_indirect_count(
                &self.indirect_draw_buf,
                offset,
                &self.draw_counts_buf,
                count_offset,
                max_count,
            );
        } else {
            rpass.multi_draw_indexed_indirect(&self.indirect_draw_buf, offset, max_count);
        }
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
        let mut static_draw_descs: Vec<GpuDrawDesc> = Vec::with_capacity(geo.sectors.len() * 7 + 1);
        let mut exact_sector_bounds: Vec<(Vec3, Vec3)> =
            geo.sectors.iter().map(|s| (s.aabb_min, s.aabb_max)).collect();

        for (sid, sec) in geo.sectors.iter().enumerate() {
            let sec_flag = (sid as u32) | 512u32; // bit 9 = is_sector_surface
            let sec_half = ((sec.aabb_max - sec.aabb_min) * 0.5).max(Vec3::splat(0.1));
            if sec.road_index_count > 0 && static_draw_descs.len() < MAX_INDIRECT_COMMANDS {
                static_draw_descs.push(GpuDrawDesc {
                    pos_radius: [sec.center.x, sec.center.y, sec.center.z, sec.bounding_radius],
                    aabb_half: [sec_half.x, sec_half.y, sec_half.z, 0.0],
                    index_count: sec.road_index_count,
                    first_index: road_first_index + sec.road_index_offset,
                    base_vertex: road_base_vertex,
                    object_idx: 0,
                    flags: sec_flag,
                    material_bucket: BUCKET_ROAD as u32,
                    _pad0: 0,
                    _pad1: 0,
                });
            }
            if sec.terrain_index_count > 0 && static_draw_descs.len() < MAX_INDIRECT_COMMANDS {
                static_draw_descs.push(GpuDrawDesc {
                    pos_radius: [sec.center.x, sec.center.y, sec.center.z, sec.bounding_radius],
                    aabb_half: [sec_half.x, sec_half.y, sec_half.z, 0.0],
                    index_count: sec.terrain_index_count,
                    first_index: terrain_first_index + sec.terrain_index_offset,
                    base_vertex: terrain_base_vertex,
                    object_idx: 1,
                    flags: sec_flag,
                    material_bucket: BUCKET_TERRAIN as u32,
                    _pad0: 0,
                    _pad1: 0,
                });
            }
            if sec.wire_index_count > 0 && static_draw_descs.len() < MAX_INDIRECT_COMMANDS {
                static_draw_descs.push(GpuDrawDesc {
                    pos_radius: [sec.center.x, sec.center.y, sec.center.z, sec.bounding_radius],
                    aabb_half: [sec_half.x, sec_half.y, sec_half.z, 0.0],
                    index_count: sec.wire_index_count,
                    first_index: wire_first_index + sec.wire_index_offset,
                    base_vertex: wire_base_vertex,
                    object_idx: 2,
                    flags: sec_flag,
                    material_bucket: BUCKET_PROP as u32,
                    _pad0: 0,
                    _pad1: 0,
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
                    let t_half =
                        ((tier_max[tier_idx] - tier_min[tier_idx]) * 0.5).max(Vec3::splat(0.1))
                            + Vec3::splat(0.25);
                    let t_radius = (tier_max[tier_idx] - t_center).length().max(0.5) + 1.0;
                    exact_sector_bounds[sid].0 =
                        exact_sector_bounds[sid].0.min(tier_min[tier_idx] - Vec3::splat(1.0));
                    exact_sector_bounds[sid].1 =
                        exact_sector_bounds[sid].1.max(tier_max[tier_idx] + Vec3::splat(1.0));
                    let (cast_shadow, rep_caster_cm): (bool, u32) = match tier_idx {
                        0 => (false, 0),
                        1 => (true, 50),  // 0.50 m -> Cascade 0 only
                        2 => (true, 160), // 1.60 m -> Cascades 0 & 1
                        _ => (true, 600), // 6.00 m -> Cascades 0, 1 & 2
                    };
                    let flags = (sid as u32)
                        | if cast_shadow { 256u32 } else { 0u32 }
                        | (rep_caster_cm << 16);
                    static_draw_descs.push(GpuDrawDesc {
                        pos_radius: [t_center.x, t_center.y, t_center.z, t_radius],
                        aabb_half: [t_half.x, t_half.y, t_half.z, 0.0],
                        index_count: tier_index_count,
                        first_index: tier_first_index,
                        base_vertex: tier_base_vertex,
                        object_idx: 3,
                        flags,
                        material_bucket: BUCKET_PROP as u32,
                        _pad0: 0,
                        _pad1: 0,
                    });
                }
            }
        }

        if water_index_count > 0
            && !geo.water_vertices.is_empty()
            && static_draw_descs.len() < MAX_INDIRECT_COMMANDS
        {
            let mut w_min = Vec3::splat(f32::INFINITY);
            let mut w_max = Vec3::splat(f32::NEG_INFINITY);
            for v in &geo.water_vertices {
                let p = Vec3::from_array(v.pos);
                w_min = w_min.min(p);
                w_max = w_max.max(p);
            }
            let w_center = (w_min + w_max) * 0.5;
            let w_half = ((w_max - w_min) * 0.5).max(Vec3::splat(0.1));
            let w_radius = (w_max - w_center).length().max(1.0);
            static_draw_descs.push(GpuDrawDesc {
                pos_radius: [w_center.x, w_center.y, w_center.z, w_radius],
                aabb_half: [w_half.x, w_half.y, w_half.z, 0.0],
                index_count: water_index_count,
                first_index: water_first_index,
                base_vertex: water_base_vertex,
                object_idx: 4,
                flags: 127u32 | 512u32, // always-visible sector 127 + is_sector_surface, cast_shadow = false
                material_bucket: BUCKET_WATER as u32,
                _pad0: 0,
                _pad1: 0,
            });
        }

        // Group static draw descriptors by (material_bucket, sector_id, object_idx)
        static_draw_descs.sort_by_key(|d| (d.material_bucket, d.flags & 127, d.object_idx));

        let mut static_bucket_counts = [0u32; NUM_MATERIAL_BUCKETS];
        let mut static_shadow_counts = [0u32; NUM_SHADOW_CASCADES];
        for d in &static_draw_descs {
            let b = (d.material_bucket as usize).min(NUM_MATERIAL_BUCKETS - 1);
            static_bucket_counts[b] += 1;
            if (d.flags & 256) != 0 {
                let raw_caster_cm = d.flags >> 16;
                let caster_r = if raw_caster_cm > 0 {
                    (raw_caster_cm as f32) * 0.01
                } else {
                    d.pos_radius[3]
                };
                for ci in 0..NUM_SHADOW_CASCADES {
                    if caster_r >= CSM_MIN_CASTER_R[ci] && caster_r <= CSM_MAX_CASTER_R[ci] {
                        static_shadow_counts[ci] += 1;
                    }
                }
            }
        }

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
            skinned_base_vertex,
        };
        self.static_draw_descs = static_draw_descs;
        self.static_bucket_counts = static_bucket_counts;
        self.static_shadow_counts = static_shadow_counts;
        self.static_descs_dirty = true;
        self.world_octree = geo.octree.clone();
        crate::world::terrain::update_octree_bounds_from_sector_aabbs(
            &mut self.world_octree,
            &exact_sector_bounds,
        );
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
    fn push_object_data(
        &mut self,
        device: &wgpu::Device,
        idx: usize,
        model: Mat4,
        tint: [f32; 4],
        object_id: u32,
        outline_weight: f32,
        crease_sensitivity: f32,
        use_vertex_meta: bool,
    ) {
        if idx >= self.obj_capacity {
            let new_cap = (idx + 512).next_power_of_two();
            self.obj_buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("objects_storage_buf"),
                size: (std::mem::size_of::<ObjectData>() * new_cap) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.obj_bg = Self::make_obj_bg(device, &self.obj_bgl, &self.obj_buf);
            self.instance_id_vb = Self::make_instance_id_vb(device, new_cap);
            self.objects_staging.resize(new_cap, ObjectData::zeroed());
            self.obj_capacity = new_cap;
        }
        let normal_mat = if model == Mat4::IDENTITY {
            Mat4::IDENTITY
        } else {
            model.inverse().transpose()
        };
        self.objects_staging[idx] = ObjectData {
            model: model.to_cols_array_2d(),
            normal_mat: normal_mat.to_cols_array_2d(),
            tint,
            meta: [
                (object_id.min(255) as f32) / 255.0,
                outline_weight,
                crease_sensitivity,
                if use_vertex_meta { 1.0 } else { 0.0 },
            ],
        };
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
            // Chest-mounted POV: sits just in front of the sternum (like a chest-strap action cam),
            // so the shoulders are behind the lens and the arms enter whole from the screen edges.
            // Pitched ~26 deg down so the bars, stem, head unit and hands frame the bottom.
            CameraMode::ChestPov => {
                let eye = player_pos + up * CHEST_POV_EYE_UP + forward * CHEST_POV_EYE_FWD;
                (eye, eye + forward * 10.0 - up * 4.88, 76.0)
            }
        };
        let fov_deg = (base_fov_deg + graphics.fov_offset_deg).clamp(32.0, 80.0);

        let aspect = target_width.max(1) as f32 / target_height.max(1) as f32;
        // The chest cam has forearms ~0.1 m from the lens; everything else keeps the wider near plane
        // for depth precision at the 1.9 km far plane.
        let near = if camera_mode == CameraMode::ChestPov {
            CHEST_POV_NEAR_M
        } else {
            0.25_f32
        };
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
        let csm_ortho_radii = CSM_ORTHO_RADII;
        let csm_split_depths = CSM_SPLIT_DEPTHS;
        let csm_min_caster_r = CSM_MIN_CASTER_R;
        let csm_max_caster_r = CSM_MAX_CASTER_R;

        let cascade_vp: [Mat4; NUM_SHADOW_CASCADES] = std::array::from_fn(|ci| {
            compute_csm_cascade_vp(csm_centers[ci], shadow_sun_dir, csm_ortho_radii[ci])
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

        let active_octree = if self.world_octree.nodes.is_empty() {
            &geo.octree
        } else {
            &self.world_octree
        };

        // 1. Traverse 3D WorldOctree against Camera View Frustum on CPU
        let mut cam_vis_sectors: Vec<usize> = Vec::with_capacity(geo.sectors.len());
        active_octree.query_frustum(&cam_frustum, &mut cam_vis_sectors);
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
                active_octree.query_frustum(csm_f, &mut csm_vis_sectors);
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
        let mut rider_draw_descs: Vec<GpuDrawDesc> = Vec::with_capacity(96);
        let mut skin_jobs: Vec<SkinJob> = Vec::with_capacity(64);
        let mut skin_bones: Vec<[[f32; 4]; 4]> = Vec::with_capacity(256);
        let mut max_skin_verts_in_job = 0u32;
        let mut skinned_vert_cursor = self.world_offsets.skinned_base_vertex;

        // 1. Fixed static world ObjectData slots (0..5 shared across the entire static world)
        self.push_object_data(device, 0, Mat4::IDENTITY, [1.0; 4], 1, 0.35, 0.0, false); // Road
        self.push_object_data(device, 1, Mat4::IDENTITY, [1.0; 4], 2, 0.68, 0.0, false); // Terrain
        self.push_object_data(device, 2, Mat4::IDENTITY, [1.0; 4], 3, 0.55, 0.0, false); // Wire
        self.push_object_data(device, 3, Mat4::IDENTITY, [1.0; 4], 0, 0.0, 0.0, true);   // Baked prop tiers
        self.push_object_data(device, 4, Mat4::IDENTITY, [1.0; 4], 4, 0.0, 0.0, false);  // Water
        let mut obj_count = 5usize;

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
        self.head_unit_screen_uv = if camera_mode == CameraMode::ChestPov {
            rider_parts
                .iter()
                .find(|p| p.model == ModelId::BikeComputer)
                .and_then(|p| project_head_unit_screen(p.transform, unjittered_view_proj))
        } else {
            None
        };
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
            let sec_count = match self.models.get(&rp.model) {
                Some(m) => m.sections.len(),
                None => continue,
            };
            let (outline_w, crease_s) = match rp.model {
                ModelId::BikeWheel => (0.72, 0.0),
                ModelId::BikeFrame | ModelId::BikeCrank | ModelId::BikeComputer | ModelId::BikeBottleCage => (0.86, 0.0),
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
                let obj_idx = obj_count as u32;
                self.push_object_data(
                    device,
                    obj_count,
                    rp.transform,
                    tint,
                    rp.object_id,
                    outline_w,
                    crease_s,
                    false,
                );
                obj_count += 1;

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

                let (sec_pos, mut sec_half) =
                    transform_aabb(rp.transform, ms.bounds_center, ms.bounds_half_extents);
                if is_skinned {
                    sec_half += Vec3::splat(0.20);
                }
                let sec_radius = sec_half.length().max(0.05);

                rider_draw_descs.push(GpuDrawDesc {
                    pos_radius: [sec_pos.x, sec_pos.y, sec_pos.z, sec_radius],
                    aabb_half: [sec_half.x, sec_half.y, sec_half.z, 0.0],
                    index_count: ms.index_count,
                    first_index: ms.first_index,
                    base_vertex: eff_base_vertex,
                    object_idx: obj_idx,
                    flags: 127u32 | 256u32, // sector 127 (dynamic rider), cast_shadow = true
                    material_bucket: BUCKET_SKINNED as u32,
                    _pad0: 0,
                    _pad1: 0,
                });
            }
        }

        // Upload shared ObjectData[], skinning jobs, bone matrices, and dynamic rider draw descriptors to GPU
        if obj_count > 0 {
            queue.write_buffer(
                &self.obj_buf,
                0,
                bytemuck::cast_slice(&self.objects_staging[..obj_count]),
            );
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

        // Compute non-overlapping contiguous slices in shared indirect_draw_buf
        let rider_count = rider_draw_descs.len() as u32;
        let mut cam_bucket_caps = self.static_bucket_counts;
        cam_bucket_caps[BUCKET_SKINNED] = cam_bucket_caps[BUCKET_SKINNED].saturating_add(rider_count);
        let csm_caps: [u32; NUM_SHADOW_CASCADES] = if graphics.shadows_enabled {
            std::array::from_fn(|ci| self.static_shadow_counts[ci].saturating_add(rider_count))
        } else {
            [0; NUM_SHADOW_CASCADES]
        };
        let (bases, caps, total_slots) = compute_indirect_layout(cam_bucket_caps, csm_caps);

        let shadows_flag = if graphics.shadows_enabled { 1.0_f32 } else { 0.0_f32 };
        let cull_u = MultiPassCullUniforms {
            cam_view_proj: unjittered_view_proj.to_cols_array_2d(),
            cam_planes: cam_frustum.planes,
            csm_planes_0: csm_frustums[0].planes,
            csm_planes_1: csm_frustums[1].planes,
            csm_planes_2: csm_frustums[2].planes,
            cam_pos_dist: [cam_pos.x, cam_pos.y, cam_pos.z, max_prop_dist],
            screen_params: [
                target_width.max(1) as f32,
                target_height.max(1) as f32,
                MIN_PROJECTED_BBOX_PX,
                0.0,
            ],
            csm_min_r: [
                csm_min_caster_r[0],
                csm_min_caster_r[1],
                csm_min_caster_r[2],
                shadows_flag,
            ],
            csm_max_r: [
                csm_max_caster_r[0],
                csm_max_caster_r[1],
                csm_max_caster_r[2],
                0.0,
            ],
            cam_sector_mask,
            csm_sector_mask_0: csm_sector_masks[0],
            csm_sector_mask_1: csm_sector_masks[1],
            csm_sector_mask_2: csm_sector_masks[2],
            bucket_bases: [
                [bases[0], bases[1], bases[2], bases[3]],
                [bases[4], bases[5], bases[6], bases[7]],
            ],
            bucket_caps: [
                [caps[0], caps[1], caps[2], caps[3]],
                [caps[4], caps[5], caps[6], caps[7]],
            ],
            cmd_info: [
                cmd_count,
                if self.supports_indirect_first_instance { 1 } else { 0 },
                0,
                0,
            ],
        };
        queue.write_buffer(&self.cull_uniform_buf, 0, bytemuck::bytes_of(&cull_u));

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("cadencia_encoder"),
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

        let indirect_cmd_size = std::mem::size_of::<DrawIndexedIndirectCmd>() as u64;

        // -----------------------------------------------------------------
        // Pass 0b: Single-Dispatch GPU Culling & Stream Compaction (cull_indirect.wgsl)
        // -----------------------------------------------------------------
        if self.supports_indirect_first_instance {
            encoder.clear_buffer(&self.draw_counts_buf, 0, None);
            if !self.supports_mdi_count && total_slots > 0 {
                encoder.clear_buffer(
                    &self.indirect_draw_buf,
                    0,
                    Some((total_slots as u64) * indirect_cmd_size),
                );
            }
        }
        if cmd_count > 0 {
            let workgroups = cmd_count.div_ceil(64);
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("cull_indirect_single_dispatch_pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.cull_pipeline);
            cpass.set_bind_group(0, &self.cull_bg, &[]);
            cpass.dispatch_workgroups(workgroups, 1, 1);
        }

        // -----------------------------------------------------------------
        // Passes 1..3: 3-Cascade Directional Shadow Maps (1 MDI call per cascade)
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
            let csm_cap = caps[NUM_MATERIAL_BUCKETS + ci];
            if graphics.shadows_enabled && csm_cap > 0 {
                rpass.set_pipeline(&self.shadow_pipeline);
                rpass.set_bind_group(0, &self.shadow_frame_bgs[ci], &[]);
                rpass.set_bind_group(1, &self.obj_bg, &[]);
                rpass.set_vertex_buffer(0, self.mega_vb.slice(..));
                rpass.set_index_buffer(self.mega_ib.slice(..), wgpu::IndexFormat::Uint32);
                if self.supports_indirect_first_instance {
                    rpass.set_vertex_buffer(1, self.instance_id_vb.slice(..));
                    self.submit_indirect_bucket(
                        &mut rpass,
                        bases[NUM_MATERIAL_BUCKETS + ci],
                        csm_cap,
                        (NUM_MATERIAL_BUCKETS + ci) as u32,
                    );
                } else {
                    let mut last_obj_idx: Option<u32> = None;
                    let csm_mask = &csm_sector_masks[ci];
                    let cmd_count_usize = cmd_count as usize;
                    for (cmd_id, desc) in self
                        .static_draw_descs
                        .iter()
                        .chain(rider_draw_descs.iter())
                        .enumerate()
                    {
                        if (desc.flags & 256) != 0
                            && is_sector_bit(csm_mask, (desc.flags & 127) as usize)
                        {
                            if last_obj_idx != Some(desc.object_idx) {
                                rpass.set_vertex_buffer(
                                    1,
                                    self.instance_id_vb.slice((desc.object_idx as u64 * 4)..),
                                );
                                last_obj_idx = Some(desc.object_idx);
                            }
                            let indirect_offset = (((1 + ci) * cmd_count_usize + cmd_id) as u64)
                                * indirect_cmd_size;
                            rpass.draw_indexed_indirect(&self.indirect_draw_buf, indirect_offset);
                        }
                    }
                }
            }
        }

        // -----------------------------------------------------------------
        // Pass 4: Pure Unlit G-Buffer Pass (1 Indirect Call Per Material Type)
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

            // 4b. Bind shared frame, objects storage buffer, vertex/index buffers ONCE
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
            rpass.set_bind_group(1, &self.obj_bg, &[]);
            rpass.set_vertex_buffer(0, self.mega_vb.slice(..));
            rpass.set_index_buffer(self.mega_ib.slice(..), wgpu::IndexFormat::Uint32);

            if self.supports_indirect_first_instance {
                rpass.set_vertex_buffer(1, self.instance_id_vb.slice(..));
                let bucket_pipelines: [(usize, &wgpu::RenderPipeline); NUM_MATERIAL_BUCKETS] = [
                    (BUCKET_ROAD, road_pipe),
                    (BUCKET_TERRAIN, terrain_pipe),
                    (BUCKET_PROP, &self.gbuf_prop_pipeline),
                    (BUCKET_SKINNED, &self.gbuf_skinned_pipeline),
                    (BUCKET_WATER, &self.gbuf_water_pipeline),
                ];
                for (bucket_idx, pipeline) in bucket_pipelines {
                    let cap = caps[bucket_idx];
                    if cap > 0 {
                        rpass.set_pipeline(pipeline);
                        self.submit_indirect_bucket(
                            &mut rpass,
                            bases[bucket_idx],
                            cap,
                            bucket_idx as u32,
                        );
                    }
                }
            } else {
                let mut last_bucket: Option<u32> = None;
                let mut last_obj_idx: Option<u32> = None;
                for (cmd_id, desc) in self
                    .static_draw_descs
                    .iter()
                    .chain(rider_draw_descs.iter())
                    .enumerate()
                {
                    if !is_sector_bit(&cam_sector_mask, (desc.flags & 127) as usize) {
                        continue;
                    }
                    if last_bucket != Some(desc.material_bucket) {
                        let pipeline = match desc.material_bucket as usize {
                            BUCKET_ROAD => road_pipe,
                            BUCKET_TERRAIN => terrain_pipe,
                            BUCKET_PROP => &self.gbuf_prop_pipeline,
                            BUCKET_SKINNED => &self.gbuf_skinned_pipeline,
                            _ => &self.gbuf_water_pipeline,
                        };
                        rpass.set_pipeline(pipeline);
                        last_bucket = Some(desc.material_bucket);
                    }
                    if last_obj_idx != Some(desc.object_idx) {
                        rpass.set_vertex_buffer(
                            1,
                            self.instance_id_vb.slice((desc.object_idx as u64 * 4)..),
                        );
                        last_obj_idx = Some(desc.object_idx);
                    }
                    let indirect_offset = (cmd_id as u64) * indirect_cmd_size;
                    rpass.draw_indexed_indirect(&self.indirect_draw_buf, indirect_offset);
                }
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
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpu_struct_sizes_match_wgsl_layout() {
        assert_eq!(std::mem::size_of::<ObjectData>(), 160);
        assert_eq!(std::mem::size_of::<GpuDrawDesc>(), 64);
        assert_eq!(std::mem::size_of::<DrawIndexedIndirectCmd>(), 20);
        assert_eq!(std::mem::size_of::<MultiPassCullUniforms>(), 656);
        assert_eq!(std::mem::size_of::<MultiPassCullUniforms>() % 16, 0);
    }

    #[test]
    fn projected_aabb_culling_drops_tiny_meshes_and_keeps_visible_ones() {
        let view = Mat4::look_at_rh(Vec3::ZERO, Vec3::NEG_Z, Vec3::Y);
        let proj = Mat4::perspective_rh(50.0_f32.to_radians(), 1280.0 / 960.0, 0.25, 1900.0);
        let vp = proj * view;

        // Distant 0.2m cube at 200m: < 8 px -> dropped
        let tiny_px = projected_aabb_max_extent_px(
            vp,
            Vec3::new(0.0, 0.0, -200.0),
            Vec3::splat(0.1),
            1280.0,
            960.0,
        );
        assert!(tiny_px < MIN_PROJECTED_BBOX_PX, "expected < 8 px, got {tiny_px}");

        // Midground 1.0m box at 40m: >> 8 px -> kept
        let mid_px = projected_aabb_max_extent_px(
            vp,
            Vec3::new(0.0, 0.0, -40.0),
            Vec3::splat(0.5),
            1280.0,
            960.0,
        );
        assert!(mid_px >= MIN_PROJECTED_BBOX_PX, "expected >= 8 px, got {mid_px}");

        // Thin tall post at 60m (0.04m wide, 2.0m tall): max(w, h) >= 8 px -> kept
        let pole_px = projected_aabb_max_extent_px(
            vp,
            Vec3::new(0.0, 0.0, -60.0),
            Vec3::new(0.02, 1.0, 0.02),
            1280.0,
            960.0,
        );
        assert!(pole_px >= MIN_PROJECTED_BBOX_PX, "expected >= 8 px, got {pole_px}");

        // Box straddling near plane (z = 0): returns INFINITY -> kept
        let near_px = projected_aabb_max_extent_px(
            vp,
            Vec3::new(0.0, 0.0, -0.2),
            Vec3::splat(0.5),
            1280.0,
            960.0,
        );
        assert!(near_px.is_infinite());

        // transform_aabb with 90-degree Y rotation swaps X/Z half-extents cleanly
        let rot_y = Mat4::from_translation(Vec3::new(10.0, 2.0, -5.0))
            * Mat4::from_rotation_y(std::f32::consts::FRAC_PI_2);
        let (wc, wh) = transform_aabb(rot_y, Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.5, 1.0, 2.0));
        assert!((wc - Vec3::new(10.0, 2.0, -6.0)).length() < 1e-4);
        assert!((wh - Vec3::new(2.0, 1.0, 0.5)).length() < 1e-4);
    }

    #[test]
    fn indirect_layout_partitions_without_overlap() {
        // Campo case: water bucket has 0 capacity, shadows enabled
        let cam_caps = [20, 20, 45, 30, 0];
        let csm_caps = [40, 35, 32];
        let (bases, caps, total) = compute_indirect_layout(cam_caps, csm_caps);
        assert_eq!(caps, [20, 20, 45, 30, 0, 40, 35, 32]);
        assert_eq!(bases, [0, 20, 40, 85, 115, 115, 155, 190]);
        assert_eq!(total, 222);
        for i in 0..(NUM_DRAW_COUNT_SLOTS - 1) {
            assert_eq!(bases[i] + caps[i], bases[i + 1]);
        }

        // Shadows disabled: csm_caps are all 0
        let (bases_no_shad, caps_no_shad, total_no_shad) =
            compute_indirect_layout([10, 10, 25, 12, 1], [0, 0, 0]);
        assert_eq!(caps_no_shad, [10, 10, 25, 12, 1, 0, 0, 0]);
        assert_eq!(bases_no_shad, [0, 10, 20, 45, 57, 58, 58, 58]);
        assert_eq!(total_no_shad, 58);
    }
}

