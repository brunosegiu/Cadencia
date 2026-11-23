struct SectorCullUniforms {
    camera_pos_max_dist: vec4<f32>,         // xyz = camera_pos, w = max_prop_draw_distance_m
    cam_frustum_planes: array<vec4<f32>, 6>, // 6 normalized camera view-projection frustum planes
    csm0_frustum_planes: array<vec4<f32>, 6>,// 6 normalized CSM Cascade 0 orthographic frustum planes
    csm1_frustum_planes: array<vec4<f32>, 6>,// 6 normalized CSM Cascade 1 orthographic frustum planes
    csm2_frustum_planes: array<vec4<f32>, 6>,// 6 normalized CSM Cascade 2 orthographic frustum planes
    csm_min_caster_r: vec4<f32>,            // xyz = min caster bounding radius per cascade, w = shadows_enabled
    csm_max_caster_r: vec4<f32>,            // xyz = max caster bounding radius per cascade, w = 0.0
    cam_sector_mask: vec4<u32>,             // 128-bit bitmask of CPU-octree-culled visible camera sectors
    shadow_sector_mask: vec4<u32>,          // 128-bit bitmask of CPU-octree-culled visible shadow sectors
    cmd_info: vec4<u32>,                    // x = command_count, y = pass_stride, z = 0, w = 0
};

struct GpuDrawDesc {
    pos_radius: vec4<f32>, // xyz = world bounding center, w = world bounding radius
    index_count: u32,
    first_index: u32,
    base_vertex: i32,
    flags: u32,            // bits 0..6 = sector_id (0..126 or 127 for riders), bit 8 = cast_shadow, bit 9 = is_sector_surface
};

struct DrawIndexedIndirectCmd {
    index_count: u32,
    instance_count: u32,
    first_index: u32,
    base_vertex: i32,
    first_instance: u32,
};

@group(0) @binding(0) var<uniform> cull: SectorCullUniforms;
@group(0) @binding(1) var<storage, read> draw_descs: array<GpuDrawDesc>;
@group(0) @binding(2) var<storage, read_write> indirect_cmds: array<DrawIndexedIndirectCmd>;

fn is_sector_bit_set(mask: vec4<u32>, sector_id: u32) -> bool {
    let word = sector_id >> 5u;
    let bit = sector_id & 31u;
    var w_val = mask.x;
    if (word == 1u) {
        w_val = mask.y;
    } else if (word == 2u) {
        w_val = mask.z;
    } else if (word >= 3u) {
        w_val = mask.w;
    }
    return ((w_val >> bit) & 1u) != 0u;
}

fn frustum_contains_sphere(planes: array<vec4<f32>, 6>, center: vec3<f32>, radius: f32) -> bool {
    for (var i = 0u; i < 6u; i = i + 1u) {
        let p = planes[i];
        if (dot(p.xyz, center) + p.w < -radius) {
            return false;
        }
    }
    return true;
}

@compute @workgroup_size(64, 1, 1)
fn cs_build_indirect_commands(@builtin(global_invocation_id) gid: vec3<u32>) {
    let cmd_id = gid.x;
    let count = cull.cmd_info.x;
    let stride = cull.cmd_info.y;
    if (cmd_id >= count) {
        return;
    }

    let desc = draw_descs[cmd_id];
    let sector_id = desc.flags & 127u;
    let cast_shadow = (desc.flags & 256u) != 0u;
    let is_sector_surface = (desc.flags & 512u) != 0u;
    let raw_caster_cm = desc.flags >> 16u;

    let pos = desc.pos_radius.xyz;
    let world_r = desc.pos_radius.w;
    let caster_r = select(world_r, f32(raw_caster_cm) * 0.01, raw_caster_cm > 0u);

    // 1. Scene G-Buffer Pass Indirect Command:
    //    - Sector must be registered visible by CPU Octree frustum traversal
    //    - Draw call bounding sphere must pass 6-plane GPU Camera Frustum test + draw distance
    let cam_sec_vis = is_sector_bit_set(cull.cam_sector_mask, sector_id);
    let dist_cam = distance(pos, cull.camera_pos_max_dist.xyz);
    let in_cam_frustum = frustum_contains_sphere(cull.cam_frustum_planes, pos, world_r);
    let vis_scene = cam_sec_vis &&
        in_cam_frustum &&
        (is_sector_surface || (dist_cam <= cull.camera_pos_max_dist.w + world_r));

    indirect_cmds[cmd_id] = DrawIndexedIndirectCmd(
        desc.index_count,
        select(0u, 1u, vis_scene),
        desc.first_index,
        desc.base_vertex,
        0u
    );

    // 2. CSM Shadow Cascade Passes (passes 1..=3):
    //    - Sector must be registered visible by CPU Octree traversal
    //    - Draw call caster size must pass per-cascade bounding-radius size filter AND 6-plane GPU CSM Frustum test
    let shadows_on = cull.csm_min_caster_r.w > 0.5 &&
        cast_shadow &&
        is_sector_bit_set(cull.shadow_sector_mask, sector_id);

    // Cascade 0
    let vis_c0 = shadows_on &&
        caster_r >= cull.csm_min_caster_r.x &&
        caster_r <= cull.csm_max_caster_r.x &&
        frustum_contains_sphere(cull.csm0_frustum_planes, pos, world_r);
    indirect_cmds[stride + cmd_id] = DrawIndexedIndirectCmd(
        desc.index_count,
        select(0u, 1u, vis_c0),
        desc.first_index,
        desc.base_vertex,
        0u
    );

    // Cascade 1
    let vis_c1 = shadows_on &&
        caster_r >= cull.csm_min_caster_r.y &&
        caster_r <= cull.csm_max_caster_r.y &&
        frustum_contains_sphere(cull.csm1_frustum_planes, pos, world_r);
    indirect_cmds[2u * stride + cmd_id] = DrawIndexedIndirectCmd(
        desc.index_count,
        select(0u, 1u, vis_c1),
        desc.first_index,
        desc.base_vertex,
        0u
    );

    // Cascade 2
    let vis_c2 = shadows_on &&
        caster_r >= cull.csm_min_caster_r.z &&
        caster_r <= cull.csm_max_caster_r.z &&
        frustum_contains_sphere(cull.csm2_frustum_planes, pos, world_r);
    indirect_cmds[3u * stride + cmd_id] = DrawIndexedIndirectCmd(
        desc.index_count,
        select(0u, 1u, vis_c2),
        desc.first_index,
        desc.base_vertex,
        0u
    );
}
