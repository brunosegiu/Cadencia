struct SinglePassCullUniforms {
    frustum_planes: array<vec4<f32>, 6>, // 6 normalized Gribb-Hartmann frustum planes for this pass
    filter_params: vec4<f32>,            // color pass: [cam.x, cam.y, cam.z, max_draw_dist]; shadow pass: [min_caster_r, max_caster_r, shadows_enabled, 0.0]
    sector_mask: vec4<u32>,              // 128-bit bitmask of CPU-octree-culled visible sectors for this pass
    cmd_info: vec4<u32>,                 // x = command_count, y = is_shadow_pass (0 = color view, 1 = CSM cascade), z = 0, w = 0
};

struct GpuDrawDesc {
    pos_radius: vec4<f32>, // xyz = world bounding center, w = world bounding radius
    index_count: u32,
    first_index: u32,
    base_vertex: i32,
    flags: u32,            // bits 0..6 = sector_id (0..126 or 127 for riders), bit 8 = cast_shadow, bit 9 = is_sector_surface, bits 16..31 = rep_caster_cm
};

struct DrawIndexedIndirectCmd {
    index_count: u32,
    instance_count: u32,
    first_index: u32,
    base_vertex: i32,
    first_instance: u32,
};

@group(0) @binding(0) var<uniform> cull: SinglePassCullUniforms;
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
    let sector_vis = is_sector_bit_set(cull.sector_mask, sector_id);

    var visible = false;
    if (cull.cmd_info.y == 0u) {
        // Color G-Buffer View Culling Dispatch
        let dist_cam = distance(pos, cull.filter_params.xyz);
        visible = sector_vis &&
            (is_sector_surface || (dist_cam <= cull.filter_params.w + world_r)) &&
            frustum_contains_sphere(cull.frustum_planes, pos, world_r);
    } else {
        // Single CSM Shadow Cascade Culling Dispatch
        let caster_r = select(world_r, f32(raw_caster_cm) * 0.01, raw_caster_cm > 0u);
        visible = (cull.filter_params.z > 0.5) &&
            cast_shadow &&
            sector_vis &&
            (caster_r >= cull.filter_params.x) &&
            (caster_r <= cull.filter_params.y) &&
            frustum_contains_sphere(cull.frustum_planes, pos, world_r);
    }

    indirect_cmds[cmd_id] = DrawIndexedIndirectCmd(
        desc.index_count,
        select(0u, 1u, visible),
        desc.first_index,
        desc.base_vertex,
        0u
    );
}
