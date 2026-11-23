struct SectorCullUniforms {
    camera_pos_max_dist: vec4<f32>, // xyz = camera_pos, w = max_prop_draw_distance_m
    csm_center_r0: vec4<f32>,       // xyz = c0_center, w = c0_ortho_bound_r
    csm_center_r1: vec4<f32>,       // xyz = c1_center, w = c1_ortho_bound_r
    csm_center_r2: vec4<f32>,       // xyz = c2_center, w = c2_ortho_bound_r
    csm_min_caster_r: vec4<f32>,    // xyz = min caster radius per cascade, w = shadows_enabled (1.0 or 0.0)
    csm_max_caster_r: vec4<f32>,    // xyz = max caster radius per cascade, w = 0.0
    active_sector_info: vec4<u32>,  // x = active_sector_bitmask, y = command_count, z = pass_stride, w = 0
};

struct GpuDrawDesc {
    pos_radius: vec4<f32>, // xyz = world bounding center, w = world bounding radius
    index_count: u32,
    first_index: u32,
    base_vertex: i32,
    flags: u32,            // bits 0..7 = sector_id (0..15 or 31 for riders), bit 8 = cast_shadow, bit 9 = is_sector_surface
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

@compute @workgroup_size(64, 1, 1)
fn cs_build_indirect_commands(@builtin(global_invocation_id) gid: vec3<u32>) {
    let cmd_id = gid.x;
    let count = cull.active_sector_info.y;
    let stride = cull.active_sector_info.z;
    if (cmd_id >= count) {
        return;
    }

    let desc = draw_descs[cmd_id];
    let sector_id = desc.flags & 31u;
    let cast_shadow = (desc.flags & 256u) != 0u;
    let is_sector_surface = (desc.flags & 512u) != 0u;

    let sector_active = ((cull.active_sector_info.x >> sector_id) & 1u) != 0u;
    let pos = desc.pos_radius.xyz;
    let world_r = desc.pos_radius.w;

    // 1. Scene G-Buffer Pass Indirect Command (pass 0: slot = cmd_id)
    let dist_cam = distance(pos, cull.camera_pos_max_dist.xyz);
    let vis_scene = sector_active && (is_sector_surface || (dist_cam <= cull.camera_pos_max_dist.w + world_r));
    indirect_cmds[cmd_id] = DrawIndexedIndirectCmd(
        desc.index_count,
        select(0u, 1u, vis_scene),
        desc.first_index,
        desc.base_vertex,
        0u
    );

    // 2. CSM Shadow Cascade Passes (passes 1..=3: slot = (ci + 1) * stride + cmd_id)
    let shadows_on = cull.csm_min_caster_r.w > 0.5 && sector_active && cast_shadow;

    // Cascade 0
    let vis_c0 = shadows_on &&
        world_r >= cull.csm_min_caster_r.x &&
        world_r <= cull.csm_max_caster_r.x &&
        distance(pos, cull.csm_center_r0.xyz) <= (cull.csm_center_r0.w + world_r);
    indirect_cmds[stride + cmd_id] = DrawIndexedIndirectCmd(
        desc.index_count,
        select(0u, 1u, vis_c0),
        desc.first_index,
        desc.base_vertex,
        0u
    );

    // Cascade 1
    let vis_c1 = shadows_on &&
        world_r >= cull.csm_min_caster_r.y &&
        world_r <= cull.csm_max_caster_r.y &&
        distance(pos, cull.csm_center_r1.xyz) <= (cull.csm_center_r1.w + world_r);
    indirect_cmds[2u * stride + cmd_id] = DrawIndexedIndirectCmd(
        desc.index_count,
        select(0u, 1u, vis_c1),
        desc.first_index,
        desc.base_vertex,
        0u
    );

    // Cascade 2
    let vis_c2 = shadows_on &&
        world_r >= cull.csm_min_caster_r.z &&
        world_r <= cull.csm_max_caster_r.z &&
        distance(pos, cull.csm_center_r2.xyz) <= (cull.csm_center_r2.w + world_r);
    indirect_cmds[3u * stride + cmd_id] = DrawIndexedIndirectCmd(
        desc.index_count,
        select(0u, 1u, vis_c2),
        desc.first_index,
        desc.base_vertex,
        0u
    );
}
