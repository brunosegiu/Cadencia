struct MultiPassCullUniforms {
    cam_view_proj: mat4x4<f32>,
    cam_planes: array<vec4<f32>, 6>,
    csm_planes_0: array<vec4<f32>, 6>,
    csm_planes_1: array<vec4<f32>, 6>,
    csm_planes_2: array<vec4<f32>, 6>,
    cam_pos_dist: vec4<f32>,            // [cam.x, cam.y, cam.z, max_prop_dist]
    screen_params: vec4<f32>,           // [viewport_w_px, viewport_h_px, min_projected_px, 0.0]
    csm_min_r: vec4<f32>,               // [min_r0, min_r1, min_r2, shadows_enabled]
    csm_max_r: vec4<f32>,               // [max_r0, max_r1, max_r2, 0.0]
    cam_sector_mask: vec4<u32>,
    csm_sector_mask_0: vec4<u32>,
    csm_sector_mask_1: vec4<u32>,
    csm_sector_mask_2: vec4<u32>,
    bucket_bases: array<vec4<u32>, 2>,  // [0]=[road,terrain,prop,skinned], [1]=[water,csm0,csm1,csm2]
    bucket_caps: array<vec4<u32>, 2>,   // capacities for bounds guard
    cmd_info: vec4<u32>,                // [cmd_count, compact_enabled (0/1), 0, 0]
};

struct GpuDrawDesc {
    pos_radius: vec4<f32>, // xyz = world bounding center, w = world bounding radius
    aabb_half: vec4<f32>,  // xyz = world AABB half-extents, w = 0.0
    index_count: u32,
    first_index: u32,
    base_vertex: i32,
    object_idx: u32,       // shared ObjectData[] index -> written to first_instance
    flags: u32,            // bits 0..6 = sector_id (0..126 or 127), bit 8 = cast_shadow, bit 9 = is_sector_surface, bits 16..31 = rep_caster_cm
    material_bucket: u32,  // 0 = Road, 1 = Terrain, 2 = Prop, 3 = Skinned, 4 = Water
    _pad0: u32,
    _pad1: u32,
};

struct DrawIndexedIndirectCmd {
    index_count: u32,
    instance_count: u32,
    first_index: u32,
    base_vertex: i32,
    first_instance: u32,
};

@group(0) @binding(0) var<uniform> cull: MultiPassCullUniforms;
@group(0) @binding(1) var<storage, read> draw_descs: array<GpuDrawDesc>;
@group(0) @binding(2) var<storage, read_write> indirect_cmds: array<DrawIndexedIndirectCmd>;
@group(0) @binding(3) var<storage, read_write> draw_counts: array<atomic<u32>, 8>;

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

fn get_u32_8(v: array<vec4<u32>, 2>, idx: u32) -> u32 {
    let vec_val = select(v[0], v[1], idx >= 4u);
    let lane = idx & 3u;
    if (lane == 0u) {
        return vec_val.x;
    } else if (lane == 1u) {
        return vec_val.y;
    } else if (lane == 2u) {
        return vec_val.z;
    } else {
        return vec_val.w;
    }
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

fn projected_aabb_meets_min_px(center: vec3<f32>, half: vec3<f32>) -> bool {
    var min_ndc = vec2<f32>(1e9, 1e9);
    var max_ndc = vec2<f32>(-1e9, -1e9);
    for (var i = 0u; i < 8u; i = i + 1u) {
        let sx = select(-1.0, 1.0, (i & 1u) != 0u);
        let sy = select(-1.0, 1.0, (i & 2u) != 0u);
        let sz = select(-1.0, 1.0, (i & 4u) != 0u);
        let corner = center + vec3<f32>(sx * half.x, sy * half.y, sz * half.z);
        let clip = cull.cam_view_proj * vec4<f32>(corner, 1.0);
        if (clip.w <= 0.001) {
            return true;
        }
        let ndc = clip.xy / clip.w;
        min_ndc = min(min_ndc, ndc);
        max_ndc = max(max_ndc, ndc);
    }
    let ext_px = (max_ndc - min_ndc) * 0.5 * cull.screen_params.xy;
    return max(ext_px.x, ext_px.y) >= cull.screen_params.z;
}

fn eval_csm_visible(
    planes: array<vec4<f32>, 6>,
    sector_mask: vec4<u32>,
    min_r: f32,
    max_r: f32,
    sector_id: u32,
    cast_shadow: bool,
    pos: vec3<f32>,
    world_r: f32,
    caster_r: f32,
) -> bool {
    return (cull.csm_min_r.w > 0.5) &&
        cast_shadow &&
        is_sector_bit_set(sector_mask, sector_id) &&
        (caster_r >= min_r) &&
        (caster_r <= max_r) &&
        frustum_contains_sphere(planes, pos, world_r);
}

fn emit_csm_cascade(
    ci: u32,
    cmd_id: u32,
    cmd_count: u32,
    compact_enabled: bool,
    visible: bool,
    desc: GpuDrawDesc,
) {
    if (compact_enabled) {
        if (visible) {
            let count_idx = 5u + ci;
            let slot = atomicAdd(&draw_counts[count_idx], 1u);
            if (slot < get_u32_8(cull.bucket_caps, count_idx)) {
                let dst = get_u32_8(cull.bucket_bases, count_idx) + slot;
                indirect_cmds[dst] = DrawIndexedIndirectCmd(
                    desc.index_count,
                    1u,
                    desc.first_index,
                    desc.base_vertex,
                    desc.object_idx
                );
            }
        }
    } else {
        let dst = (1u + ci) * cmd_count + cmd_id;
        indirect_cmds[dst] = DrawIndexedIndirectCmd(
            desc.index_count,
            select(0u, 1u, visible),
            desc.first_index,
            desc.base_vertex,
            0u
        );
    }
}

@compute @workgroup_size(64, 1, 1)
fn cs_build_indirect_commands(@builtin(global_invocation_id) gid: vec3<u32>) {
    let cmd_id = gid.x;
    let cmd_count = cull.cmd_info.x;
    if (cmd_id >= cmd_count) {
        return;
    }

    let compact_enabled = cull.cmd_info.y != 0u;
    let desc = draw_descs[cmd_id];
    let sector_id = desc.flags & 127u;
    let cast_shadow = (desc.flags & 256u) != 0u;
    let is_sector_surface = (desc.flags & 512u) != 0u;
    let raw_caster_cm = desc.flags >> 16u;

    let pos = desc.pos_radius.xyz;
    let world_r = desc.pos_radius.w;
    let caster_r = select(world_r, f32(raw_caster_cm) * 0.01, raw_caster_cm > 0u);

    // 1. Camera G-Buffer View Culling (sector bitmask + draw distance + frustum + projected AABB >= 8 px)
    let dist_cam = distance(pos, cull.cam_pos_dist.xyz);
    let cam_vis = is_sector_bit_set(cull.cam_sector_mask, sector_id) &&
        (is_sector_surface || (dist_cam <= cull.cam_pos_dist.w + world_r)) &&
        frustum_contains_sphere(cull.cam_planes, pos, world_r) &&
        projected_aabb_meets_min_px(pos, desc.aabb_half.xyz);

    if (compact_enabled) {
        if (cam_vis) {
            let b = desc.material_bucket;
            let slot = atomicAdd(&draw_counts[b], 1u);
            if (slot < get_u32_8(cull.bucket_caps, b)) {
                let dst = get_u32_8(cull.bucket_bases, b) + slot;
                indirect_cmds[dst] = DrawIndexedIndirectCmd(
                    desc.index_count,
                    1u,
                    desc.first_index,
                    desc.base_vertex,
                    desc.object_idx
                );
            }
        }
    } else {
        indirect_cmds[cmd_id] = DrawIndexedIndirectCmd(
            desc.index_count,
            select(0u, 1u, cam_vis),
            desc.first_index,
            desc.base_vertex,
            0u
        );
    }

    // 2. CSM Shadow Cascades 0, 1, 2 Culling
    let csm0_vis = eval_csm_visible(
        cull.csm_planes_0,
        cull.csm_sector_mask_0,
        cull.csm_min_r.x,
        cull.csm_max_r.x,
        sector_id,
        cast_shadow,
        pos,
        world_r,
        caster_r
    );
    emit_csm_cascade(0u, cmd_id, cmd_count, compact_enabled, csm0_vis, desc);

    let csm1_vis = eval_csm_visible(
        cull.csm_planes_1,
        cull.csm_sector_mask_1,
        cull.csm_min_r.y,
        cull.csm_max_r.y,
        sector_id,
        cast_shadow,
        pos,
        world_r,
        caster_r
    );
    emit_csm_cascade(1u, cmd_id, cmd_count, compact_enabled, csm1_vis, desc);

    let csm2_vis = eval_csm_visible(
        cull.csm_planes_2,
        cull.csm_sector_mask_2,
        cull.csm_min_r.z,
        cull.csm_max_r.z,
        sector_id,
        cast_shadow,
        pos,
        world_r,
        caster_r
    );
    emit_csm_cascade(2u, cmd_id, cmd_count, compact_enabled, csm2_vis, desc);
}
