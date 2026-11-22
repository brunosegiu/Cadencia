struct FrameUniforms {
    view_proj: mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    view_mat: mat4x4<f32>,
    light_view_proj: mat4x4<f32>,
    camera_pos: vec4<f32>,
    sun_dir: vec4<f32>,
    sun_color: vec4<f32>,
    sky_zenith: vec4<f32>,
    sky_horizon: vec4<f32>,
    fog_color: vec4<f32>,
    screen_params: vec4<f32>,
    outline_params: vec4<f32>,
};

@group(0) @binding(0) var<uniform> frame: FrameUniforms;

struct SkyOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) ndc: vec2<f32>,
};

struct GBufferOut {
    @location(0) lit_color: vec4<f32>,
    @location(1) normal_id: vec4<f32>,
    @location(2) depth_meta: vec4<f32>,
};

fn hash2(p: vec2<f32>) -> f32 {
    let q = fract(p * vec2<f32>(123.34, 456.21));
    return fract(dot(q, q + 45.32));
}

fn noise2(p: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    let a = hash2(i);
    let b = hash2(i + vec2<f32>(1.0, 0.0));
    let c = hash2(i + vec2<f32>(0.0, 1.0));
    let d = hash2(i + vec2<f32>(1.0, 1.0));
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

fn fbm(p_in: vec2<f32>) -> f32 {
    var p = p_in;
    var val = 0.0;
    var amp = 0.5;
    for (var i = 0; i < 4; i = i + 1) {
        val += amp * noise2(p);
        p = p * 2.04 + vec2<f32>(17.3, 9.2);
        amp *= 0.5;
    }
    return val;
}

@vertex
fn vs_sky(@builtin(vertex_index) idx: u32) -> SkyOut {
    let pos = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 3.0, -1.0),
        vec2<f32>(-1.0,  3.0)
    );
    let p = pos[idx];
    var out: SkyOut;
    out.clip_pos = vec4<f32>(p, 1.0, 1.0);
    out.ndc = p;
    return out;
}

@fragment
fn fs_sky(in: SkyOut) -> GBufferOut {
    let world_far = frame.inv_view_proj * vec4<f32>(in.ndc, 1.0, 1.0);
    let dir = normalize(world_far.xyz / world_far.w - frame.camera_pos.xyz);

    let y_t = clamp(dir.y * 1.9 + 0.05, 0.0, 1.0);
    var sky = mix(frame.sky_horizon.rgb, frame.sky_zenith.rgb, pow(y_t, 0.65));

    // Sun glow + crisp graphic-novel sunset disc (prominent on Coastal Promenade)
    let sun_d = max(dot(dir, normalize(frame.sun_dir.xyz)), 0.0);
    let halo = pow(sun_d, 10.0) * 0.30 + pow(sun_d, 72.0) * 0.38;
    sky += frame.sun_color.rgb * halo;

    let disc_strength = frame.sun_dir.w;
    if (disc_strength > 0.01) {
        let disc = smoothstep(0.9985, 0.9993, sun_d);
        sky = mix(sky, vec3<f32>(1.0, 0.95, 0.74), disc * disc_strength);
    }

    // Painterly Ligne Claire cumulus clouds (un-inked)
    if (dir.y > 0.02) {
        let proj = dir.xz / (dir.y + 0.18) * 1.4 + vec2<f32>(frame.camera_pos.w * 0.008, 1.7);
        let n1 = fbm(proj * 1.25);
        let n2 = fbm(proj * 2.5 + vec2<f32>(n1 * 0.7, -n1 * 0.4));
        let cloud_cov = smoothstep(0.46, 0.66, n1 * 0.68 + n2 * 0.32);
        let cloud_lit = smoothstep(0.42, 0.72, n2 + sun_d * 0.25);
        let cloud_shadow = mix(frame.sky_horizon.rgb * 0.92, vec3<f32>(0.76, 0.77, 0.80), 0.5);
        let cloud_cream = mix(vec3<f32>(0.97, 0.94, 0.87), frame.sun_color.rgb * 1.05, 0.28);
        let cloud_col = mix(cloud_shadow, cloud_cream, cloud_lit);
        let horizon_fade = smoothstep(0.02, 0.24, dir.y);
        sky = mix(sky, cloud_col, cloud_cov * 0.85 * horizon_fade);
    }

    var out: GBufferOut;
    out.lit_color = vec4<f32>(clamp(sky, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
    // Neutral sky normal & outline_weight = 0.0 so sky/clouds never get ink strokes
    out.normal_id = vec4<f32>(0.5, 0.5, 1.0, 0.0);
    out.depth_meta = vec4<f32>(1.0, 1.0, 0.0, 0.0);
    return out;
}
