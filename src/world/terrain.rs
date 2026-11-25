use glam::{Mat4, Vec3};

use crate::assets::{ModelId, SceneVertex};
use crate::world::{BiomeId, RouteSpline};

pub const OCTREE_MAX_DEPTH: usize = 4;
pub const MAX_OCTREE_SECTORS: usize = 126; // fits in 128-bit mask [u32; 4] with bit 127 reserved for dynamic riders

/// 6-plane view/light frustum extracted via Gribb-Hartmann method.
/// Each plane `[nx, ny, nz, d]` has inward-pointing unit normal `(nx, ny, nz)`
/// such that a point `P` is inside when `dot(n, P) + d >= 0`.
#[derive(Debug, Clone, Copy)]
pub struct Frustum {
    pub planes: [[f32; 4]; 6],
}

impl Frustum {
    pub fn from_view_proj(vp: Mat4) -> Self {
        let r0 = vp.row(0);
        let r1 = vp.row(1);
        let r2 = vp.row(2);
        let r3 = vp.row(3);

        let raw = [
            r3 + r0, // Left
            r3 - r0, // Right
            r3 + r1, // Bottom
            r3 - r1, // Top
            r2,      // Near (wgpu 0..1 depth)
            r3 - r2, // Far
        ];

        let mut planes = [[0.0_f32; 4]; 6];
        for (i, p) in raw.iter().enumerate() {
            let len = Vec3::new(p.x, p.y, p.z).length().max(1e-6);
            planes[i] = [p.x / len, p.y / len, p.z / len, p.w / len];
        }
        Self { planes }
    }

    #[inline]
    pub fn intersects_aabb(&self, aabb_min: Vec3, aabb_max: Vec3) -> bool {
        for &[nx, ny, nz, d] in &self.planes {
            let px = if nx >= 0.0 { aabb_max.x } else { aabb_min.x };
            let py = if ny >= 0.0 { aabb_max.y } else { aabb_min.y };
            let pz = if nz >= 0.0 { aabb_max.z } else { aabb_min.z };
            if nx * px + ny * py + nz * pz + d < 0.0 {
                return false;
            }
        }
        true
    }

    #[inline]
    pub fn intersects_sphere(&self, center: Vec3, radius: f32) -> bool {
        for &[nx, ny, nz, d] in &self.planes {
            if nx * center.x + ny * center.y + nz * center.z + d < -radius {
                return false;
            }
        }
        true
    }
}

#[derive(Debug, Clone)]
pub struct PropInstance {
    pub model: ModelId,
    pub transform: Mat4,
    pub tint: [f32; 4],
    pub object_id: u32,
    pub outline_weight: f32,
    pub crease_sensitivity: f32,
    pub cast_shadow: bool,
    pub sector_id: u32,
}

/// A leaf chunk (sector) of the 3D `WorldOctree`.
#[derive(Debug, Clone)]
pub struct WorldSector {
    pub sector_id: u32,
    pub aabb_min: Vec3,
    pub aabb_max: Vec3,
    pub center: Vec3,
    pub bounding_radius: f32,
    pub road_index_offset: u32,
    pub road_index_count: u32,
    pub terrain_index_offset: u32,
    pub terrain_index_count: u32,
    pub wire_index_offset: u32,
    pub wire_index_count: u32,
    pub prop_indices: Vec<usize>,
}

#[derive(Debug, Clone)]
pub struct OctreeNode {
    pub aabb_min: Vec3,
    pub aabb_max: Vec3,
    pub children: [Option<usize>; 8],
    pub leaf_sector_id: Option<u32>,
}

#[derive(Debug, Clone)]
pub struct WorldOctree {
    pub nodes: Vec<OctreeNode>,
    pub root: usize,
}

impl WorldOctree {
    /// Traverses the 3D octree top-down on the CPU using 6-plane AABB frustum culling
    /// and registers every visible leaf sector (`WorldSector`) into `out_visible_sectors`.
    pub fn query_frustum(&self, frustum: &Frustum, out_visible_sectors: &mut Vec<usize>) {
        if self.nodes.is_empty() {
            return;
        }
        self.query_node(self.root, frustum, out_visible_sectors);
    }

    fn query_node(&self, node_idx: usize, frustum: &Frustum, out: &mut Vec<usize>) {
        let node = &self.nodes[node_idx];
        if !frustum.intersects_aabb(node.aabb_min, node.aabb_max) {
            return;
        }
        if let Some(sec_id) = node.leaf_sector_id {
            out.push(sec_id as usize);
            return;
        }
        for &child in &node.children {
            if let Some(c_idx) = child {
                self.query_node(c_idx, frustum, out);
            }
        }
    }

    /// Finds the leaf `sector_id` containing 3D point `pos`.
    pub fn find_leaf_sector(&self, pos: Vec3) -> u32 {
        if self.nodes.is_empty() {
            return 0;
        }
        let mut cur = self.root;
        loop {
            let node = &self.nodes[cur];
            if let Some(sec_id) = node.leaf_sector_id {
                return sec_id;
            }
            let mid = 0.5 * (node.aabb_min + node.aabb_max);
            let oct = ((pos.x >= mid.x) as usize)
                | (((pos.y >= mid.y) as usize) << 1)
                | (((pos.z >= mid.z) as usize) << 2);
            if let Some(c_idx) = node.children[oct] {
                cur = c_idx;
            } else {
                // Fallback to first available non-empty child
                if let Some(c_idx) = node.children.iter().flatten().copied().next() {
                    cur = c_idx;
                } else {
                    return 0;
                }
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct StaticWorldGeometry {
    pub road_vertices: Vec<SceneVertex>,
    pub road_indices: Vec<u32>,
    pub terrain_vertices: Vec<SceneVertex>,
    pub terrain_indices: Vec<u32>,
    pub wire_vertices: Vec<SceneVertex>,
    pub wire_indices: Vec<u32>,
    pub water_vertices: Vec<SceneVertex>,
    pub water_indices: Vec<u32>,
    pub props: Vec<PropInstance>,
    pub sectors: Vec<WorldSector>,
    pub octree: WorldOctree,
    pub hero_sun_dir: Vec3,
}

fn build_recursive_octree_node(
    nodes: &mut Vec<OctreeNode>,
    aabb_min: Vec3,
    aabb_max: Vec3,
    depth: usize,
    sample_points: &[Vec3],
    pt_indices: &[usize],
    next_sector_id: &mut u32,
    sector_aabbs: &mut Vec<(Vec3, Vec3)>,
) -> usize {
    let node_idx = nodes.len();
    nodes.push(OctreeNode {
        aabb_min,
        aabb_max,
        children: [None; 8],
        leaf_sector_id: None,
    });

    if depth >= OCTREE_MAX_DEPTH
        || (depth >= 3 && pt_indices.len() < 22)
        || (*next_sector_id as usize) + 32 > MAX_OCTREE_SECTORS
    {
        let sid = *next_sector_id;
        *next_sector_id += 1;
        nodes[node_idx].leaf_sector_id = Some(sid);
        sector_aabbs.push((aabb_min, aabb_max));
        return node_idx;
    }

    let mid = 0.5 * (aabb_min + aabb_max);
    let mut oct_pts: [Vec<usize>; 8] = std::array::from_fn(|_| Vec::new());
    for &pi in pt_indices {
        let p = sample_points[pi];
        let oct = ((p.x >= mid.x) as usize)
            | (((p.y >= mid.y) as usize) << 1)
            | (((p.z >= mid.z) as usize) << 2);
        oct_pts[oct].push(pi);
    }

    let non_empty_count = oct_pts.iter().filter(|v| !v.is_empty()).count();
    if non_empty_count == 0 {
        let sid = *next_sector_id;
        *next_sector_id += 1;
        nodes[node_idx].leaf_sector_id = Some(sid);
        sector_aabbs.push((aabb_min, aabb_max));
        return node_idx;
    }

    let mut children = [None; 8];
    for oct in 0..8 {
        if oct_pts[oct].is_empty() {
            continue;
        }
        let c_min = Vec3::new(
            if (oct & 1) != 0 { mid.x } else { aabb_min.x },
            if (oct & 2) != 0 { mid.y } else { aabb_min.y },
            if (oct & 4) != 0 { mid.z } else { aabb_min.z },
        );
        let c_max = Vec3::new(
            if (oct & 1) != 0 { aabb_max.x } else { mid.x },
            if (oct & 2) != 0 { aabb_max.y } else { mid.y },
            if (oct & 4) != 0 { aabb_max.z } else { mid.z },
        );
        let child_idx = build_recursive_octree_node(
            nodes,
            c_min,
            c_max,
            depth + 1,
            sample_points,
            &oct_pts[oct],
            next_sector_id,
            sector_aabbs,
        );
        children[oct] = Some(child_idx);
    }
    nodes[node_idx].children = children;
    node_idx
}

fn tighten_octree_bounds(nodes: &mut [OctreeNode], node_idx: usize, sectors: &[WorldSector]) -> (Vec3, Vec3) {
    if let Some(sid) = nodes[node_idx].leaf_sector_id {
        let sec = &sectors[sid as usize];
        nodes[node_idx].aabb_min = sec.aabb_min;
        nodes[node_idx].aabb_max = sec.aabb_max;
        return (sec.aabb_min, sec.aabb_max);
    }
    let children = nodes[node_idx].children;
    let mut b_min = Vec3::splat(f32::MAX);
    let mut b_max = Vec3::splat(f32::MIN);
    for c_idx in children.into_iter().flatten() {
        let (c_min, c_max) = tighten_octree_bounds(nodes, c_idx, sectors);
        b_min = b_min.min(c_min);
        b_max = b_max.max(c_max);
    }
    if b_min.x <= b_max.x {
        nodes[node_idx].aabb_min = b_min;
        nodes[node_idx].aabb_max = b_max;
    }
    (nodes[node_idx].aabb_min, nodes[node_idx].aabb_max)
}

fn partition_indices_by_octree(
    octree: &WorldOctree,
    num_sectors: usize,
    verts: &[SceneVertex],
    indices: &mut Vec<u32>,
    sector_tight_bounds: &mut [(Vec3, Vec3)],
) -> Vec<(u32, u32)> {
    let mut buckets: Vec<Vec<u32>> = vec![Vec::new(); num_sectors];
    for tri in indices.chunks_exact(3) {
        let p0 = Vec3::from(verts[tri[0] as usize].pos);
        let p1 = Vec3::from(verts[tri[1] as usize].pos);
        let p2 = Vec3::from(verts[tri[2] as usize].pos);
        let centroid = (p0 + p1 + p2) * (1.0 / 3.0);
        let sec = (octree.find_leaf_sector(centroid) as usize).min(num_sectors - 1);
        buckets[sec].extend_from_slice(tri);
        let b = &mut sector_tight_bounds[sec];
        b.0 = b.0.min(p0).min(p1).min(p2);
        b.1 = b.1.max(p0).max(p1).max(p2);
    }
    indices.clear();
    let mut ranges = vec![(0u32, 0u32); num_sectors];
    for (sec, b) in buckets.into_iter().enumerate() {
        let start = indices.len() as u32;
        let count = b.len() as u32;
        indices.extend(b);
        ranges[sec] = (start, count);
    }
    ranges
}

fn hash_u32(mut x: u32) -> f32 {
    x ^= x >> 16;
    x = x.wrapping_mul(0x7feb_352d);
    x ^= x >> 15;
    x = x.wrapping_mul(0x846c_a68b);
    x ^= x >> 16;
    (x & 0x00FF_FFFF) as f32 / 16_777_215.0
}

pub fn terrain_offset_y(biome: BiomeId, lat: f32, dist_m: f32, road_hw: f32) -> f32 {
    let w1 = (dist_m * 0.032 + lat * 0.042).sin();
    let w2 = (dist_m * 0.075 - lat * 0.065).cos();
    let w3 = (dist_m * 0.015 + lat * 0.018).sin();

    match biome {
        BiomeId::MountainClimb => {
            if lat < -road_hw {
                let d = (-lat - road_hw).max(0.0);
                // Rises into a natural rolling boulder-capped crest, never shooting upward endlessly
                let hill_profile = (1.0 - (-d / 34.0).exp()) * 17.5 + (d * 0.045).min(3.2);
                let roll = (dist_m * 0.024).sin() * 2.2 * (d / 28.0).min(1.0);
                hill_profile + roll + (w1 * 1.4 + w2 * 0.6) * (d / 14.0).min(1.0)
            } else if lat > road_hw {
                let d = (lat - road_hw).max(0.0);
                if d < 18.0 {
                    // Sheer rocky canyon cliff drop right past fence
                    let t = d / 18.0;
                    let smooth_t = t * t * (3.0 - 2.0 * t);
                    let drop = -23.5 * smooth_t;
                    drop + (w1 * 1.1 + w2 * 0.5) * (t * (1.0 - t) * 4.0)
                } else if d < 62.0 {
                    // Wide deep valley floor & winding silver river bed
                    let t = (d - 18.0) / 44.0;
                    -23.5 - 1.6 * (t * std::f32::consts::PI).sin()
                } else {
                    // Multi-layered distant forested canyon ridges receding into sky haze
                    let d_far = d - 62.0;
                    let ridge_base = -23.5 + d_far.powf(0.92) * 0.52;
                    let range1 = (dist_m * 0.018 + d_far * 0.045).sin() * 21.5;
                    let range2 = (dist_m * 0.009 - d_far * 0.030).cos() * 16.0;
                    let peaks = (range1 + range2 + w1 * 6.5) * (d_far / 42.0).min(1.0);
                    (ridge_base + peaks).min(84.0)
                }
            } else {
                0.0
            }
        }
        BiomeId::CoastalPromenade => {
            if lat < -road_hw {
                let d = (-lat - road_hw).max(0.0);
                if d < 0.3 {
                    d * 0.6
                } else if d < 6.6 {
                    0.18
                } else {
                    0.18 + (d - 6.6) * 0.015
                }
            } else if lat > road_hw {
                let d = (lat - road_hw).max(0.0);
                if d < 0.3 {
                    d * 0.65
                } else if d < 6.8 {
                    0.20 // Sandstone boardwalk promenade
                } else if d < 7.30 {
                    0.72 // Low seawall parapet so ocean vista opens widely
                } else {
                    let d_beach = d - 7.30;
                    -1.2 - d_beach * 0.32
                }
            } else {
                0.0
            }
        }
        BiomeId::RollingRuralGravel => {
            let edge = (lat.abs() - road_hw).max(0.0);
            if edge <= 0.0 {
                0.0
            } else if lat < 0.0 {
                // Gentle open meadow verge on left rising gradually into rolling distant hills
                let near_blend = (edge / 32.0).min(1.0);
                let far_rise = (edge - 12.0).max(0.0).powf(0.88) * 0.11;
                let roll = ((dist_m * 0.018 - edge * 0.035).sin() * 3.8 + w1 * 1.4) * near_blend * near_blend;
                far_rise + roll
            } else {
                // Gently undulating estancia pasture on right
                let blend = (edge / 26.0).min(1.0);
                let hill = (w3 * 3.6 + w1 * 1.8 + w2 * 0.6) * blend * blend;
                let gentle_slope = (edge - 8.0).max(0.0).powf(0.88) * 0.06;
                hill + gentle_slope
            }
        }
    }
}

fn terrain_color(biome: BiomeId, lat: f32, dist_m: f32, road_hw: f32, _y_off: f32) -> [f32; 4] {
    let n = (dist_m * 0.08 + lat * 0.11).sin() * 0.5 + 0.5;
    match biome {
        BiomeId::MountainClimb => {
            if lat < -road_hw {
                let d = (-lat - road_hw).max(0.0);
                if d < 2.2 {
                    [0.54, 0.58, 0.36, 1.0]
                } else {
                    [0.50 - n * 0.06, 0.56 - n * 0.04, 0.34, 1.0]
                }
            } else if lat > road_hw {
                let d = (lat - road_hw).max(0.0);
                if d < 3.5 {
                    [0.53, 0.57, 0.36, 1.0]
                } else if d < 22.0 {
                    // Warm stratified sandstone canyon cliff wall
                    [0.60 - n * 0.08, 0.53 - n * 0.05, 0.40, 1.0]
                } else if d < 105.0 {
                    // Rich deep canyon forest olive-green contrast around silver river
                    [0.24 + n * 0.06, 0.36 + n * 0.06, 0.21 + n * 0.04, 1.0]
                } else {
                    // Distant misty green-slate ridges
                    [0.41 + n * 0.05, 0.48 + n * 0.04, 0.37 + n * 0.04, 1.0]
                }
            } else {
                [0.74, 0.62, 0.46, 1.0]
            }
        }
        BiomeId::CoastalPromenade => {
            if lat < -road_hw {
                let d = (-lat - road_hw).max(0.0);
                if d < 6.6 {
                    [0.77, 0.70, 0.61, 1.0]
                } else {
                    [0.64, 0.60, 0.52, 1.0]
                }
            } else if lat > road_hw {
                let d = (lat - road_hw).max(0.0);
                if d < 6.8 {
                    [0.80, 0.66, 0.51, 1.0]
                } else if d < 7.45 {
                    [0.75, 0.65, 0.55, 1.0]
                } else {
                    [0.78, 0.69, 0.55, 1.0]
                }
            } else {
                [0.48, 0.46, 0.45, 1.0]
            }
        }
        BiomeId::RollingRuralGravel => {
            let edge = (lat.abs() - road_hw).max(0.0);
            if edge < 2.2 {
                [0.57, 0.60, 0.37, 1.0]
            } else {
                [0.54 + n * 0.07, 0.58 + n * 0.05, 0.35 + n * 0.04, 1.0]
            }
        }
    }
}

fn push_quad(
    verts: &mut Vec<SceneVertex>,
    indices: &mut Vec<u32>,
    a: SceneVertex,
    b: SceneVertex,
    c: SceneVertex,
    d: SceneVertex,
) {
    let base = verts.len() as u32;
    verts.extend_from_slice(&[a, b, c, d]);
    indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
}

fn compute_grid_normals(verts: &mut [SceneVertex], indices: &[u32]) {
    let mut accum = vec![Vec3::ZERO; verts.len()];
    for tri in indices.chunks_exact(3) {
        let i0 = tri[0] as usize;
        let i1 = tri[1] as usize;
        let i2 = tri[2] as usize;
        let p0 = Vec3::from(verts[i0].pos);
        let p1 = Vec3::from(verts[i1].pos);
        let p2 = Vec3::from(verts[i2].pos);
        let n = (p1 - p0).cross(p2 - p0);
        accum[i0] += n;
        accum[i1] += n;
        accum[i2] += n;
    }
    for (v, n) in verts.iter_mut().zip(accum) {
        let nn = n.normalize_or_zero();
        let n_arr = if nn.length_squared() > 0.01 {
            nn.into()
        } else {
            [0.0, 1.0, 0.0]
        };
        v.set_normal(n_arr);
    }
}

pub fn build_biome_world(biome: BiomeId, spline: &RouteSpline) -> StaticWorldGeometry {
    let road_hw = biome.road_half_width();
    let ring_count = (spline.samples.len() / 2).max(240);
    let step = spline.total_length_m / ring_count as f32;

    // 1. Road ribbon mesh
    let mut road_vertices = Vec::new();
    let mut road_indices = Vec::new();
    let road_mat_type = match biome {
        BiomeId::CoastalPromenade => 3,
        BiomeId::MountainClimb | BiomeId::RollingRuralGravel => 1,
    };
    let road_col = match biome {
        BiomeId::CoastalPromenade => [0.47, 0.45, 0.44, 1.0],
        BiomeId::MountainClimb => [0.77, 0.65, 0.48, 1.0],
        BiomeId::RollingRuralGravel => [0.77, 0.65, 0.47, 1.0],
    };

    let road_lats = [-road_hw, -road_hw * 0.5, 0.0, road_hw * 0.5, road_hw];
    let n_rl = road_lats.len();
    for r in 0..=ring_count {
        let d = (r % ring_count) as f32 * step;
        let s = spline.sample(d);
        let d_uv = r as f32 * step;
        for (li, &lat) in road_lats.iter().enumerate() {
            let crown = if biome == BiomeId::CoastalPromenade {
                0.02
            } else {
                (1.0 - (lat / road_hw).powi(2)) * 0.05
            };
            let p = s.pos + s.right * lat + Vec3::Y * crown;
            let u = li as f32 / (n_rl - 1) as f32;
            road_vertices.push(SceneVertex::new(
                p.into(),
                s.up.into(),
                [u, d_uv],
                road_col,
                road_mat_type,
            ));
        }
    }
    for r in 0..ring_count {
        for li in 0..(n_rl - 1) {
            let i0 = (r * n_rl + li) as u32;
            let i1 = (r * n_rl + li + 1) as u32;
            let i2 = ((r + 1) * n_rl + li + 1) as u32;
            let i3 = ((r + 1) * n_rl + li) as u32;
            road_indices.extend_from_slice(&[i0, i1, i2, i0, i2, i3]);
        }
    }
    compute_grid_normals(&mut road_vertices, &road_indices);

    // 2. Smooth high-resolution terrain skirts
    let mut terrain_vertices = Vec::new();
    let mut terrain_indices = Vec::new();

    let (left_offsets, right_offsets): (Vec<f32>, Vec<f32>) = match biome {
        BiomeId::MountainClimb => (
            vec![
                -road_hw,
                -road_hw - 1.5,
                -road_hw - 4.0,
                -road_hw - 8.5,
                -road_hw - 15.0,
                -road_hw - 24.0,
                -road_hw - 36.0,
                -road_hw - 52.0,
                -road_hw - 72.0,
                -road_hw - 96.0,
            ],
            vec![
                road_hw,
                road_hw + 1.2,
                road_hw + 4.5,
                road_hw + 10.0,
                road_hw + 18.0,
                road_hw + 27.0,
                road_hw + 38.0,
                road_hw + 49.0,
                road_hw + 62.0,
                road_hw + 82.0,
                road_hw + 110.0,
                road_hw + 148.0,
                road_hw + 195.0,
                road_hw + 255.0,
                road_hw + 325.0,
            ],
        ),
        BiomeId::CoastalPromenade => (
            // Keep left skirt compact (<=42m) so city terrain never crosses concave bay return leg!
            vec![
                -road_hw,
                -road_hw - 0.3,
                -road_hw - 6.6,
                -road_hw - 18.0,
                -road_hw - 38.0,
            ],
            vec![
                road_hw,
                road_hw + 0.3,
                road_hw + 6.75,
                road_hw + 6.90,
                road_hw + 7.25,
                road_hw + 7.45,
                road_hw + 16.0,
                road_hw + 28.0,
            ],
        ),
        BiomeId::RollingRuralGravel => (
            vec![
                -road_hw,
                -road_hw - 2.0,
                -road_hw - 5.5,
                -road_hw - 11.0,
                -road_hw - 18.0,
                -road_hw - 28.0,
                -road_hw - 42.0,
                -road_hw - 64.0,
                -road_hw - 92.0,
            ],
            vec![
                road_hw,
                road_hw + 2.0,
                road_hw + 5.5,
                road_hw + 11.0,
                road_hw + 18.0,
                road_hw + 28.0,
                road_hw + 42.0,
                road_hw + 64.0,
                road_hw + 92.0,
            ],
        ),
    };

    for side_lats in [&left_offsets, &right_offsets] {
        let base_vert = terrain_vertices.len();
        let cols = side_lats.len();
        let is_left = side_lats[1] < 0.0;
        for r in 0..=ring_count {
            let d = (r % ring_count) as f32 * step;
            let s = spline.sample(d);
            let d_uv = r as f32 * step;
            for &lat in side_lats {
                let y_off = terrain_offset_y(biome, lat, d, road_hw);
                let mut p = s.pos + s.right * lat;
                p.y = s.pos.y + y_off;
                if biome == BiomeId::CoastalPromenade && lat > road_hw + 7.4 {
                    p.y = (s.pos.y + y_off).min(0.4);
                }
                let col = terrain_color(biome, lat, d, road_hw, y_off);
                let mat_type = if biome == BiomeId::CoastalPromenade
                    && ((lat > road_hw + 0.15 && lat < road_hw + 6.8)
                        || (lat < -road_hw - 0.15 && lat > -road_hw - 6.6))
                {
                    6
                } else {
                    0
                };
                terrain_vertices.push(SceneVertex::new(
                    p.into(),
                    [0.0, 1.0, 0.0],
                    [lat, d_uv],
                    col,
                    mat_type,
                ));
            }
        }
        for r in 0..ring_count {
            for c in 0..(cols - 1) {
                let i0 = (base_vert + r * cols + c) as u32;
                let i1 = (base_vert + r * cols + c + 1) as u32;
                let i2 = (base_vert + (r + 1) * cols + c + 1) as u32;
                let i3 = (base_vert + (r + 1) * cols + c) as u32;
                if is_left {
                    terrain_indices.extend_from_slice(&[i0, i2, i1, i0, i3, i2]);
                } else {
                    terrain_indices.extend_from_slice(&[i0, i1, i2, i0, i2, i3]);
                }
            }
        }
    }

    compute_grid_normals(&mut terrain_vertices, &terrain_indices);

    // 3. Continuous 3-strand fence wires
    let mut wire_vertices = Vec::new();
    let mut wire_indices = Vec::new();
    let wire_lanes: Vec<f32> = match biome {
        BiomeId::MountainClimb => vec![road_hw + 0.35],
        BiomeId::RollingRuralGravel => vec![road_hw + 0.45],
        BiomeId::CoastalPromenade => vec![],
    };
    for wire_lat in wire_lanes {
        for strand_h in [0.32_f32, 0.62, 0.92] {
            for r in 0..ring_count {
                let d0 = r as f32 * step;
                let d1 = ((r + 1) % ring_count) as f32 * step;
                let s0 = spline.sample(d0);
                let s1 = spline.sample(d1);
                let y0 = terrain_offset_y(biome, wire_lat, d0, road_hw);
                let y1 = terrain_offset_y(biome, wire_lat, d1, road_hw);
                let p0 = s0.pos + s0.right * wire_lat + Vec3::Y * (y0 + strand_h);
                let p1 = s1.pos + s1.right * wire_lat + Vec3::Y * (y1 + strand_h);
                let up_off = Vec3::Y * 0.015;
                let side_off = s0.right * 0.014;
                let col = [0.22, 0.19, 0.16, 1.0];
                push_quad(
                    &mut wire_vertices,
                    &mut wire_indices,
                    SceneVertex::new((p0 - up_off).into(), s0.right.into(), [0.0, 0.0], col, 0),
                    SceneVertex::new((p0 + up_off).into(), s0.right.into(), [1.0, 0.0], col, 0),
                    SceneVertex::new((p1 + up_off).into(), s1.right.into(), [1.0, 1.0], col, 0),
                    SceneVertex::new((p1 - up_off).into(), s1.right.into(), [0.0, 1.0], col, 0),
                );
                push_quad(
                    &mut wire_vertices,
                    &mut wire_indices,
                    SceneVertex::new((p0 - side_off).into(), [0.0, 1.0, 0.0], [0.0, 0.0], col, 0),
                    SceneVertex::new((p0 + side_off).into(), [0.0, 1.0, 0.0], [1.0, 0.0], col, 0),
                    SceneVertex::new((p1 + side_off).into(), [0.0, 1.0, 0.0], [1.0, 1.0], col, 0),
                    SceneVertex::new((p1 - side_off).into(), [0.0, 1.0, 0.0], [0.0, 1.0], col, 0),
                );
            }
        }
    }

    // 4. Water meshes
    let mut water_vertices = Vec::new();
    let mut water_indices = Vec::new();
    match biome {
        BiomeId::MountainClimb => {
            // Sweeping gleaming silver river through the open valley gorge (lat ~ 35m, y = -20.2m)
            for r in 0..ring_count {
                let d0 = r as f32 * step;
                let d1 = ((r + 1) % ring_count) as f32 * step;
                let s0 = spline.sample(d0);
                let s1 = spline.sample(d1);
                let lat0 = 44.5 + (d0 * 0.028).sin() * 7.5;
                let lat1 = 44.5 + (d1 * 0.028).sin() * 7.5;
                let w = 5.8;
                let y0 = s0.pos.y - 22.9;
                let y1 = s1.pos.y - 22.9;
                let p0l = Vec3::new(s0.pos.x + s0.right.x * (lat0 - w), y0, s0.pos.z + s0.right.z * (lat0 - w));
                let p0r = Vec3::new(s0.pos.x + s0.right.x * (lat0 + w), y0, s0.pos.z + s0.right.z * (lat0 + w));
                let p1l = Vec3::new(s1.pos.x + s1.right.x * (lat1 - w), y1, s1.pos.z + s1.right.z * (lat1 - w));
                let p1r = Vec3::new(s1.pos.x + s1.right.x * (lat1 + w), y1, s1.pos.z + s1.right.z * (lat1 + w));
                let col = [0.90, 0.93, 0.91, 1.0];
                push_quad(
                    &mut water_vertices,
                    &mut water_indices,
                    SceneVertex::new(p0l.into(), [0.0, 1.0, 0.0], [0.0, d0], col, 7),
                    SceneVertex::new(p0r.into(), [0.0, 1.0, 0.0], [1.0, d0], col, 7),
                    SceneVertex::new(p1r.into(), [0.0, 1.0, 0.0], [1.0, d1], col, 7),
                    SceneVertex::new(p1l.into(), [0.0, 1.0, 0.0], [0.0, d1], col, 7),
                );
            }
        }
        BiomeId::CoastalPromenade => {
            let hero = spline.sample(spline.hero_start_dist_m);
            let ocean_y = 0.25_f32;
            let cx = hero.pos.x + hero.right.x * 520.0;
            let cz = hero.pos.z + hero.right.z * 520.0;
            let half = 1800.0_f32;
            let col = [0.39, 0.51, 0.61, 1.0];
            push_quad(
                &mut water_vertices,
                &mut water_indices,
                SceneVertex::new([cx - half, ocean_y, cz + half], [0.0, 1.0, 0.0], [0.0, 0.0], col, 7),
                SceneVertex::new([cx + half, ocean_y, cz + half], [0.0, 1.0, 0.0], [1.0, 0.0], col, 7),
                SceneVertex::new([cx + half, ocean_y, cz - half], [0.0, 1.0, 0.0], [1.0, 1.0], col, 7),
                SceneVertex::new([cx - half, ocean_y, cz - half], [0.0, 1.0, 0.0], [0.0, 1.0], col, 7),
            );
        }
        BiomeId::RollingRuralGravel => {}
    }

    // 5. Biome Prop Placement
    let mut props = Vec::new();
    let mut obj_id_counter = 20_u32;
    let mut push_prop = |model: ModelId, pos: Vec3, yaw: f32, scale: Vec3, tint: [f32; 4], outline_w: f32, crease: f32, shadow: bool| {
        obj_id_counter = ((obj_id_counter - 20 + 1) % 210) + 20;
        let xform = Mat4::from_translation(pos)
            * Mat4::from_rotation_y(yaw)
            * Mat4::from_scale(scale);
        props.push(PropInstance {
            model,
            transform: xform,
            tint,
            object_id: obj_id_counter,
            outline_weight: outline_w,
            crease_sensitivity: crease,
            cast_shadow: shadow,
            sector_id: 0,
        });
    };

    match biome {
        BiomeId::MountainClimb => {
            let total = spline.total_length_m;
            let mut d = 0.0_f32;
            let mut idx = 0_u32;
            while d < total {
                let s = spline.sample(d);
                let yaw = (-s.tangent.x).atan2(-s.tangent.z);
                let f_lat = road_hw + 0.35;
                let fy = terrain_offset_y(biome, f_lat, d, road_hw);
                let tilt = (hash_u32(idx * 17) - 0.5) * 0.07;
                push_prop(
                    ModelId::FencePost,
                    s.pos + s.right * f_lat + Vec3::Y * fy,
                    yaw + tilt,
                    Vec3::splat(1.0 + (hash_u32(idx * 31) - 0.5) * 0.14),
                    [1.0; 4],
                    0.95,
                    0.9,
                    true,
                );

                // Lush red & orange wildflower tufts along foreground/roadside verges
                for &side in &[-1.0_f32, 1.0] {
                    let r0 = hash_u32(idx * 43 + if side < 0.0 { 1 } else { 2 });
                    if r0 > 0.18 {
                        let lat = side * (road_hw + 0.45 + r0 * 1.15);
                        let y_off = terrain_offset_y(biome, lat, d, road_hw);
                        push_prop(
                            ModelId::WildflowerTuft,
                            s.pos + s.right * lat + Vec3::Y * (y_off - 0.06),
                            r0 * 6.28,
                            Vec3::splat(0.95 + r0 * 0.55),
                            [1.0; 4],
                            0.55,
                            0.3,
                            false,
                        );
                    }
                }

                // Left mountain hillside: stratified angular boulders, cliff crags, scalloped bushes & hilltop rocks
                if idx % 2 == 0 {
                    let r1 = hash_u32(idx * 71 + 5);
                    let lat = -(road_hw + 2.4 + r1 * 12.5);
                    let y_off = terrain_offset_y(biome, lat, d, road_hw);
                    let model = if r1 > 0.72 {
                        ModelId::CliffOutcrop
                    } else if r1 > 0.35 {
                        ModelId::BoulderLarge
                    } else {
                        ModelId::BoulderMedium
                    };
                    let sc = if model == ModelId::CliffOutcrop {
                        0.68 + r1 * 0.38
                    } else {
                        0.90 + r1 * 0.72
                    };
                    push_prop(
                        model,
                        s.pos + s.right * lat + Vec3::Y * (y_off - 0.35),
                        r1 * 6.28,
                        Vec3::splat(sc),
                        [1.0; 4],
                        0.85,
                        0.42,
                        true,
                    );
                }
                if idx % 6 == 2 {
                    let r1b = hash_u32(idx * 83 + 13);
                    let lat = -(road_hw + 18.0 + r1b * 18.0);
                    let y_off = terrain_offset_y(biome, lat, d, road_hw);
                    let model = if r1b > 0.42 { ModelId::CliffOutcrop } else { ModelId::BoulderLarge };
                    push_prop(
                        model,
                        s.pos + s.right * lat + Vec3::Y * (y_off - 0.65),
                        r1b * 6.28,
                        Vec3::splat(0.75 + r1b * 0.45),
                        [1.0; 4],
                        0.75,
                        0.38,
                        true,
                    );
                }
                if idx % 2 == 1 {
                    let r2 = hash_u32(idx * 89 + 9);
                    let lat = -(road_hw + 1.8 + r2 * 16.0);
                    let y_off = terrain_offset_y(biome, lat, d, road_hw);
                    push_prop(
                        ModelId::BushCluster,
                        s.pos + s.right * lat + Vec3::Y * (y_off - 0.28),
                        r2 * 6.28,
                        Vec3::splat(0.95 + r2 * 0.85),
                        [0.88, 0.92, 0.82, 1.0],
                        0.82,
                        0.38,
                        true,
                    );
                }

                // Right cliff outcrops & dense canyon forest slopes flanking the winding river
                if idx % 2 == 0 {
                    let r3 = hash_u32(idx * 107 + 3);
                    let lat = road_hw + 4.8 + r3 * 11.5;
                    let y_off = terrain_offset_y(biome, lat, d, road_hw);
                    push_prop(
                        ModelId::CliffOutcrop,
                        s.pos + s.right * lat + Vec3::Y * (y_off - 2.2),
                        r3 * 6.28,
                        Vec3::new(0.90 + r3 * 0.6, 0.80 + r3 * 0.55, 0.90 + r3 * 0.6),
                        [1.0; 4],
                        0.82,
                        0.42,
                        false,
                    );
                }
                if idx % 2 == 1 {
                    let r4 = hash_u32(idx * 131 + 11);
                    // Put lush valley forest trees on BOTH near riverbank (lat ~20..26) and far canyon slope (lat ~58..112)
                    let (lat, model) = if r4 > 0.72 {
                        (road_hw + 19.0 + (r4 - 0.72) * 20.0, ModelId::TreeCanopy)
                    } else {
                        (road_hw + 58.0 + r4 * 62.0, ModelId::TreeCanopy)
                    };
                    let y_off = terrain_offset_y(biome, lat, d, road_hw);
                    push_prop(
                        model,
                        s.pos + s.right * lat + Vec3::Y * (y_off - 0.55),
                        r4 * 6.28,
                        Vec3::splat(1.55 + r4 * 1.25),
                        [0.58, 0.70, 0.52, 1.0],
                        0.45,
                        0.26,
                        false,
                    );
                }
                {
                    let r5 = hash_u32(idx * 149 + 19);
                    let lat = road_hw + 62.0 + r5 * 58.0;
                    let y_off = terrain_offset_y(biome, lat, d, road_hw);
                    push_prop(
                        if r5 > 0.35 { ModelId::TreeCanopy } else { ModelId::TreePine },
                        s.pos + s.right * lat + Vec3::Y * (y_off - 0.6),
                        r5 * 6.28,
                        Vec3::splat(1.65 + r5 * 1.35),
                        [0.54, 0.66, 0.50, 1.0],
                        0.38,
                        0.22,
                        false,
                    );
                }

                d += 3.5;
                idx += 1;
            }

            // Soaring mountain condors over canyon
            let hero = spline.sample(spline.hero_start_dist_m);
            for (fwd, right, up, yaw, sc) in [
                (82.0, 26.0, 21.0, 0.7, 1.25),
                (108.0, 40.0, 27.0, 1.2, 1.0),
                (132.0, 32.0, 32.0, 0.3, 0.85),
                (90.0, 46.0, 17.5, 1.8, 1.1),
            ] {
                push_prop(
                    ModelId::CondorBird,
                    hero.pos + hero.tangent * fwd + hero.right * right + Vec3::Y * up,
                    yaw,
                    Vec3::splat(sc),
                    [1.0; 4],
                    0.85,
                    0.5,
                    false,
                );
            }
        }
        BiomeId::CoastalPromenade => {
            let total = spline.total_length_m;
            let mut d = 0.0_f32;
            let mut idx = 0_u32;
            while d < total {
                let s = spline.sample(d);
                let yaw = (-s.tangent.x).atan2(-s.tangent.z);

                // Palms & leafy street trees on left sidewalk
                if idx % 2 == 0 {
                    let lat_tree = -(road_hw + 2.5);
                    let model = if idx % 6 == 0 { ModelId::TreeCanopy } else { ModelId::PalmTree };
                    let sc = if model == ModelId::PalmTree { 1.15 + hash_u32(idx * 19) * 0.25 } else { 1.0 };
                    push_prop(
                        model,
                        s.pos + s.right * lat_tree + Vec3::Y * 0.14,
                        yaw + hash_u32(idx * 23) * 1.8,
                        Vec3::splat(sc),
                        [1.0; 4],
                        0.85,
                        0.6,
                        true,
                    );
                }

                // Buildings along left promenade avenue: colonial townhouses near hero, stepped coastal mid/tall blocks curving away
                if idx % 2 == 0 {
                    let r = hash_u32(idx * 41);
                    let dist_from_hero = (d - spline.hero_start_dist_m).rem_euclid(total);
                    let (model, lat_b) = if (0.0..52.0).contains(&dist_from_hero) {
                        (ModelId::ColonialTownhouse, -(road_hw + 11.4))
                    } else if r > 0.40 {
                        (ModelId::CoastalBuildingTall, -(road_hw + 13.6))
                    } else {
                        (ModelId::CoastalBuildingMid, -(road_hw + 12.6))
                    };
                    let tint = match idx % 6 {
                        0 => [1.0, 0.98, 0.94, 1.0],
                        2 => [0.98, 0.92, 0.88, 1.0],
                        _ => [0.95, 0.94, 0.92, 1.0],
                    };
                    push_prop(
                        model,
                        s.pos + s.right * lat_b + Vec3::Y * 0.14,
                        yaw,
                        Vec3::ONE,
                        tint,
                        0.90,
                        1.0,
                        true,
                    );
                }

                // Right boardwalk: vintage streetlamps, benches, pedestrians
                if idx % 2 == 1 {
                    push_prop(
                        ModelId::Streetlamp,
                        s.pos + s.right * (road_hw + 1.45) + Vec3::Y * 0.18,
                        yaw,
                        Vec3::ONE,
                        [1.0; 4],
                        0.90,
                        0.8,
                        true,
                    );
                }
                if idx % 3 == 0 {
                    push_prop(
                        ModelId::PromenadeBench,
                        s.pos + s.right * (road_hw + 5.2) + Vec3::Y * 0.18,
                        yaw + std::f32::consts::FRAC_PI_2,
                        Vec3::ONE,
                        [1.0; 4],
                        0.85,
                        0.9,
                        true,
                    );
                }
                if idx % 3 == 1 {
                    let r = hash_u32(idx * 67);
                    let lat_ped = road_hw + 2.4 + r * 2.4;
                    push_prop(
                        ModelId::Pedestrian,
                        s.pos + s.right * lat_ped + Vec3::Y * 0.18,
                        yaw + if r > 0.5 { 0.0 } else { std::f32::consts::PI },
                        Vec3::ONE,
                        [1.0; 4],
                        0.95,
                        0.6,
                        true,
                    );
                }

                d += 9.5;
                idx += 1;
            }

            // Coastal flagpole in left foreground curb in front of white colonial townhouses
            let flag_s = spline.sample(spline.hero_start_dist_m + 31.5);
            let flag_yaw = (-flag_s.tangent.x).atan2(-flag_s.tangent.z) + 0.35;
            push_prop(
                ModelId::Flagpole,
                flag_s.pos - flag_s.right * (road_hw + 3.1) + Vec3::Y * 0.14,
                flag_yaw,
                Vec3::splat(0.84),
                [1.0; 4],
                0.85,
                0.7,
                true,
            );

            // Sailboats on sunset ocean
            let hero = spline.sample(spline.hero_start_dist_m);
            for (fwd, right, yaw, sc) in [
                (165.0, 105.0, 0.9, 1.3),
                (225.0, 155.0, 0.6, 1.5),
                (285.0, 210.0, 1.2, 1.8),
                (140.0, 190.0, 0.4, 1.4),
            ] {
                push_prop(
                    ModelId::Sailboat,
                    Vec3::new(
                        hero.pos.x + hero.tangent.x * fwd + hero.right.x * right,
                        0.25,
                        hero.pos.z + hero.tangent.z * fwd + hero.right.z * right,
                    ),
                    yaw,
                    Vec3::splat(sc),
                    [1.0; 4],
                    0.75,
                    0.7,
                    false,
                );
            }
        }
        BiomeId::RollingRuralGravel => {
            let total = spline.total_length_m;
            let hero_d = spline.hero_start_dist_m;

            let pick_rock = |h: u32| -> ModelId {
                let v = ((h / 3) % 10) as u8;
                match h % 3 {
                    0 => ModelId::CampoRockGranite(v),
                    1 => ModelId::CampoRockSlate(v),
                    _ => ModelId::CampoRockCluster(v),
                }
            };
            let pick_tree = |h: u32| -> ModelId {
                let v = ((h / 3) % 10) as u8;
                match h % 3 {
                    0 => ModelId::TreeEucalyptus(v),
                    1 => ModelId::TreeOmbu(v),
                    _ => ModelId::TreeCeibo(v),
                }
            };

            // 1. Right-side wooden post fence (strictly fence posts only; vegetation is distributed in irregular 2D patches below!)
            let mut d = 0.0_f32;
            let mut idx = 0_u32;
            while d < total {
                let s = spline.sample(d);
                let yaw = (-s.tangent.x).atan2(-s.tangent.z);

                // Right-side wooden fence post (matches continuous 3-strand wire at +road_hw + 0.45)
                let f_lat = road_hw + 0.45;
                let fy = terrain_offset_y(biome, f_lat, d, road_hw);
                let post_tilt = (hash_u32(idx * 19 + 3) - 0.5) * 0.06;
                push_prop(
                    ModelId::FencePost,
                    s.pos + s.right * f_lat + Vec3::Y * (fy - 0.05),
                    yaw + post_tilt,
                    Vec3::splat(0.96 + hash_u32(idx * 29) * 0.10),
                    [1.0; 4],
                    0.95,
                    0.9,
                    true,
                );

                d += 4.0;
                idx += 1;
            }

            // 1b. Uruguayan Campo Bush Lumps & Pasture Grass Patches
            // Matches aerial photography of Uruguayan Campo: wide open rolling green pastures interrupted by
            // tight, cohesive "lumps" of bushes (often with no trees anywhere nearby) and occasional grass patches!
            let mut lump_d = 14.0_f32;
            let mut p_seed = 300_u32;
            while lump_d < total {
                p_seed = p_seed.wrapping_add(1);
                let h0 = hash_u32(p_seed * 181 + 13);
                let h1 = hash_u32(p_seed * 239 + 37);
                let h2 = hash_u32(p_seed * 307 + 73);
                let h3 = hash_u32(p_seed * 389 + 109);

                let rel_hero = (lump_d - hero_d).rem_euclid(total);
                let in_hero_foreground = (0.0..68.0).contains(&rel_hero);

                let side = if h0 > 0.47 { -1.0_f32 } else { 1.0_f32 };
                let depth_m = if in_hero_foreground {
                    18.0 + h1 * 32.0
                } else {
                    2.2 + (h1 * h1) * 44.0
                };
                let center_lat = side * (road_hw + depth_m);

                if p_seed % 3 != 2 {
                    // Tight Bush Lump (4..7 bushes clustered closely within a 2.0..5.2m radius, plus grass & occasional rock)
                    let bush_count = 4 + ((h2 * 3.8) as u32);
                    let lump_radius = 2.2 + h3 * 3.0;
                    for k in 0..bush_count {
                        let ks = p_seed * 43 + k * 19;
                        let ang = (k as f32 / bush_count as f32) * std::f32::consts::TAU
                            + (hash_u32(ks + 1) - 0.5) * 0.9;
                        let r_dist = hash_u32(ks + 2).sqrt() * lump_radius;
                        let cd = (lump_d + ang.cos() * r_dist).rem_euclid(total);
                        let raw_lat = center_lat + ang.sin() * r_dist;
                        let clat = if side < 0.0 {
                            raw_lat.min(-(road_hw + 1.55))
                        } else {
                            raw_lat.max(road_hw + 1.85)
                        };
                        let cs = spline.sample(cd);
                        let cy = terrain_offset_y(biome, clat, cd, road_hw);
                        let v_idx = ((ks / 3) % 10) as u8;
                        let kind_r = hash_u32(ks + 5);
                        let (b_model, sc) = if kind_r < 0.62 {
                            (ModelId::ChircaBush(v_idx), 0.72 + hash_u32(ks + 3) * 0.46)
                        } else if kind_r < 0.84 && clat.abs() < road_hw + 22.0 {
                            (ModelId::PampasCortadera(v_idx), 0.76 + hash_u32(ks + 3) * 0.38)
                        } else {
                            (ModelId::CeiboBush(v_idx), 0.74 + hash_u32(ks + 3) * 0.40)
                        };
                        push_prop(
                            b_model,
                            cs.pos + cs.right * clat + Vec3::Y * (cy - 0.06),
                            hash_u32(ks + 4) * 6.28,
                            Vec3::splat(sc),
                            [1.0; 4],
                            0.84,
                            0.0,
                            true,
                        );
                    }
                    // Anchor the bush lump with 2..4 CampoGrass tufts around its perimeter
                    for g_k in 0..3_u32 {
                        let gs = p_seed * 71 + g_k * 13;
                        let gang = hash_u32(gs + 1) * std::f32::consts::TAU;
                        let gr = lump_radius * (0.65 + hash_u32(gs + 2) * 0.55);
                        let gd = (lump_d + gang.cos() * gr).rem_euclid(total);
                        let glat = if side < 0.0 {
                            (center_lat + gang.sin() * gr).min(-(road_hw + 0.95))
                        } else {
                            (center_lat + gang.sin() * gr).max(road_hw + 1.15)
                        };
                        let gs_s = spline.sample(gd);
                        let gy = terrain_offset_y(biome, glat, gd, road_hw);
                        push_prop(
                            ModelId::CampoGrass(((gs / 3) % 10) as u8),
                            gs_s.pos + gs_s.right * glat + Vec3::Y * (gy - 0.03),
                            hash_u32(gs + 3) * 6.28,
                            Vec3::splat(0.85 + hash_u32(gs + 4) * 0.50),
                            [1.0; 4],
                            0.58,
                            0.0,
                            false,
                        );
                    }
                } else {
                    // Pure Pasture Grass & Wildflower Patch in the open grassland (4..8 grass clumps)
                    let grass_base_lat = side * (road_hw + 0.85 + (h1 * h1) * 18.0);
                    let count = 4 + ((h2 * 4.5) as u32);
                    for k in 0..count {
                        let ks = p_seed * 67 + k * 17;
                        let fd = (lump_d + (hash_u32(ks + 1) - 0.5) * 4.8).rem_euclid(total);
                        let flat = if side < 0.0 {
                            (grass_base_lat + (hash_u32(ks + 2) - 0.5) * 3.8).min(-(road_hw + 0.55))
                        } else {
                            (grass_base_lat + (hash_u32(ks + 2) - 0.5) * 3.8).max(road_hw + 0.65)
                        };
                        let fs = spline.sample(fd);
                        let fy = terrain_offset_y(biome, flat, fd, road_hw);
                        let is_flower = k == 0 && h3 > 0.45;
                        let g_model = if is_flower {
                            ModelId::WildflowerTuft
                        } else {
                            ModelId::CampoGrass(((ks / 3) % 10) as u8)
                        };
                        let fsc = 0.80 + hash_u32(ks + 3) * 0.55;
                        push_prop(
                            g_model,
                            fs.pos + fs.right * flat + Vec3::Y * (fy - 0.03),
                            hash_u32(ks + 4) * 6.28,
                            Vec3::splat(fsc),
                            [1.0; 4],
                            0.55,
                            0.0,
                            false,
                        );
                    }
                }

                // Wide open pasture intervals (24m to 62m) between bush lumps & grass patches
                lump_d += 24.0 + h0 * 38.0;
            }

            // 2. Non-repetitive geological rock outcrops & boulder lumps (sparse across open pastures)
            let mut rock_d = 18.0_f32;
            let mut r_seed = 500_u32;
            while rock_d < total {
                r_seed = r_seed.wrapping_add(1);
                let h0 = hash_u32(r_seed * 193 + 17);
                let h1 = hash_u32(r_seed * 251 + 41);
                let h2 = hash_u32(r_seed * 313 + 89);

                let vein_density = (rock_d * 0.037).sin() * 0.5 + (rock_d * 0.091).cos() * 0.5;
                if vein_density > -0.05 {
                    let side = if h0 > 0.36 { -1.0_f32 } else { 1.0_f32 };
                    let center_lat = side * (road_hw + 4.2 + h1 * 38.0);
                    let cluster_count = 2 + ((h2 * 3.2) as u32);

                    for c_i in 0..cluster_count {
                        let cs = r_seed * 37 + c_i * 19;
                        let off_d = (hash_u32(cs + 1) - 0.5) * 4.5;
                        let off_lat = (hash_u32(cs + 2) - 0.5) * 4.2;
                        let cur_d = (rock_d + off_d).rem_euclid(total);
                        let cur_lat = if side < 0.0 {
                            (center_lat + off_lat).min(-(road_hw + 2.0))
                        } else {
                            (center_lat + off_lat).max(road_hw + 2.4)
                        };
                        let sc_sample = spline.sample(cur_d);
                        let y_r = terrain_offset_y(biome, cur_lat, cur_d, road_hw);
                        let rock_model = pick_rock(cs + c_i * 7);
                        let sx = 0.75 + hash_u32(cs + 3) * 0.85;
                        let sy = 0.70 + hash_u32(cs + 4) * 0.75;
                        let sz = 0.75 + hash_u32(cs + 5) * 0.85;
                        push_prop(
                            rock_model,
                            sc_sample.pos + sc_sample.right * cur_lat + Vec3::Y * (y_r - 0.18),
                            hash_u32(cs + 6) * 6.28,
                            Vec3::new(sx, sy, sz),
                            [1.0; 4],
                            0.84,
                            0.0,
                            true,
                        );
                    }
                }
                rock_d += 22.0 + h0 * h1 * 48.0;
            }

            // 3. Rare, Tight Tree Lumps ("Montes" / "Isletas") — trees are rare and clustered tightly together!
            // Leave the opening hero vista left hillsides wide open (start first loop tree lump well past hero_d)
            let mut grove_d = (hero_d + 165.0).rem_euclid(total);
            let mut g_traversed = 0.0_f32;
            let mut g_seed = 1200_u32;
            while g_traversed < total - 180.0 {
                g_seed = g_seed.wrapping_add(1);
                let h0 = hash_u32(g_seed * 167 + 29);
                let h1 = hash_u32(g_seed * 229 + 53);
                let h2 = hash_u32(g_seed * 293 + 71);

                let side = if h0 > 0.45 { 1.0_f32 } else { -1.0_f32 };
                let base_lat = side * (road_hw + 12.0 + h1 * 38.0);
                // Tight copse of 5..8 trees within a 5.5m radius + underbrush bush lump around its base
                let trees_in_lump = 5 + ((h2 * 3.8) as u32);
                let copse_radius = 4.5 + h0 * 3.2;

                for t_i in 0..trees_in_lump {
                    let ts = g_seed * 59 + t_i * 23;
                    let ang = (t_i as f32 / trees_in_lump as f32) * std::f32::consts::TAU
                        + (hash_u32(ts + 1) - 0.5) * 0.7;
                    let r_off = hash_u32(ts + 2).sqrt() * copse_radius;
                    let td = (grove_d + ang.cos() * r_off).rem_euclid(total);
                    let tlat = if side < 0.0 {
                        (base_lat + ang.sin() * r_off).min(-(road_hw + 6.0))
                    } else {
                        (base_lat + ang.sin() * r_off).max(road_hw + 6.0)
                    };
                    let s = spline.sample(td);
                    let ty = terrain_offset_y(biome, tlat, td, road_hw);
                    let tree_model = if hash_u32(ts + 3) > 0.80 {
                        ModelId::SmallPalm(((ts / 5) % 10) as u8)
                    } else {
                        pick_tree(ts)
                    };
                    let sc = 0.90 + hash_u32(ts + 4) * 0.42;
                    push_prop(
                        tree_model,
                        s.pos + s.right * tlat + Vec3::Y * (ty - 0.25),
                        hash_u32(ts + 5) * 6.28,
                        Vec3::splat(sc),
                        [1.0; 4],
                        0.78,
                        0.0,
                        true,
                    );
                }
                // Underbrush bush lump skirting the edge of the tree lump
                for b_i in 0..4_u32 {
                    let bs = g_seed * 83 + b_i * 19;
                    let bang = hash_u32(bs + 1) * std::f32::consts::TAU;
                    let br = copse_radius * (0.75 + hash_u32(bs + 2) * 0.45);
                    let bd = (grove_d + bang.cos() * br).rem_euclid(total);
                    let blat = base_lat + bang.sin() * br;
                    let s = spline.sample(bd);
                    let by = terrain_offset_y(biome, blat, bd, road_hw);
                    push_prop(
                        ModelId::ChircaBush(((bs / 3) % 10) as u8),
                        s.pos + s.right * blat + Vec3::Y * (by - 0.06),
                        hash_u32(bs + 3) * 6.28,
                        Vec3::splat(0.76 + hash_u32(bs + 4) * 0.35),
                        [1.0; 4],
                        0.82,
                        0.0,
                        true,
                    );
                }

                // 190m to 310m of open treeless campo between tree lumps!
                let step_grove = 190.0 + h0 * 120.0;
                grove_d = (grove_d + step_grove).rem_euclid(total);
                g_traversed += step_grove;
            }

            // 4. Estancia Ranch Houses (2 remote estancias along the 1.6 km loop, each sheltered by a tight tree lump)
            for (ranch_i, &r_dist_off) in [56.0_f32, 860.0].iter().enumerate() {
                let rd = (hero_d + r_dist_off).rem_euclid(total);
                let rs = spline.sample(rd);
                let ryaw = (-rs.tangent.x).atan2(-rs.tangent.z);
                let side = if ranch_i % 2 == 0 { 1.0_f32 } else { -1.0_f32 };
                let r_lat = side * (road_hw + 14.5);
                let ry = terrain_offset_y(biome, r_lat, rd, road_hw);
                let house_pos = rs.pos + rs.right * r_lat + Vec3::Y * (ry - 0.35);
                push_prop(
                    ModelId::RanchHouse,
                    house_pos,
                    ryaw + if side < 0.0 { std::f32::consts::FRAC_PI_2 } else { -std::f32::consts::FRAC_PI_2 },
                    Vec3::splat(0.92),
                    [1.0; 4],
                    0.88,
                    0.85,
                    true,
                );
                // Uruguayan Flagpole & spreading Ombu shade tree next to the estancia ranch house
                push_prop(
                    ModelId::Flagpole,
                    house_pos - rs.tangent * 5.5 + rs.right * (side * 3.2) + Vec3::Y * 0.20,
                    ryaw,
                    Vec3::splat(0.85),
                    [1.0; 4],
                    0.85,
                    0.5,
                    true,
                );
                push_prop(
                    ModelId::TreeOmbu((ranch_i % 10) as u8),
                    house_pos + rs.tangent * 8.5 - rs.right * (side * 1.5),
                    ryaw + 0.8,
                    Vec3::splat(1.05),
                    [1.0; 4],
                    0.82,
                    0.0,
                    true,
                );
                // Tall Eucalyptus grove sheltering the right side of the estancia
                for e_i in 0..4_u32 {
                    let ed = (rd - 14.0 + e_i as f32 * 6.5).rem_euclid(total);
                    let es = spline.sample(ed);
                    let elat = side * (road_hw + 16.5 + (e_i % 2) as f32 * 5.0);
                    let ey = terrain_offset_y(biome, elat, ed, road_hw);
                    push_prop(
                        ModelId::TreeEucalyptus(((ranch_i as u32 * 3 + e_i) % 10) as u8),
                        es.pos + es.right * elat + Vec3::Y * (ey - 0.20),
                        (e_i as f32) * 1.7,
                        Vec3::splat(1.05 + (e_i % 3) as f32 * 0.14),
                        [1.0; 4],
                        0.80,
                        0.0,
                        true,
                    );
                }
            }

            // Curated foreground bush lumps, grass patches & boulder clusters at hero_start_dist_m
            for &(d_off, lat_off, model, sc, yaw_off) in &[
                // Left foreground Bush Lump #1 (tight cluster of Ceibo, Cortadera, Chirca, rock & pasture grass tufts)
                (8.6_f32, -(road_hw + 2.35), ModelId::CeiboBush(0), 0.90_f32, 0.4_f32),
                (10.8, -(road_hw + 1.65), ModelId::PampasCortadera(1), 0.90, 1.1),
                (11.4, -(road_hw + 3.15), ModelId::ChircaBush(2), 0.85, 2.0),
                (9.8, -(road_hw + 3.45), ModelId::ChircaBush(7), 0.76, 0.8),
                (7.8, -(road_hw + 1.45), ModelId::CampoGrass(0), 1.05, 0.5),
                (9.4, -(road_hw + 1.15), ModelId::CampoGrass(3), 0.95, 1.8),
                (12.2, -(road_hw + 2.10), ModelId::CampoGrass(6), 1.00, 2.5),
                // Left mid-slope pure Bush Lump #2 (tight thicket of Chirca & rock on the open green hillside, no trees!)
                (38.0, -(road_hw + 14.5), ModelId::CampoRockGranite(4), 1.05, 1.2),
                (39.4, -(road_hw + 15.8), ModelId::ChircaBush(1), 0.88, 0.6),
                (40.8, -(road_hw + 14.2), ModelId::ChircaBush(4), 0.82, 1.9),
                (37.2, -(road_hw + 16.2), ModelId::ChircaBush(8), 0.78, 2.7),
                (38.5, -(road_hw + 13.4), ModelId::CampoGrass(2), 1.05, 1.1),
                // Right fence-side Small Palm & Chirca Lump + roadside grass patch
                (15.8, road_hw + 2.35, ModelId::SmallPalm(0), 0.92, 0.3),
                (17.4, road_hw + 3.15, ModelId::ChircaBush(5), 0.84, 2.2),
                (18.6, road_hw + 2.15, ModelId::SmallPalm(3), 0.95, 1.4),
                (14.5, road_hw + 1.35, ModelId::CampoGrass(1), 0.98, 0.9),
                (16.8, road_hw + 1.45, ModelId::CampoGrass(5), 0.92, 2.1),
            ] {
                let cd = (hero_d + d_off).rem_euclid(total);
                let cs = spline.sample(cd);
                let cyaw = (-cs.tangent.x).atan2(-cs.tangent.z) + yaw_off;
                let cy = terrain_offset_y(biome, lat_off, cd, road_hw);
                push_prop(
                    model,
                    cs.pos + cs.right * lat_off + Vec3::Y * (cy - 0.04),
                    cyaw,
                    Vec3::splat(sc),
                    [1.0; 4],
                    0.88,
                    0.0,
                    true,
                );
            }
        }
    }

    // 6. Build 3D WorldOctree and assign Road, Terrain, Wires, and Props to Octree Leaf Sectors (Chunks)
    let mut sample_points = Vec::with_capacity(
        road_indices.len() / 3 + terrain_indices.len() / 3 + wire_indices.len() / 3 + props.len(),
    );
    let mut world_min = Vec3::splat(f32::MAX);
    let mut world_max = Vec3::splat(f32::MIN);

    for (verts, idxs) in [
        (&road_vertices, &road_indices),
        (&terrain_vertices, &terrain_indices),
        (&wire_vertices, &wire_indices),
    ] {
        for tri in idxs.chunks_exact(3) {
            let p0 = Vec3::from(verts[tri[0] as usize].pos);
            let p1 = Vec3::from(verts[tri[1] as usize].pos);
            let p2 = Vec3::from(verts[tri[2] as usize].pos);
            world_min = world_min.min(p0).min(p1).min(p2);
            world_max = world_max.max(p0).max(p1).max(p2);
            sample_points.push((p0 + p1 + p2) * (1.0 / 3.0));
        }
    }
    for p in &props {
        let pos = p.transform.w_axis.truncate();
        world_min = world_min.min(pos - Vec3::splat(16.0));
        world_max = world_max.max(pos + Vec3::splat(16.0));
        sample_points.push(pos);
    }
    world_min -= Vec3::splat(2.0);
    world_max += Vec3::splat(2.0);

    let all_pt_indices: Vec<usize> = (0..sample_points.len()).collect();
    let mut octree_nodes = Vec::new();
    let mut next_sector_id = 0u32;
    let mut sector_cell_aabbs = Vec::new();
    let root = build_recursive_octree_node(
        &mut octree_nodes,
        world_min,
        world_max,
        0,
        &sample_points,
        &all_pt_indices,
        &mut next_sector_id,
        &mut sector_cell_aabbs,
    );
    let mut octree = WorldOctree {
        nodes: octree_nodes,
        root,
    };
    let num_sectors = (next_sector_id as usize).max(1);

    let mut sector_tight_bounds: Vec<(Vec3, Vec3)> =
        vec![(Vec3::splat(f32::MAX), Vec3::splat(f32::MIN)); num_sectors];

    let road_ranges = partition_indices_by_octree(
        &octree,
        num_sectors,
        &road_vertices,
        &mut road_indices,
        &mut sector_tight_bounds,
    );
    let terrain_ranges = partition_indices_by_octree(
        &octree,
        num_sectors,
        &terrain_vertices,
        &mut terrain_indices,
        &mut sector_tight_bounds,
    );
    let wire_ranges = partition_indices_by_octree(
        &octree,
        num_sectors,
        &wire_vertices,
        &mut wire_indices,
        &mut sector_tight_bounds,
    );

    let mut sector_props: Vec<Vec<usize>> = vec![Vec::new(); num_sectors];
    for (pi, p) in props.iter_mut().enumerate() {
        let pos = p.transform.w_axis.truncate();
        let sid = (octree.find_leaf_sector(pos) as usize).min(num_sectors - 1);
        p.sector_id = sid as u32;
        sector_props[sid].push(pi);
        let pad = Vec3::splat(16.0);
        sector_tight_bounds[sid].0 = sector_tight_bounds[sid].0.min(pos - pad);
        sector_tight_bounds[sid].1 = sector_tight_bounds[sid].1.max(pos + pad);
    }

    let mut sectors = Vec::with_capacity(num_sectors);
    for sec_i in 0..num_sectors {
        let (mut b_min, mut b_max) = sector_tight_bounds[sec_i];
        if b_min.x > b_max.x {
            b_min = sector_cell_aabbs[sec_i].0;
            b_max = sector_cell_aabbs[sec_i].1;
        } else {
            b_min -= Vec3::splat(1.0);
            b_max += Vec3::splat(1.0);
        }
        let center = 0.5 * (b_min + b_max);
        let bounding_radius = (0.5 * (b_max - b_min)).length().max(2.0);
        sectors.push(WorldSector {
            sector_id: sec_i as u32,
            aabb_min: b_min,
            aabb_max: b_max,
            center,
            bounding_radius,
            road_index_offset: road_ranges[sec_i].0,
            road_index_count: road_ranges[sec_i].1,
            terrain_index_offset: terrain_ranges[sec_i].0,
            terrain_index_count: terrain_ranges[sec_i].1,
            wire_index_offset: wire_ranges[sec_i].0,
            wire_index_count: wire_ranges[sec_i].1,
            prop_indices: std::mem::take(&mut sector_props[sec_i]),
        });
    }

    // Tighten internal octree node AABBs bottom-up to match their leaf sectors
    tighten_octree_bounds(&mut octree.nodes, root, &sectors);

    let hero = spline.sample(spline.hero_start_dist_m);
    let hero_sun_dir = match biome {
        BiomeId::CoastalPromenade => (hero.tangent * 0.85 + hero.right * 0.50 + Vec3::Y * 0.195).normalize(),
        BiomeId::MountainClimb => (hero.tangent * 0.72 + hero.right * 0.68 + Vec3::Y * 0.40).normalize(),
        BiomeId::RollingRuralGravel => (hero.tangent * 0.72 + hero.right * 0.68 + Vec3::Y * 0.38).normalize(),
    };

    StaticWorldGeometry {
        road_vertices,
        road_indices,
        terrain_vertices,
        terrain_indices,
        wire_vertices,
        wire_indices,
        water_vertices,
        water_indices,
        props,
        sectors,
        octree,
        hero_sun_dir,
    }
}
