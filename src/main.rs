#![allow(non_snake_case, mixed_script_confusables)]
#![feature(explicit_tail_calls)]

use clap::Parser as _;
use image::{Rgb, Rgb32FImage, RgbImage};
use indicatif::ProgressStyle;
use nalgebra::ComplexField;
use ndarray::{Array1, Array3, Axis, Dim, Ix1, Ix2, NdIndex, s};
use std::f32::consts::PI as PI32;
use std::f64::consts::PI as PI64;
use std::fmt::Debug;
use std::ops::Mul as _;
use std::{
    io::Write,
    path::{Path, PathBuf},
};
use ultraviolet::{self as uv, DMat4, DVec2, DVec3, DVec3x2, DVec4, IVec3, UVec2, f64x2};

#[derive(Debug, clap::Parser)]
struct Opts {
    #[arg(long, short)]
    output: PathBuf,
    #[arg(long, short)]
    prim: PathBuf,
    #[arg(long, short, default_value_t = 1)]
    grid: u32,
    #[arg(long, short, default_value_t = 1024)]
    size: u32,

    render: String,
}

fn dv3_contains(min: DVec3, max: DVec3, p: DVec3) -> bool {
    min.min_by_component(p) == min && max.max_by_component(p) == max
}

fn load_hdf5_prim(path: impl AsRef<Path>) -> Disk {
    // Primitive quantities (density, velocity, pressure) + cell-centered magnetic field vector
    let prim = hdf5::File::open(path.as_ref()).unwrap();

    // rest-frame dens rho,press,vel1,vel2,vel3
    let ds_prim = prim
        .dataset("prim")
        .unwrap()
        .read::<f32, ndarray::Ix5>()
        .unwrap();

    // coordinates: block, nx3, nx2, nx1
    let rest_frame_density = ds_prim.slice(s![0, .., .., .., ..]);
    let pressure = ds_prim.slice(s![1, .., .., .., ..]);
    // [Bcc1,Bcc2,Bcc3]
    // let Bcc = ds_prim.slice(s![2..5, .., .., .., ..]);

    // The RootGridXk (k=1..3) attributes describe the ranges of the coordinates used in the file;
    // in our case, these are r,θ,φ spherical coordinates
    let coords = {
        let root_grid_x1 = prim.attr("RootGridX1").unwrap().read::<f64, Ix1>().unwrap();
        let root_grid_x2 = prim.attr("RootGridX2").unwrap().read::<f64, Ix1>().unwrap();
        let root_grid_x3 = prim.attr("RootGridX3").unwrap().read::<f64, Ix1>().unwrap();
        [
            Coord::from(root_grid_x1),
            Coord::from(root_grid_x2),
            Coord::from(root_grid_x3),
        ]
    };
    println!("Coordinate limits: {coords:?}");

    // Actual data is stored in _cells_, which are divided into _mesh blocks_.
    // The "root grid" is the global logical cell grid, which exists only virtually.
    //
    // The dimensions of the root grid and the mesh blocks are stored in the RootGridSize and
    // MeshBlockSize attributes, respectively.
    let root_grid_size = {
        let root_grid_size = prim
            .attr("RootGridSize")
            .unwrap()
            .read::<i32, Ix1>()
            .unwrap();
        IVec3::new(root_grid_size[0], root_grid_size[1], root_grid_size[2])
    };
    let mesh_block_size = {
        let mesh_block_size = prim
            .attr("MeshBlockSize")
            .unwrap()
            .read::<i32, Ix1>()
            .unwrap();
        IVec3::new(mesh_block_size[0], mesh_block_size[1], mesh_block_size[2])
    };

    // LogicalLocations stores the locations of mesh blocks. Since our GR-MHD sims are running
    // without refinement, interpretation is fairly simple: the coordinates stored in
    // LogicalLocations are simply indices into a dense grid of mesh blocks
    let logical_locations = {
        let logical_locations = prim
            .dataset("LogicalLocations")
            .unwrap()
            .read::<i64, Ix2>() // [#block, 3] of i64
            .unwrap();
        logical_locations.map_axis(Axis(1), |ll| {
            IVec3::new(ll[0] as i32, ll[1] as i32, ll[2] as i32)
        })
    };

    // Cells are not uniform in size, so the xQf and xQv datasets store maps from cell coordinates
    // to worldspace coordinates.
    // block -> nx1 / nx2 / nx3 + 1 ::f32 cell boundary
    let x1f = prim.dataset("x1f").unwrap().read::<f32, Ix2>().unwrap();
    let x2f = prim.dataset("x2f").unwrap().read::<f32, Ix2>().unwrap();
    let x3f = prim.dataset("x3f").unwrap().read::<f32, Ix2>().unwrap();
    // block -> nx1 / nx2 / nx3 ::f32 of cell center
    let x1v = prim.dataset("x1v").unwrap().read::<f32, Ix2>().unwrap();
    let x2v = prim.dataset("x2v").unwrap().read::<f32, Ix2>().unwrap();
    let x3v = prim.dataset("x3v").unwrap().read::<f32, Ix2>().unwrap();

    // The absolute bounds of our disk
    let coord_min = DVec3::from(coords.map(|c| c.min));
    let coord_max = DVec3::from(coords.map(|c| c.max));

    // We'll be defining a lot of things based on cells, so we introduce a helper here.
    let cell_coords = Array3::from_shape_fn(
        root_grid_size.as_array().map(|x| x as usize),
        |(i, j, k)| IVec3::new(i as i32, j as i32, k as i32),
    );

    // We need to be able to map from cell coordinates to mesh block indices
    let cell_block = {
        let mut cell_block_table =
            Array3::from_elem(root_grid_size.as_array().map(|x| x as usize), 0);
        for (block_idx, iv) in logical_locations.iter().enumerate() {
            cell_block_table[iv.as_array().map(|x| x as usize)] = block_idx;
        }
        cell_coords
            .mapv(|iv| cell_block_table[(iv / mesh_block_size).as_array().map(|x| x as usize)])
    };
    let cell_sub_block = {
        cell_coords.mapv(|iv| {
            let (sub_i, sub_j, sub_k) = (
                (iv.x % mesh_block_size.x) as usize,
                (iv.y % mesh_block_size.y) as usize,
                (iv.z % mesh_block_size.z) as usize,
            );
            nalgebra::Vector3::new(sub_i, sub_j, sub_k)
        })
    };

    // Calculate some useful spatial information about cells
    let cell_min = cell_coords.mapv(|iv| {
        let block = cell_block[Idx3(iv)];
        let sub_ijk = cell_sub_block[Idx3(iv)];
        DVec3::new(
            x1f[[block, sub_ijk.x]] as f64,
            x2f[[block, sub_ijk.y]] as f64,
            x3f[[block, sub_ijk.z]] as f64,
        )
    });
    let cell_max = cell_coords.mapv(|iv| {
        let block = cell_block[Idx3(iv)];
        let sub_ijk = cell_sub_block[Idx3(iv)];
        DVec3::new(
            x1f[[block, sub_ijk.x + 1]] as f64,
            x2f[[block, sub_ijk.y + 1]] as f64,
            x3f[[block, sub_ijk.z + 1]] as f64,
        )
    });
    let cell_centroid = cell_coords.mapv(|iv| {
        let block = cell_block[Idx3(iv)];
        let sub_ijk = cell_sub_block[Idx3(iv)];
        DVec3::new(
            x1v[[block, sub_ijk.x]] as f64,
            x2v[[block, sub_ijk.y]] as f64,
            x3v[[block, sub_ijk.z]] as f64,
        )
    });
    // println!("cell_min[[0,0,0]] = {:?}", cell_min[[0, 0, 0]]);
    // println!("cell_min = {:?}", cell_min);
    // let cell_seq = [Array1::from_shape_fn([x1v.len()], |i| {
    //     DVec3::new(
    //         cell_min[[i, 0, 0]].x,
    //         cell_min[[0, i, 0]].y,
    //         cell_min[[0, 0, i]].z,
    //     )
    // });
    let cell_seq = [
        cell_min.slice(s![.., 0, 0]).mapv(|t| t.x),
        cell_min.slice(s![0, .., 0]).mapv(|t| t.y),
        cell_min.slice(s![0, 0, ..]).mapv(|t| t.z),
    ];
    // println!("cell_seq = {:?}", cell_seq[0]);

    // let cell_min = cell_coords.mapv(|iv| DVec3::from(iv) * cell_coord_span);
    // let cell_max = cell_coords.mapv(|iv| DVec3::from(iv) * cell_coord_span + cell_coord_span);
    // let cell_centroid = (&cell_min + &cell_max) / 2.;

    // Calculate some useful information about mesh blocks
    let block_min = logical_locations.mapv(|iv| cell_min[Idx3(iv * mesh_block_size)]);
    let block_max = logical_locations.mapv(|iv| cell_max[Idx3(iv * mesh_block_size)]);
    let block_centroid = block_min
        .iter()
        .zip(block_max.iter())
        .map(|(&min, &max)| (min + max) / 2.)
        .collect();

    // Sanity check
    print!("Sanity checking cell grid... ");
    for i in 0..root_grid_size.x {
        for j in 0..root_grid_size.y {
            for k in 0..root_grid_size.z {
                let c_min = cell_min[Idx3([i, j, k])];
                let c_max = cell_max[Idx3([i, j, k])];
                let c_block = cell_block[Idx3([i, j, k])];
                let (ii, jj, kk) = (
                    (i % mesh_block_size.x) as usize,
                    (j % mesh_block_size.y) as usize,
                    (k % mesh_block_size.z) as usize,
                );
                let xQv = DVec3::new(
                    x1v[[c_block, ii as usize]] as f64,
                    x2v[[c_block, jj as usize]] as f64,
                    x3v[[c_block, kk as usize]] as f64,
                );

                if !dv3_contains(c_min, c_max, xQv) {
                    println!("cell: {c_min:?} .. {c_max:?} DOES NOT CONTAIN xQv = {xQv:?}");
                }
            }
        }
    }
    println!("ok");

    print!("Extracting density and pressure information... ");
    // Extract density and pressure data from the HDF5 file
    let cell_density = cell_coords.mapv(|iv| {
        let block = cell_block[Idx3(iv)];
        let cell_ijk = cell_sub_block[Idx3(iv)];
        rest_frame_density[[block, cell_ijk.z, cell_ijk.y, cell_ijk.x]]
    });
    let density_min = *cell_density
        .iter()
        .reduce(|a, b| if a < b { a } else { b })
        .unwrap();
    let density_max = *cell_density
        .iter()
        .reduce(|a, b| if a > b { a } else { b })
        .unwrap();
    let cell_pressure = cell_coords.mapv(|iv| {
        let block = cell_block[Idx3(iv)];
        let cell_ijk = cell_sub_block[Idx3(iv)];
        pressure[[block, cell_ijk.z, cell_ijk.y, cell_ijk.x]]
    });
    let pressure_min = *cell_pressure
        .iter()
        .reduce(|a, b| if a < b { a } else { b })
        .unwrap();
    let pressure_max = *cell_pressure
        .iter()
        .reduce(|a, b| if a > b { a } else { b })
        .unwrap();
    println!("done");

    Disk {
        coord_min,
        coord_max,

        cell_density,
        density_limits: (density_min, density_max),
        cell_pressure,
        pressure_limits: (pressure_min, pressure_max),

        block_min,
        block_max,
        block_centroid,

        cell_min,
        cell_max,
        cell_centroid,

        cell_seq,

        coords,
    }
}

#[derive(Debug, Copy, Clone)]
struct Idx3<T: Debug + Copy + Into<[i32; 3]>>(T);
unsafe impl<T: Debug + Copy + Into<[i32; 3]>> NdIndex<Dim<[usize; 3]>> for Idx3<T> {
    fn index_checked(&self, dim: &Dim<[usize; 3]>, strides: &Dim<[usize; 3]>) -> Option<isize> {
        NdIndex::index_checked(&self.0.into().map(|x| x as usize), dim, strides)
    }

    fn index_unchecked(&self, strides: &Dim<[usize; 3]>) -> isize {
        NdIndex::index_unchecked(&self.0.into().map(|x| x as usize), strides)
    }
}

#[derive(Debug, Copy, Clone)]
struct Coord {
    min: f64,
    max: f64,
    grat: f64,
}
impl<A: std::ops::Index<usize, Output = f64>> From<A> for Coord {
    fn from(value: A) -> Self {
        Coord {
            min: value[0],
            max: value[1],
            grat: value[2],
        }
    }
}

struct Disk {
    coord_min: DVec3,
    coord_max: DVec3,

    cell_density: Array3<f32>,
    density_limits: (f32, f32),
    cell_pressure: Array3<f32>,
    pressure_limits: (f32, f32),

    block_min: Array1<DVec3>,
    block_max: Array1<DVec3>,
    block_centroid: Array1<DVec3>,

    cell_min: Array3<DVec3>,
    cell_max: Array3<DVec3>,
    cell_centroid: Array3<DVec3>,

    cell_seq: [Array1<f64>; 3],
    coords: [Coord; 3],
}
impl Disk {
    fn contains(&self, point: DVec3) -> bool {
        // point.min_by_component(self.coord_min) == self.coord_min
        //     && point.max_by_component(self.coord_max) == self.coord_max
        self.coord_min.x <= point.x && point.x <= self.coord_max.x
    }
    fn search(&self, point: DVec3) -> IVec3 {
        let (mut i, mut j, mut k) = (None, None, None);
        for l in 0..self.cell_seq[0].len() {
            if i.is_none() && point.x < self.cell_seq[0][l] {
                i = Some(l - 1);
            }
            if j.is_none() && point.y < self.cell_seq[1][l] {
                j = Some(l - 1);
            }
            if k.is_none() && point.z < self.cell_seq[2][l] {
                k = Some(l - 1);
            }
            if i.is_some() && j.is_some() && k.is_some() {
                break;
            }
        }

        IVec3::new(
            i.unwrap_or(self.cell_seq[0].len() - 1) as i32,
            j.unwrap_or(self.cell_seq[1].len() - 1) as i32,
            k.unwrap_or(self.cell_seq[2].len() - 1) as i32,
        )
    }
}

// struct Camera {
//     pos: DVec3,

//     fov: f64,
//     near: f64,
//     far: f64,
// }
// impl Camera {
//     fn focal_length(&self, h: u32) -> f64 {
//         (0.5 * (h as f64)) / ((self.fov / 2.).tan())
//     }
//     fn generate_ray_sph(&self, uv: DVec2, canvas_size: UVec2) -> (DVec3, DVec3) {
//         todo!()
//     }
// }

struct Ray {
    origin: DVec3,
    dir: DVec3,
}

fn checkerboard(uv: DVec2, n: u32) -> bool {
    let ij = uv * n as f64;
    (ij.x as u32 + ij.y as u32).is_multiple_of(2)
}

struct Uniforms {
    render_equatorial_slice: bool,
    render_lens_distortion: bool,
    render_non_relativistic: bool,
    render_momenta: bool,
    render_equatorial_photon_geodesics: bool,

    image_size: UVec2,
    gridding: UVec2,

    vertical_fov: f64,
    camera_pos_sph: DVec3,
    camera_movement_dir_sph: DVec3,

    J: f64,
    M: f64,

    dzeta: f64,
    eps: f64x2,
}

#[derive(Debug)]
struct Stats {
    minmax1: (f64, f64),
    minmax2: (f64, f64),
    minmax3: (f64, f64),
}
impl Stats {
    fn track1(&mut self, x: f64) {
        self.minmax1 = (self.minmax1.0.min(x), self.minmax1.1.max(x));
    }
    fn track2(&mut self, x: f64) {
        self.minmax2 = (self.minmax2.0.min(x), self.minmax2.1.max(x));
    }
    fn track3(&mut self, x: f64) {
        self.minmax3 = (self.minmax3.0.min(x), self.minmax3.1.max(x));
    }
}

fn march(
    disk: &Disk,
    stats: &mut Stats,
    uniforms: &Uniforms,
    screen_uv: DVec2,
    image: &mut Rgb32FImage,
) -> Rgb<f64> {
    if uniforms.render_equatorial_slice {
        let world_quadrant = DVec2::broadcast(disk.coords[0].max);
        let world_xy = (screen_uv * 2. - DVec2::broadcast(1.)) * world_quadrant;
        let world_r = world_xy.mag();
        let world_phi = world_xy.y.atan2(world_xy.x);
        let world_phi = if world_phi < 0. {
            PI64 + world_phi + PI64
        } else {
            world_phi
        };

        let eq_rtp = DVec3::new(world_r, PI64 / 2., world_phi);
        return if disk.contains(eq_rtp) {
            let cell_ijk = disk.search(eq_rtp);
            let cell_pressure = disk.cell_pressure[Idx3(cell_ijk)] as f64;
            let cell_density = disk.cell_density[Idx3(cell_ijk)] as f64;
            let pressure_range = 1. / disk.pressure_limits.1 as f64;
            let density_range = 1. / disk.density_limits.1 as f64;
            Rgb([
                cell_pressure * pressure_range,
                cell_density * density_range,
                0.,
            ])
        } else {
            Rgb([1., 0., 1.])
        };
    }
    // Calculate the coordinates of the point on the local sky represented by screen_uv.
    // Note that (pi/2,0) points away from the BH in the plane spanned by e_rn,e_phin.
    //
    // One thing I'm unclear about: what direction is this ray pointing? towards the camera or away
    // from the camera?
    let (theta_cs, phi_cs) = {
        // Get focal length and aspect ratio of the camera
        let focal_length = 0.5 * (2.0 / 2. as f64) / (0.5 * uniforms.vertical_fov).tan();
        let aspect_ratio = uniforms.image_size.x as f64 / uniforms.image_size.y as f64;
        // Convert screen space u,v in [0,1]^2 to normalized x,y in [-1,1]^2, purely for
        // convenience when converting to theta_cs, phi_cs
        let normalized_xy = screen_uv * 2. - DVec2::broadcast(1.);
        (
            PI64 / 2. - (normalized_xy.y / focal_length).atan(),
            PI64 - (normalized_xy.x / aspect_ratio / focal_length).atan(),
        )
    };

    // VISUALIZATION: Visualize fisheye lens distortion with checkerboard
    if uniforms.render_lens_distortion {
        // Convert theta_cs,phi_cs to a u,v in [0,1]^2 describing the coordinates in the local sky
        // representing points on the screen, and then render a checkerboard that has axis-aligned
        // rectangular tiles in the u,v space. The size of tiling will appear to increase towards
        // the edges if FOV is high.
        let sky_uv = DVec2::new(
            (theta_cs * PI64 / uniforms.vertical_fov + PI64) / 2. / PI64,
            (phi_cs * PI64 / uniforms.vertical_fov + PI64) / 2. / PI64,
        );
        return if checkerboard(sky_uv, 16) {
            Rgb([1., 1., 1.])
        } else {
            Rgb([0., 0., 0.])
        };
    }

    // Kerr parameter of the BH.
    let a = uniforms.J / uniforms.M;

    let [r_c, theta_c, phi_c] = *uniforms.camera_pos_sph.as_array();

    let rho_c = (r_c.powi(2) + a.powi(2) * theta_c.cos().powi(2)).sqrt();
    let Delta_c = r_c.powi(2) - 2. * r_c + a.powi(2);
    let Sigma_c =
        ((r_c.powi(2) + a.powi(2)).powi(2) - a.powi(2) * Delta_c * theta_c.sin().powi(2)).sqrt();
    let omega_bar_c = Sigma_c * theta_c.sin() / rho_c;

    // Speed of the camera
    let beta = {
        // Geodesic angular velocity at the camera's radius r_c
        let Omega = (a + r_c.powf(1.5)).recip();
        // ... a bunch of common quantities in the calculation of beta
        let alpha = rho_c * Delta_c.sqrt() / Sigma_c;
        let omega = 2. * a * r_c / Sigma_c.powi(2);
        omega_bar_c / alpha * (Omega - omega)
    };
    // Components of the direction of the camera's motion relative to the FIDO at its location.
    // NOTE: this MUST be a unit vector
    let [B_rn, B_thetan, B_phin] = *uniforms.camera_movement_dir_sph.as_array();

    // Cartesian components of the unit vector N that points in the direction of the incoming ray
    // in the camera's proper reference frame.
    let (N_x, N_y, N_z) = (
        theta_cs.sin() * phi_cs.cos(),
        theta_cs.sin() * phi_cs.sin(),
        theta_cs.cos(),
    );

    // The direction of motion of the incoming ray n_F as measured by the FIDO in Cartesian
    // coordinates _aligned_ with the camera
    let (n_Fy, n_Fx, n_Fz) = {
        let denom = 1. - beta * N_y;
        let fac = (1. - beta.powi(2)).sqrt();
        (
            (-N_y + beta) / denom,
            (-fac * N_x) / denom,
            (-fac * N_z) / denom,
        )
    };

    // The components of n_F on the FIDO's spherical orthonormal basis
    let (n_Frn, n_Fthetan, n_Fphin) = {
        let kappa = (1. - B_thetan.powi(2)).sqrt();
        (
            B_phin / kappa * n_Fx + B_rn * n_Fy + B_rn * B_thetan / kappa * n_Fz,
            B_thetan * n_Fy - kappa * n_Fz,
            -B_rn / kappa * n_Fx + B_phin * n_Fy + B_thetan * B_phin / kappa * n_Fz,
        )
    };

    // note: e_y of the camera's proper reference frame is in the camera's proper reference frame
    //       e_x is then the perpendicular of e_y that lies in the e_rn,e_phin plane
    //       e_z is the natural 3rd
    //       theta_cs,phi_cs are defined with regard to these!
    if uniforms.render_non_relativistic {
        let e_rn = DVec3::new(Delta_c.sqrt() / rho_c, 0., 0.);
        let e_thetan = DVec3::new(0., 1. / rho_c, 0.);
        let e_phin = DVec3::new(0., 0., 1. / omega_bar_c);

        let e_y = DVec3::new(B_rn, B_thetan, B_phin);
        // e_rn = Delta.sqrt()/rho d/dr, e_thetan = 1/rho d/dtheta, e_phin = 1/omega_bar d/dphi
        // then if e_x is in the plane of e_rn,e_phin, e_x perp e_thetan and e_x perp e_y, so
        // e_x is just e_y x e_thetan
        // Thankfully, e_y and e_thetan are both normal {e_x,e_y,e_z} is an orthonormal vasis, as
        // is {e_rn,e_thetan,e_phin}
        let e_x = -e_y.cross(e_thetan);
        let e_z = -e_x.cross(e_y);

        let origin_r = DVec3::new(
            r_c * theta_c.sin() * phi_c.cos(),
            r_c * theta_c.sin() * phi_c.sin(),
            r_c * theta_c.cos(),
        );

        // Spherical components that point in the direction of the incoming ray in the camera's
        // reference frame
        let N_sph = N_x * e_x + N_y * e_y + N_z * e_z;
        let N_sph_world = N_sph.x * e_rn + N_sph.y * e_thetan + N_sph.z * e_phin;

        let dir_r = -DVec3::new(
            N_sph_world.x * N_sph_world.y.sin() * N_sph_world.z.cos(),
            N_sph_world.x * N_sph_world.y.sin() * N_sph_world.z.sin(),
            N_sph_world.x * N_sph_world.y.cos(),
        );
        // .normalized();

        // Convert N_sph to to worldspace by handling the rotation

        // let N_F_xyz = DVec3::new(
        //     N_F_sph.x * N_F_sph.y.sin() * N_F_sph.z.cos(),
        //     N_F_sph.x * N_F_sph.y.sin() * N_F_sph.z.sin(),
        //     N_F_sph.x * N_F_sph.y.cos(),
        // );
        stats.track1(theta_cs);
        stats.track2(phi_cs);
        // stats.track3(N_sph_world.z);
        println!(
            "uv={screen_uv:0.3?} θφ_cs = {:0.3},{:0.3}, N = {:0.4?} N_sph_world = {N_sph_world:0.4?}, dir_r = {dir_r:04.?} n_FQn = {:0.4?}, e_y={e_y:0.4?} e_x={e_x:0.4?} e_z={e_z:0.4?} e_thetan={e_thetan:0.4?}",
            theta_cs / PI64,
            phi_cs / PI64,
            DVec3::new(N_x, N_y, N_z),
            DVec3::new(n_Frn, n_Fthetan, n_Fphin)
        );

        for i in 0..32 {
            let x_r_t = origin_r + (i as f64) * DVec3::new(N_x, N_y, N_z);
            println!("x_r_t = {x_r_t:?}");
            if x_r_t.mag() <= 20. {
                return Rgb([1., 0., 0.]);
            }
        }

        return Rgb([0., 1., 0.]);

        // return Rgb([0.5 + 1000. * N_sph_world.y, 0.5 + 100. * N_sph_world.z, 0.]);
        // return Rgb([theta_cs / PI64, phi_cs / PI64 / 2., 0.]);
    }

    // Ray's canonical momenta (covariant coordinate components of its 4-momentum) with conserved
    // energy -p_t set to unity as a convention
    let (p_t, p_r, p_theta, p_phi) = {
        // Not quite sure where we're supposed to be evaluating alpha/omega/omega_bar/rho/Delta but
        // probably at the FIDO, which has the same coords as the camera
        let rho = (r_c.powi(2) + a.powi(2) * theta_c.cos().powi(2)).sqrt();
        let Delta = r_c.powi(2) - 2. * r_c + a.powi(2);
        let Sigma =
            ((r_c.powi(2) + a.powi(2)).powi(2) - a.powi(2) * Delta * theta_c.sin().powi(2)).sqrt();
        let alpha = rho * Delta.sqrt() / Sigma;
        let omega = 2. * a * r_c / Sigma.powi(2);
        let omega_bar = Sigma * theta_c.sin() / rho;
        let E_F = 1. / (alpha + omega * omega_bar * n_Fphin);
        (
            -1.,
            E_F * rho / Delta.sqrt() * n_Frn,
            E_F * rho * n_Fthetan,
            E_F * omega_bar * n_Fphin,
        )
    };

    if uniforms.render_momenta {
        stats.track1(p_r);
        stats.track2(p_theta);
        stats.track3(p_phi);

        return Rgb([0.5 + (p_theta / 2. / 40.), 0.5 + (p_phi / 2. / 47.), 0.]);
    }

    // Other two conserved quantities: axial angular momentum and Carter constant
    let (b, q) = {
        // I'm somewhat unsure what the theta in the Carter constant expression is supposed to
        // refer to; again assuming that it's the camera
        (
            p_phi,
            p_theta.powi(2)
                + theta_c.cos().powi(2) * (p_phi.powi(2) / theta_c.sin().powi(2) - a.powi(2)),
        )
    };

    if uniforms.render_equatorial_photon_geodesics {
        let N = 1024 * 16;
        let mut zeta = 1.0;
        let mut rtp = DVec3::new(r_c, theta_c, phi_c);
        let mut p_sph = -DVec3::new(p_r, p_theta, p_phi);
        let mut h = uniforms.dzeta;
        // println!("Initial p_sph: {p_sph:?} a={a}  b={b} q={q}");
        for _ in 0..N {
            let (drtp, dp_sph, h_new, _h_fwd) =
                tsit5(compute_motion, rtp, p_sph, (a, b, q), h, uniforms.eps);
            rtp += drtp;
            p_sph += dp_sph;
            h = h_new;
            // println!(
            //     "Rtp: {:0.4},{:0.4},{:0.4} + {:0.4},{:0.4},{:0.4} -> {:0.4},{:0.4},{:0.4}",
            //     rtp.x, rtp.y, rtp.z, drtp.x, drtp.y, drtp.z, rtp_p.x, rtp_p.y, rtp_p.z
            // );
            // println!(
            //     "p: {:0.4},{:0.4},{:0.4} + {:0.4},{:0.4},{:0.4} -> {:0.4},{:0.4},{:0.4}",
            //     p_sph.x, p_sph.y, p_sph.z, dp_sph.x, dp_sph.y, dp_sph.z, p_sph_p.x, p_sph_p.y, p_sph_p.z
            // );

            let x = 512 + (511. * rtp.x / 100. * rtp.z.cos()) as i32;
            let y = 512 + (511. * rtp.x / 100. * rtp.z.sin()) as i32;
            if x < 0
                || x as u32 >= uniforms.image_size.x
                || y < 0
                || y as u32 >= uniforms.image_size.y
            {
                return Rgb([screen_uv.x, screen_uv.y, 0.]);
            }
            // println!("xy={x},{y} uv={screen_uv:?}");
            zeta *= 0.99;
            image.put_pixel(
                x as u32,
                y as u32,
                Rgb([screen_uv.x as f32, screen_uv.y as f32, zeta]),
            );
        }
        return Rgb([screen_uv.x, screen_uv.y, 0.]);
    }

    // FULL RAYMARCHING!
    let N = 1024 * 4;
    let mut eta = 1.0;
    let mut rtp = DVec3::new(r_c, theta_c, phi_c);
    let mut p_sph = -DVec3::new(p_r, p_theta, p_phi);
    // println!("Initial p_sph: {p_sph:?} a={a}  b={b} q={q}");

    fn sphere2cart(x: DVec3) -> DVec3 {
        DVec3::new(
            x.x * x.y.sin() * x.z.cos(),
            x.x * x.y.sin() * x.z.sin(),
            x.x * x.y.cos(),
        )
    }

    let mut w = DVec3::zero();
    let mut h = uniforms.dzeta;
    let mut h_tot = 0.;

    for _ in 0..N {
        // println!();
        // println!("rtp={rtp:0.4?} p={p_sph:0.4?} h={h}");
        let (drtp, dp_sph, h_new, h_fwd) =
            tsit5(compute_motion, rtp, p_sph, (a, b, q), h, uniforms.eps);
        // let (drtp, dp_sph) = rk4(compute_motion, rtp, p_sph, (a, b, q), h);
        rtp += drtp;
        p_sph += dp_sph;
        h = h_new;
        h_tot += h_fwd;
        // h_tot += h;

        // let rtp_pre = rtp;

        rtp = canonicalize_spherical_coords(rtp);
        // p_sph = canonicalize(p_sph);

        // {
        //     let xyz_pre = sphere2cart(rtp_pre);
        //     let xyz = sphere2cart(rtp);
        //     let pre_post_rat = (xyz_pre - xyz).mag() / (xyz.mag() + xyz_pre.mag());
        //     assert!(
        //         pre_post_rat.abs() < 0.00001,
        //         "differ too much: {:?} ({rtp_pre:?}) vs. {:?} ({rtp:?}), ratio={pre_post_rat}",
        //         sphere2cart(rtp_pre),
        //         sphere2cart(rtp)
        //     );
        // }
        if rtp.x <= 1. {
            println!("horizon hit! h_tot = {h_tot}");
            return Rgb([0., 0., 0.]);
        }

        if disk.contains(rtp) {
            // ============ Code Units ============
            // code Mass = 8.540000e+39 g
            // code Length = 2.997925e+10 cm
            // code Time = 1.000000e+00 s
            // code density = 3.169537e+08 g/cm^3
            // code velocity = 2.997925e+10 cm/s
            // code pressure = 2.848637e+29 erg/cm^3
            // code temperature = 1.089254e+13 ??
            // ====================================
            // ===== Constants  in Code Units =====
            // dyne in code = 3.905903e-51
            // erg in code = 1.302869e-61
            // Gconst in code = 2.114902e+01
            // Msun in code = 2.328349e-07
            // Lsun in code = 4.985819e-28
            // Myr in code = 3.155760e+13
            // kB in code = 1.798816e-77
            // c in code = 1.000000e+00
            // e in code = 3.549572e-45
            // mH in code = 1.959368e-64
            // ====================================
            let cell_ijk = disk.search(rtp);
            let cell_pressure = disk.cell_pressure[Idx3(cell_ijk)] as f64;
            let cell_density = disk.cell_density[Idx3(cell_ijk)] as f64;
            let pressure_range = 1. / disk.pressure_limits.1 as f64;
            let density_range = 1. / disk.density_limits.1 as f64;
            stats.track1(cell_pressure);
            stats.track2(cell_density);
            let k_B = 1.798816e-77;
            let m_H = 1.959368e-64;
            // let k_B_ism = 2.006008e-58;
            // let m_H_ism = 2.431198e-56;
            let R_spec = k_B / m_H;
            let cell_temp = cell_pressure / cell_density / R_spec / 1.089254e+13;
            let h_cgs = 6.26196e-27; // erg s
            // erg = 1.302 869 e -61 [E]
            // [E] = [M] [L^2] [T^-2] = 8.54e39 g 3e20 cm^2 / s^2
            let E_per_erg = 1.302869e-61;
            let h = h_cgs * E_per_erg;
            let nu = 3e16;
            let T = cell_temp;
            // E/sr/L^3 = [M] [L^2] [T^-2] sr^-1 [L^-3] = M [T^-2] sr^-1 [L^-1]
            let B_nu = 2. * h * nu.powi(3);
            // h = e-88 nu = e16
            // k_B = e-77 T=e-2 .. e-5
            let expf = h * nu / k_B / T;
            let div = expf.exp_m1() - 1.;
            stats.track3(T);
            let B_nu = B_nu / div;
            // let temp = cell_pressure;
            // println!(
            //     "cell_ijk={cell_ijk:0.4?} P={cell_pressure}/{pressure_range} rho={cell_density}/{density_range}"
            // );
            w = w.max_by_component(DVec3::new(
                cell_pressure * pressure_range,
                cell_density * density_range,
                B_nu,
            ));
        }
        if rtp.x.abs() >= 200. {
            break;
        }
    }
    // println!("h_tot = {h_tot} rtp={rtp:0.2?}");
    // println!("w = {w:?}");
    return Rgb([w.x, w.y, w.z]);

    // let b_o = |r_o: f64| -> f64 {
    //     -(r_o.powi(3) - 3. * r_o.powi(2) + kerr_p.powi(2) * r_o + kerr_p.powi(2))
    //         / kerr_p
    //         / (r_o - 1.)
    // };
    // let q_o = |r_o: f64| -> f64 {
    //     -(r_o.powi(3) * (r_o.powi(3) - 6. * r_o.powi(2) + 9. * r_o - 4. * kerr_p.powi(2)))
    //         / kerr_p.powi(2)
    //         / (r_o - 1.).powi(2)
    // };
    // let r_1 = 2. * { 1. + (2. / 3. * (-kerr_p).acos()).cos() };
    // let r_2 = 2. * { 1. + (2. / 3. * (kerr_p).acos()).cos() };

    // let from_celestial_sphere = {
    //     let (b_1, b_2) = (b_o(r_2), b_o(r_1));
    //     let q_ob = q_o(b);

    //     if b_1 < b && b < b_2 && q < q_ob {
    //         // There are no radial turning points for this {b,q}, whence if p_r>0 at the camera's
    //         // location, the ray comes from the horizon, and if p_r<0 it comes from the celestial
    //         // sphere.
    //         p_r < 0.
    //     } else {
    //         // There are two radial turning points for this {b,q}, and if the camera radius
    //         // r_c>=r_up of the upper turning point, then the ray comes from the celestial sphere;
    //         // otherwise it comes from the horizon (here r_up is the largest real root of R(r) = 0,
    //         // with R(r) being from the equations for the null geodesic).
    //         let r_up = todo!();
    //         r_c >= r_up
    //     }
    // };

    // Rgb([1., 0., 1.])
}

// Equations of motion for the null geodesic
fn compute_motion(y: DVec3x2, (a, b, q): (f64, f64, f64)) -> DVec3x2 {
    let [rtp, p_sph] = y.into();
    let Delta = rtp.x.powi(2) - rtp.x * 2. + a.powi(2);
    let rho2 = rtp.x.powi(2) + a.powi(2) * rtp.y.cos().powi(2);
    let P = rtp.x.powi(2) + a.powi(2) - a * b;
    let R = P.powi(2) - Delta * ((b - a).powi(2) + q);
    let Theta = q - rtp.y.cos().powi(2) * (b.powi(2) / rtp.y.sin().powi(2) - a.powi(2));

    let dr_dzeta = Delta * p_sph.x / rho2;
    let dtheta_dzeta = p_sph.y / rho2;
    let dphi_dzeta = -(Delta * rho2).recip() * {
        a * (Delta + 1.) - b * (Delta + rtp.y.cos().powi(2) / rtp.y.sin().powi(2))
    };

    #[rustfmt::skip]
        let dp_r_dzeta = {
            let dDelta_dr = 2. * rtp.x - 1.;
            let dP_dr = 2. * rtp.x;
            let dR_dr = 2. * P * dP_dr - ((b - a).powi(2) + q) * dDelta_dr;
            let drho2_dr = 2. * rtp.x;
            -(
                a.powi(2) * rtp.y.cos().powi(2) * (rtp.x - 1.) + rtp.x * (rtp.x - a.powi(2))
            ) / rho2.powi(2) * p_sph.x.powi(2)
            + p_sph.y.powi(2) * rtp.x / rho2.powi(2)
            + (
                2. * Delta * rho2 * (dR_dr + dDelta_dr * Theta) - (R + Delta * Theta) * (2. * Delta * drho2_dr + 2. * rho2 + dDelta_dr)
            ) / (2. * Delta * rho2).powi(2)
        };
    #[rustfmt::skip]
        let dp_theta_dzeta = {
            -(rtp.x.powi(2) - 2. * rtp.x + a.powi(2)) * p_sph.x.powi(2) * (a.powi(2) * rtp.y.cos() * rtp.y.sin()) / rho2.powi(2)
            - a.powi(2) * rtp.y.cos() * rtp.y.sin() * p_sph.y.powi(2) / rho2.powi(2)
            + (
                2. * Delta.powi(2) * rho2 * (-a.powi(2) * rtp.y.mul(2.).sin() - 2. * b.powi(2) * rtp.y.tan().recip() * rtp.y.sin().recip())
                - (R + Delta * Theta) * (-4. * Delta * a.powi(2) * rtp.y.cos() * rtp.y.sin())
            ) / 4. / Delta.powi(2) / rho2.powi(2)
        };

    DVec3x2::from([
        DVec3::new(dr_dzeta, dtheta_dzeta, dphi_dzeta),
        DVec3::new(dp_r_dzeta, dp_theta_dzeta, 0.),
    ])
}

// Explicit 4th-order Runge-Kutta Integrator
fn rk4<T: Copy>(
    f: fn(DVec3x2, T) -> DVec3x2,
    rtp: DVec3,
    p_sph: DVec3,
    t: T,
    dzeta: f64,
) -> (DVec3, DVec3) {
    let y = DVec3x2::from([rtp, p_sph]);
    let k1 = f(y, t);
    let k2 = f(y + k1 * f64x2::splat(dzeta / 2.), t);
    let k3 = f(y + k2 * f64x2::splat(dzeta / 2.), t);
    let k4 = f(y + k3 * f64x2::splat(dzeta), t);
    let dy = f64x2::splat(dzeta / 6.) * (k1 + f64x2::splat(2.) * k2 + f64x2::splat(2.) * k3 + k4);
    let [drtp, dp_sph] = dy.into();
    (drtp, -dp_sph)
}

// Tsitouras 5/4 Runge-Kutta Integrator
fn tsit5<T: Copy>(
    f: fn(DVec3x2, T) -> DVec3x2,
    rtp: DVec3,
    p_sph: DVec3,
    t: T,
    h_: f64,
    eps: f64x2,
) -> (DVec3, DVec3, f64, f64) {
    let y = DVec3x2::from([rtp, p_sph]);

    let [c2, c3, c4, c5, c6, _c7] =
        [0.161, 0.327, 0.9, 0.9800255409045097, 1., 1.].map(f64x2::splat);
    let [b1, b2, b3, b4, b5, b6, b7] = [
        0.09646076681806523,
        0.01,
        0.4798896504144996,
        1.379008574103742,
        -3.290069515436081,
        2.324710524099774,
        0.,
    ]
    .map(f64x2::splat);
    let [B1, B2, B3, B4, B5, B6, B7] = [
        0.001780011052226,
        0.000816434459657,
        -0.007880878010262,
        0.144711007173263,
        -0.582357165452555,
        0.458082105929187,
        1. / 66.,
    ]
    .map(f64x2::splat);
    let [A3_2] = [0.3354806554923570].map(f64x2::splat);
    let [A4_2, A4_3] = [-6.359448489975075, 4.362295432869581].map(f64x2::splat);
    let [A5_2, A5_3, A5_4] =
        [-11.74888356406283, 7.495539342889836, -0.09249506636175525].map(f64x2::splat);
    let [A6_2, A6_3, A6_4, A6_5] = [
        -12.92096931784711,
        8.159367898576159,
        -0.07158497328140100,
        -0.02826905039406838,
    ]
    .map(f64x2::splat);
    fn A_k1(c: f64x2, a: &[f64x2]) -> f64x2 {
        c - a.iter().sum::<f64x2>()
    }
    let [A2_1, A3_1, A4_1, A5_1, A6_1] = [
        A_k1(c2, &[]),
        A_k1(c3, &[A3_2]),
        A_k1(c4, &[A4_2, A4_3]),
        A_k1(c5, &[A5_2, A5_3, A5_4]),
        A_k1(c6, &[A6_2, A6_3, A6_4, A6_5]),
    ];
    let [A7_1, A7_2, A7_3, A7_4, A7_5, A7_6] = [b1, b2, b3, b4, b5, b6];
    let h = f64x2::splat(h_);

    let f1 = f(y, t);
    let f2 = f(y + h * A2_1 * f1, t);
    let f3 = f(y + h * A3_1 * f1 + h * A3_2 * f2, t);
    let f4 = f(y + h * A4_1 * f1 + h * A4_2 * f2 + h * A4_3 * f3, t);
    let f5 = f(
        y + h * A5_1 * f1 + h * A5_2 * f2 + h * A5_3 * f3 + h * A5_4 * f4,
        t,
    );
    let f6 = f(
        y + h * A6_1 * f1 + h * A6_2 * f2 + h * A6_3 * f3 + h * A6_4 * f4 + h * A6_5 * f5,
        t,
    );
    let f7 = f(
        y + h * A7_1 * f1
            + h * A7_2 * f2
            + h * A7_3 * f3
            + h * A7_4 * f4
            + h * A7_5 * f5
            + A7_6 * f6,
        t,
    );

    let dy_p = h * b1 * f1
        + h * b2 * f2
        + h * b3 * f3
        + h * b4 * f4
        + h * b5 * f5
        + h * b6 * f6
        + h * b7 * f7;
    let dy_n = h * B1 * f1
        + h * B2 * f2
        + h * B3 * f3
        + h * B4 * f4
        + h * B5 * f5
        + h * B6 * f6
        + h * B7 * f7;

    // let [ddrtp, ddp] = <[DVec3; 2]>::from(y_p - y_n).map(sphere2cart);
    // let E_n_sqr = (ddrtp.mag_sq() + ddp.mag_sq());
    // let E_n = E_n_sqr.sqrt();
    let E_n = (dy_p - dy_n).abs();
    // println!("f1={f1:0.2?} f2={f2:0.2?} f3={f3:0.2?} f4={f4:0.2?} f5={f5:0.2?} f6={f6:0.2?} f7={f7:0.2?}");
    // println!("E_n={E_n:0.2?} dy_p={dy_p:0.2?} y_n={dy_n:0.2?}");
    let h_new = 0.9 * h * (DVec3x2::broadcast(eps) / E_n).map(|x| x.powf(0.2));
    let [hrtp, hp] = h_new.into();
    let hh = hrtp.min_by_component(hp);
    let h_min = hh.x.min(hh.y).min(hh.z);
    // originally: eps >= E_n
    if DVec3x2::broadcast(eps).max_by_component(E_n) == DVec3x2::broadcast(eps) {
        // For p(p-1) RK, use 1/p; this is a 5(4), so 1/5
        let [drtp, dp] = dy_p.into();
        (drtp, -dp, h_min, h_)
    } else {
        // println!("[x={rtp:0.2?} p={p_sph:0.2?}] error {E_n:0.2?} (ŷ={dy_n:0.2?}, y'={dy_p:0.2?}) exceeds tolerance {eps};
        // restarting with h: {h_} -> {h_new:0.2?} f1={f1:0.2?} f2={f2:0.2?} f3={f3:0.2?} f4={f4:0.2?} f5={f5:0.2?} f6={f6:0.2?} f7={f7:0.2?}");
        if h_min.is_nan() {
            panic!();
        }
        become tsit5(f, rtp, p_sph, t, h_min, eps)
    }
}

fn canonicalize_spherical_coords(mut sph: DVec3) -> DVec3 {
    if sph.x < 0. {
        sph.x = -sph.x;
        sph.y = PI64 - sph.y;
        sph.z += PI64;
    }
    // Put theta into the range [-2pi,2pi]
    sph.y = sph.y % (2. * PI64);
    // Narrow the range to [0,2pi]
    if sph.y < 0. {
        sph.y += 2. * PI64;
    }
    if sph.y >= PI64 {
        // Reflect across the theta axis and then rotate around phi axis
        sph.y = PI64 - (sph.y - PI64);
        sph.z += PI64;
    }
    sph.z = sph.z % (2. * PI64);
    if sph.z < 0. {
        sph.z += 2. * PI64;
    }
    sph
}

fn sample_image(uniforms: Uniforms, disk: Disk) -> RgbImage {
    let mut image = Rgb32FImage::new(uniforms.image_size.x, uniforms.image_size.y);

    let image_to_normalized = DVec2::from(uniforms.image_size);

    let mut stats = Stats {
        minmax1: (f64::INFINITY, f64::NEG_INFINITY),
        minmax2: (f64::INFINITY, f64::NEG_INFINITY),
        minmax3: (f64::INFINITY, f64::NEG_INFINITY),
    };

    let pb = indicatif::ProgressBar::new(
        (uniforms.image_size / uniforms.gridding)
            .as_slice()
            .iter()
            .copied()
            .product::<u32>() as u64,
    )
    .with_style(
        ProgressStyle::with_template(
            "{wide_bar:.cyan/blue} {percent:.green}% [{pos}/{len}] {eta_precise}",
        )
        .unwrap(),
    );

    for image_y in 0..uniforms.image_size.y {
        if !image_y.is_multiple_of(uniforms.gridding.y) {
            continue;
        }
        for image_x in 0..uniforms.image_size.x {
            if !image_x.is_multiple_of(uniforms.gridding.x) {
                continue;
            };
            pb.inc(1);
            // upper left is 0,0
            // if image_x == image_size.x / 2 || image_y == image_size.y / 2 {
            //     image.put_pixel(image_x, image_y, Rgb([0., 1., 1.]));
            //     continue;
            // }
            let image_xy = UVec2::new(image_x, image_y);
            let screen_uv = DVec2::from(UVec2::new(image_xy.x, uniforms.image_size.y - image_xy.y))
                / image_to_normalized;
            assert!((0.0..=1.0).contains(&screen_uv.x));
            assert!((0.0..=1.0).contains(&screen_uv.y));

            let rgb64 = march(&disk, &mut stats, &uniforms, screen_uv, &mut image);

            image.put_pixel(image_x, image_y, Rgb(rgb64.0.map(|x| x as f32)));
        }
    }
    pb.finish();
    println!("stats = {stats:?}");

    image::DynamicImage::ImageRgb32F(image).into_rgb8()
}

fn main() {
    let opts = Opts::parse();
    let disk = load_hdf5_prim(&opts.prim);

    let uniforms = Uniforms {
        render_equatorial_slice: opts.render == "slice",
        render_lens_distortion: opts.render == "distortion",
        render_non_relativistic: opts.render == "vp_nonrel",
        render_momenta: opts.render == "momenta",
        render_equatorial_photon_geodesics: opts.render == "eq_geodesics",

        image_size: UVec2::new(opts.size, opts.size),
        gridding: UVec2::new(opts.grid, opts.grid),
        vertical_fov: PI64 / 4.,
        // Equatorial circular geodesic orbit:
        camera_pos_sph: DVec3::new(50., PI64 / 2., 0.),
        camera_movement_dir_sph: DVec3::new(0., 0., 1.0),
        // MUST correspond with accretion disk parameters
        J: 0.5,
        M: 1.0,

        dzeta: 0.1,
        eps: f64x2::new([0.1, 1.]),
    };
    let image_buffer = sample_image(uniforms, disk);
    let mut output = std::fs::OpenOptions::new()
        .write(true)
        .create(true)
        .open(opts.output)
        .unwrap();
    image_to_png(image_buffer, &mut output);
}

fn image_to_png(image: RgbImage, out: &mut impl Write) {
    let encoder = image::codecs::png::PngEncoder::new_with_quality(
        out,
        image::codecs::png::CompressionType::Uncompressed,
        image::codecs::png::FilterType::Adaptive,
    );
    let () = image.write_with_encoder(encoder).unwrap();
}
