#![allow(non_snake_case, mixed_script_confusables)]

use std::{
    f64::{self, consts::PI},
    io::Write,
};

use image::{Rgb, Rgb32FImage, RgbImage};
use ultraviolet::{DVec2, DVec3, UVec2, Vec2};

fn main() {
    let image_buffer = sample_image(
        Params {
            image_size: UVec2::new(1024, 1024),
        },
        Render::AccretionDisk(RenderAccretionDisk {
            camera_dist_of_schwarzschild: 6.,
            focal_length: 0.00000001,
            bh_M: 1.,
            bh_J: 0.,
            prograde: true,
        }),
    );
    let path = "output.png";
    let mut output = std::fs::OpenOptions::new()
        .write(true)
        .create(true)
        .open(path)
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

struct Params {
    image_size: UVec2,
}
enum Render {
    Gradient,
    AccretionDisk(RenderAccretionDisk),
}
struct RenderAccretionDisk {
    // Mass-equivalent of the central black hole
    bh_M: f64,
    // Angular momentum of the central black hole (axis assumed to be +z)
    bh_J: f64,
    // Directionality of accretion disk rotation
    prograde: bool,
    // Camera distance as a multiple of the Schwarzschild radius
    camera_dist_of_schwarzschild: f64,
    // Focal length in natural units
    focal_length: f64,
}
fn sample_image(params: Params, render: Render) -> RgbImage {
    let mut img = Rgb32FImage::new(params.image_size.x, params.image_size.y);
    match render {
        Render::Gradient => render_gradient(params, &mut img),
        Render::AccretionDisk(disk) => render_accretion_disk(disk, params, &mut img),
    }
    image::DynamicImage::ImageRgb32F(img).into_rgb8()
}
fn render_gradient(params: Params, img: &mut Rgb32FImage) {
    for y in 0..params.image_size.y {
        for x in 0..params.image_size.x {
            let x_rel = x as f32 / (params.image_size.x - 1) as f32;
            let y_rel = y as f32 / (params.image_size.y - 1) as f32;
            img.put_pixel(x, y, Rgb([x_rel, y_rel, 0.]));
        }
    }
}
fn render_accretion_disk(disk: RenderAccretionDisk, params: Params, img: &mut Rgb32FImage) {
    // r_s = 2GM / c^2, so r_s = 2M in natural units
    let r_s = 2. * disk.bh_M;

    let r_ms = isco_nospin_rotating(disk.bh_M, disk.bh_J, disk.prograde);

    let r_outer = 10. * r_s; // TODO: properly

    // we can now calculate the size of the screen in worldspace given the focal length
    //  sz_wo : f  = r_outer : (f + d_c)
    let d_s = disk.camera_dist_of_schwarzschild * r_s;
    let wo_size = 2. * r_outer * disk.focal_length * (disk.focal_length + d_s);

    // need to figure out the units; we know that G=c=1, but not the rest
    // we could set M=1, but that's inconvenient
    // σ = 2π⁵k⁴ / 15c²h³
    let k = 0.0f64; // Boltzmann constant
    let h = 0.0f64; // Planck constant
    let σ = 2. * PI.powi(5) * k.powi(4) / 15. / h.powi(3); // Stefan-Boltzmann constant
    let v_R = 0.0f64; // radial drift velocity (oriented inwards)
    let a = disk.bh_J / disk.bh_M; // Kerr parameter

    // Calculate T_* constant
    let T_star = T_star(1., a, disk.bh_M, r_ms, σ, v_R);

    for iy in 0..params.image_size.y {
        for ix in 0..params.image_size.x {
            // screenspace xy in [0,1)^2
            let px_low = DVec2::new(ix as f64, iy as f64);
            let screen_xy0 = px_low / DVec2::from(params.image_size);
            let screen_xy1 = (px_low + DVec2::broadcast(1.)) / DVec2::from(params.image_size);
            let screen_xy0_5 = (screen_xy0 + screen_xy1) / 2.;
            let wo_xy0_5 = screen_xy0_5 * wo_size;
            let r_0 = DVec3::new(wo_xy0_5.x, wo_xy0_5.y, -d_s);
            let r_d = -DVec3::unit_z();

            let Some(t) = euclidean_isect_ray_plane(r_0, r_d, DVec3::zero(), DVec3::unit_z())
            else {
                panic!()
            };
            let x = r_0 + r_d * t;
            let R = x.mag();
        }
    }
}

/// Calculate the temperature of a thin accretion disk from the temperature parameter T_* and the
/// dimensionless radius r = R/R_ms where R_ms is the marginally stable ISCO radius.
fn T_zt(T_star: f64, r: f64) -> f64 {
    assert!(
        r.is_normal() && r >= 0.,
        "radius coordinate on accretion disk must be a positive real number"
    );
    // TODO: is .powf(-0.75) faster or is .sqrt().powi(3).recip()
    //       is .powf(0.25) faster or is .sqrt().sqrt()
    T_star * r.powf(-0.75) * (1. - r.sqrt().recip()).powf(0.25)
}

/// Calculate the maximum temperature of a thin accretion disk around a rotating black hole with a
/// zero torque boundary condition at the inner radius using the T_* parameter of
/// Zimmerman et al.'s 2005 ezdiskbb model.
fn T_max(T_star: f64) -> f64 {
    T_star * 0.488
}

/// Calculate the T_* parameter of Zimmerman et al.'s 2005 ezdiskbb model.
///
/// Parameters:
///  f - the blackbody emission factor (f=1 for canonical spectrum)
///  M - black hole mass-equivalent
///  a - Kerr parameter
///  R_in - inner radius of accretion disk (= marginally stable orbital radius)
///  σ - Stefan-Boltzmann constant
///  v_R - radial drift velocity, oriented inwards
fn T_star(f: f64, M: f64, a: f64, R_in: f64, σ: f64, v_R: f64) -> f64 {
    assert!(
        R_in.is_normal() && R_in >= 0.,
        "inner radius of accretion disk must be a positive real number"
    );
    assert!(
        M.is_normal() && M >= 0.,
        "mass of black hole must be a positive real number"
    );
    let Σ = R_in.powi(2) + a.powi(2); // surface density at radius R_in
    let Ṁ = 2. * PI * R_in * Σ * -v_R; // mass accretion rate
    (f * 3. * M * Ṁ / 8. / PI / R_in.powi(3) / σ).powf(0.25)
}

/// Get the Schwarzschild radius for a body with mass-equivalent M.
fn schwarzschild_radius(M: f64) -> f64 {
    2. * M
}

/// Innermost stable circular orbit for body orbiting a black hole of mass M with either
/// prograde or retrograde orbit.
///
/// Restrictions: this assumes a non-spinning particle.
///
/// Parameters (in geometrized units):
///  M - mass-equivalent of the black hole
///  J - the angular momentum of the black hole
///  prograde - whether or not the orbit is prograde
fn isco_nospin_rotating(M: f64, J: f64, prograde: bool) -> f64 {
    // The normal equations go as follows:
    //  r_ms = GM/c^2 (3 + Z_2 ± sqrt[(3 - Z_1)(3 + Z_1 + 2Z_2)])
    //  Z_2 = sqrt[3chi^2 + Z_1^2]
    //  Z_1 = 1 + cbrt[1 - chi^2] (cbrt[1 + chi] + cbrt[1 - chi])
    //  chi = 2a/r_s = cJ / M^2G
    // where the sign of the ± is - if prograde, else + if retrograde
    // However, we use nondimensionalized form, so M=1, G=1, c=1, so chi=a

    assert!(
        M.is_normal() && M > 0.,
        "ISCO of non-spinning particle orbiting rotating body requires normal, positive mass"
    );

    let chi = J / M.powi(2);
    // chi in [-1,1] so Z_1 in [1,3], Z_2 in [2,3]
    // (3-Z_1)(3+Z_1+2Z_2) is therefore nonnegative
    assert!(
        -1. <= chi && chi <= 1.,
        "ISCO rotation parameter ({chi}) must be in range [-1,1]"
    );

    let chi_p2 = chi.powi(2);
    let Z_1 = 1. + (1. - chi_p2).cbrt() * { (1. + chi).cbrt() + (1. - chi).cbrt() }; // 
    let Z_2 = (3. * chi_p2 + Z_1.powi(2)).sqrt(); // always positive and defined
    let tail = ((3. - Z_1) * (3. + Z_1 + 2. * Z_2)).sqrt();
    let r_ms = M * (3. + Z_2 + if prograde { -tail } else { tail });

    r_ms
}

/// Get the t-value of the intersection between a ray and a plane in Euclidean space.
///
/// Will return `None` if the ray is parallel to the plane, or if the plane is "behind" the ray
/// origin. If `Some` is returned, the t-value will be greater than or equal to zero. In
/// particular, if the ray origin is inside the plane, a t-value of zero will be returned.
fn euclidean_isect_ray_plane(r_0: DVec3, r_d: DVec3, x_0: DVec3, n: DVec3) -> Option<f64> {
    // find t so that (r_0 + t r_d - x_0) • n = 0
    //   r_0 • n + t r_d • n - x_0 • n = 0
    // thus t = (x_0 - r_0 ) • n / (r_d • n)
    // condition: r_d • n is nonzero and finite and not NaN
    let v = r_d.dot(n);
    if v.is_normal() {
        let t = (x_0 - r_0).dot(n) * v.recip();
        if t >= 0. { Some(t) } else { None }
    } else {
        None
    }
}
