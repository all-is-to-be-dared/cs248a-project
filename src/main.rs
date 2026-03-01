use std::io::Write;

use image::{Rgb, Rgb32FImage, RgbImage};
use ultraviolet::{DVec2, DVec3, UVec2, Vec2};

fn main() {
    let image_buffer = sample_image(
        Params {
            image_size: UVec2::new(1024, 1024),
        },
        Render::AccretionDisk(RenderAccretionDisk {
            bh_mass_equivalent: 1.,
            camera_dist_of_schwarzschild: 6.,
            focal_length: 0.00000001,
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
    // Mass-equivalent of the central black hole in natural units
    bh_mass_equivalent: f64,
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
    let r_s = 2. * disk.bh_mass_equivalent;
    // TODO: do this properly
    let r_outer = 10. * r_s;
    // we can now calculate the size of the screen in worldspace given the focal length
    //  sz_wo : f  = r_outer : (f + d_c)
    let d_s = disk.camera_dist_of_schwarzschild * r_s;
    let wo_size = 2. * r_outer * disk.focal_length * (disk.focal_length + d_s);
    for iy in 0..params.image_size.y {
        for ix in 0..params.image_size.x {
            // screenspace xy in [0,1)^2
            let px_low = DVec2::new(ix as f64, iy as f64);
            let screen_xy0 = px_low / DVec2::from(params.image_size);
            let screen_xy1 = (px_low + DVec2::broadcast(1.)) / DVec2::from(params.image_size);
            let screen_xy0_5 = (screen_xy0 + screen_xy1) / 2.;
            let wo_xy0_5 = screen_xy0_5 * wo_size;
            let ro = DVec3::new(wo_xy0_5.x, wo_xy0_5.y, -d_s);
            let rd = -DVec3::unit_z();
        }
    }
}
