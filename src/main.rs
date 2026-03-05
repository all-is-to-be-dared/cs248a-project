#![allow(non_snake_case, mixed_script_confusables)]

use clap::Parser as _;

mod disk;

#[derive(Debug, clap::Parser)]
struct Opts {
    #[clap(flatten)]
    figure_opts: FigureOpts,
    #[clap(subcommand)]
    figure: Figure,
}
#[derive(Debug, clap::Args)]
struct FigureOpts {
    //
}
#[derive(Debug, clap::Subcommand)]
enum Figure {
    DiskCurves(disk::plot::CurvesOpts),
}

fn main() {
    let opts = Opts::parse();
    match opts.figure {
        Figure::DiskCurves(curves_opts) => {
            disk::plot::render_curves(opts.figure_opts, curves_opts);
        }
    }

    // let image_buffer = sample_image(Params {
    //     image_size: UVec2::new(1024, 1024),
    // });
    // let path = "output.png";
    // let mut output = std::fs::OpenOptions::new()
    //     .write(true)
    //     .create(true)
    //     .open(path)
    //     .unwrap();
    // image_to_png(image_buffer, &mut output);
}

// fn image_to_png(image: RgbImage, out: &mut impl Write) {
//     let encoder = image::codecs::png::PngEncoder::new_with_quality(
//         out,
//         image::codecs::png::CompressionType::Uncompressed,
//         image::codecs::png::FilterType::Adaptive,
//     );
//     let () = image.write_with_encoder(encoder).unwrap();
// }

// struct Params {
//     image_size: UVec2,
// }
// fn sample_image(params: Params, render: Render) -> RgbImage {
//     let mut img = Rgb32FImage::new(params.image_size.x, params.image_size.y);
//     match render {
//         Render::Gradient => render_gradient(params, &mut img),
//         Render::AccretionDisk(disk) => render_accretion_disk(disk, params, &mut img),
//     }
//     image::DynamicImage::ImageRgb32F(img).into_rgb8()
// }
// fn render_gradient(params: Params, img: &mut Rgb32FImage) {
//     for y in 0..params.image_size.y {
//         for x in 0..params.image_size.x {
//             let x_rel = x as f32 / (params.image_size.x - 1) as f32;
//             let y_rel = y as f32 / (params.image_size.y - 1) as f32;
//             img.put_pixel(x, y, Rgb([x_rel, y_rel, 0.]));
//         }
//     }
// }
