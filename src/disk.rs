pub mod plot {
    use std::f64::consts::PI;

    use super::*;

    use crate::FigureOpts;

    #[derive(Debug, clap::Args, serde::Serialize)]
    pub struct CurvesOpts {
        /// Length-scaled mass of the central BH in appropriate (geometrized) units.
        #[arg(long, short = 'M')]
        M: f64,
        /// (Dimensionless) mass accretion rate of the accretion disk, expressed as Ṁ=16L_Edd/c².
        #[arg(long)]
        dMdt: f64,
        /// Angular momentum, in appropriate (geometrized) units.
        #[arg(long, short = 'J')]
        J: f64,
    }

    pub fn render_curves(_figure_opts: FigureOpts, opts: CurvesOpts) {
        let r_G = opts.M; // r_G = GM/c^2
        let a = opts.J / opts.M;
        let a_star = a / opts.M;

        #[derive(serde::Serialize)]
        struct Out {
            opts: CurvesOpts,
            r_ph: f64,
            r_mb: f64,
            r_ms_pro: f64,
            r_ms_retro: f64,
            r_H: f64,
            erg_t: Vec<f64>,
            erg_r: Vec<f64>,
        }

        let r_ph = r_ph(r_G, a_star);
        let r_mb = r_mb(r_G, a_star);
        let r_ms_pro = r_ms(r_G, a_star, true);
        let r_ms_retro = r_ms(r_G, a_star, false);
        let r_H = r_H(r_G, a_star);

        let mut erg_t = vec![];
        let mut erg_r: Vec<f64> = vec![];
        let N = 1024;
        for i in 0..N {
            let t = i as f64 * PI / N as f64;
            let r = r_0(r_G, a_star, t);
            erg_t.push(t);
            erg_r.push(r);
        }

        let out_path =
            std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("render_curves.json");
        let out = std::fs::OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(out_path)
            .unwrap();
        serde_json::to_writer(
            out,
            &Out {
                opts,
                r_ph,
                r_mb,
                r_ms_pro,
                r_ms_retro,
                r_H,
                erg_t,
                erg_r,
            },
        )
        .unwrap();
    }
}

/// The circular photon radius $r_\text{ph}$ of a Kerr BH with gravitational radius r_G and
/// relative Kerr parameter a_star.
pub fn r_ph(r_G: f64, a_star: f64) -> f64 {
    let a_star_acos = a_star.acos();
    assert!(
        !a_star_acos.is_nan(),
        "relative Kerr parameter out of range"
    );
    2. * r_G * { 1. + (2. * a_star_acos / 3.).cos() }
}

/// Marginally bound radius of a Kerr BH with gravitational radius r_G and relative Kerr parameter
/// a_star.
pub fn r_mb(r_G: f64, a_star: f64) -> f64 {
    assert!(
        -1. <= a_star && a_star <= 1.,
        "relative Kerr parameter out of range"
    );
    2. * r_G * { 1. - a_star * 0.5 + (1. - a_star).sqrt() }
}

/// Marginally stable radius of a Kerr BH with gravitational radius r_G and relative Kerr parameter
/// a_star, for a prograde/retrograde orbit.
pub fn r_ms(r_G: f64, a_star: f64, prograde: bool) -> f64 {
    let c_dir = match prograde {
        true => -1.,
        false => 1.,
    };
    assert!(
        -1. <= a_star && a_star <= 1.,
        "relative Kerr parameter out of range"
    );
    let chi = a_star;
    let chi_p2 = chi.powi(2);

    let Z_1 = 1. + (1. - chi_p2).cbrt() * { (1. + chi).cbrt() + (1. - chi).cbrt() }; // 
    let Z_2 = (3. * chi_p2 + Z_1.powi(2)).sqrt(); // always positive and defined
    let tail = ((3. - Z_1) * (3. + Z_1 + 2. * Z_2)).sqrt();
    let r_ms = r_G * (3. + Z_2 + c_dir * tail);

    r_ms
}

/// Horizon radius of a Kerr BH with gravitational radius r_G and relative Kerr parameter a_star.
pub fn r_H(r_G: f64, a_star: f64) -> f64 {
    assert!(
        -1. <= a_star && a_star <= 1.,
        "relative Kerr parameter out of range"
    );
    // Validity: if a_star in [-1,1] then the term under the square root is nonnegative
    r_G * { 1. + (1. - a_star.powi(2)).sqrt() }
}

/// Ergosphere radius of a Kerr BH with gravitational radius r_G and relative Kerr parameter a_star
/// at polar angle θ (measured from +z).
pub fn r_0(r_G: f64, a_star: f64, θ: f64) -> f64 {
    assert!(
        -1. <= a_star && a_star <= 1.,
        "relative Kerr parameter out of range"
    );
    // Validity: if a_star in [-1,1] then the term under the square root is nonnegative
    r_G * { 1. + (1. - a_star.powi(2) * θ.cos().powi(2)).sqrt() }
}
