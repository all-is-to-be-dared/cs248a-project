#import "@preview/lilaq:0.5.0" as lq
#import "@preview/funarray:0.4.0"

#let polar2cart = (theta, r) => {
  (r * calc.cos(calc.pi/2 - theta), r * calc.sin(calc.pi/2 - theta))
}
#let array_polar2cart = (a) => {
  funarray.unzip(a.map(((theta,r)) => polar2cart(theta, r)))
}

= Structure

#let (opts, r_ph, r_mb, r_ms_pro, r_ms_retro, r_H, erg_t, erg_r) = json("render_curves.json")

Gravitational structure of black hole with parameters:
$ M=#opts.M quad quad dot(M) = #opts.dMdt quad quad J = #opts.J $

#lq.diagram(
  width: 10cm,
  height: 10cm,
  lq.vlines(r_ph, stroke: teal, label: [$r_"ph"$]),
  lq.vlines(r_mb, stroke: blue, label: [$r_"mb"$]),
  lq.vlines(r_ms_pro, stroke: green, label: [$r_"ms" "(Prograde)"$]),
  lq.vlines(r_ms_retro, stroke: (paint: green, dash: "dashed"), label: [$r_"ms" "(Retrograde)"$]),
  lq.plot(
    ..array_polar2cart(range(0, 1024).map(i => (calc.pi * i / 1024., r_H) )),
    stroke: red,
    mark: none,
  ),
  lq.vlines(r_H, stroke: red, label: [$r_H$]),
  lq.plot(
    ..array_polar2cart(erg_t.zip(erg_r)),
    stroke: black,
    mark: none,
    label: [$r_"erg"$]
  ),
  xlabel: [$x$ (geometrized)],
  ylabel: [$y$ (geometrized)]
)
