plot image *args:
  #! /usr/bin/env bash
  set -e
  image="$(realpath "{{image}}")"
  cd '{{justfile_directory()}}/examples/plot'
  mkdir -p data
  data='data/{{file_stem(image)}}.dat'
  cargo run --release --example plot --all-features -- "$image" {{args}} > "$data"
  gnuplot -e "file='$data'" plot.gnuplot
