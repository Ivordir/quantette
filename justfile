check:
  cargo fmt --check
  typos
  cargo doc --no-deps
  cargo hack --feature-powerset clippy

test:
  cargo test --doc
  cargo test --lib

test-hack:
  cargo test --doc
  cargo hack --feature-powerset test --lib

plot image *args:
  #! /usr/bin/env bash
  set -e
  image="$(realpath "{{image}}")"
  cd '{{justfile_directory()}}/examples/plot'
  mkdir -p data
  data='data/{{file_stem(image)}}.dat'
  cargo run --release --example plot -- "$image" {{args}} > "$data"
  gnuplot -e "file='$data'" plot.gnuplot
