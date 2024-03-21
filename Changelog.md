# v0.2.0
- Add multi-threaded versions of the dither functions.
- Pixel deduplication through `UniqueColorCounts` and `IndexedColorCounts` should be slightly faster for small images.
- External crates that have types present in `quantette`'s public API are now reexported (`palette` and `image`).
- `PalettePipeline` and `ImagePipeline` now take `impl Into<PaletteSize>` instead of just `PaletteSize` for their `palette_size` functions.
- Similarly, the pipeline structs now take `impl Into<QuantizeMethod<_>>` instead of just `QuantizeMethod<_>` for their `quantize_method` functions.
- Bumped `image` version to `0.25.0`.
- Removed unused `wide` feature on `palette` dependency.

# v0.1.1
Fixed typos and reduced unnecessary dependencies.

# v0.1.0
First release!
