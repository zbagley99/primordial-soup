use primordial_soup::gpu::shader;

#[test]
fn every_pipeline_parses_and_validates_as_wgsl() {
    for name in [
        "cellsDensity.wgsl",
        "cellsSprites.wgsl",
        "cellsMembranes.wgsl",
        "cellsPlan.wgsl",
        "cellsTiled.wgsl",
        "cellsAverage.wgsl",
        "cellsBlur.wgsl",
        "cellsBlurFirst.wgsl",
        "cellsPost.wgsl",
        "entityUpdate.wgsl",
        "networkProbe.wgsl",
        "networkProbe3.wgsl",
        "cohortRules.wgsl",
        "entityPick.wgsl",
        "deriveRule.wgsl",
        "canvas.wgsl",
        "strafeDraw.wgsl",
        "camera.wgsl",
        "camBrush.wgsl",
        "accumulate.wgsl",
        "bloomDownsample.wgsl",
        "bloomUpsample.wgsl",
        "frameAssembly.wgsl",
        "cameraIngest.wgsl",
        "cameraBlur.wgsl",
        "cameraMap.wgsl",
        "cameraPreview.wgsl",
    ] {
        let source = shader(name);
        let module = naga::front::wgsl::parse_str(source)
            .unwrap_or_else(|e| panic!("{name}: {}", e.emit_to_string(source)));
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap_or_else(|e| panic!("{name}: {e:#?}"));
    }
}

#[test]
fn gpu_struct_offsets_match_the_frozen_contract() {
    let module = naga::front::wgsl::parse_str(shader("common.wgsl")).unwrap();
    for (name, span, offsets) in [
        ("ConfigData", 416, &[0, 320, 336, 352, 368, 384, 400][..]),
        ("WorldData", 32, &[0, 16][..]),
        ("Entity", 32, &[0, 16][..]),
    ] {
        let (_, ty) = module
            .types
            .iter()
            .find(|(_, t)| t.name.as_deref() == Some(name))
            .unwrap();
        match &ty.inner {
            naga::TypeInner::Struct {
                members,
                span: actual,
            } => {
                assert_eq!(*actual, span, "{name} stride");
                assert_eq!(
                    members.iter().map(|m| m.offset).collect::<Vec<_>>(),
                    offsets,
                    "{name} offsets"
                );
            }
            _ => panic!("{name} is not a struct"),
        }
    }
}
