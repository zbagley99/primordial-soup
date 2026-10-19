use std::{
    collections::HashSet,
    fs,
    path::{Path, PathBuf},
};
fn collect(p: &Path, out: &mut Vec<PathBuf>) {
    for e in fs::read_dir(p).unwrap() {
        let p = e.unwrap().path();
        if p.is_dir() {
            collect(&p, out)
        } else if p.extension().is_some_and(|s| s == "wgsl") {
            out.push(p)
        }
    }
}
fn expand(p: &Path, files: &[PathBuf], seen: &mut HashSet<PathBuf>) -> String {
    if !seen.insert(p.to_owned()) {
        return String::new();
    }
    let raw = fs::read_to_string(p).unwrap();
    raw.lines()
        .map(|l| {
            if let Some(n) = l
                .trim()
                .strip_prefix("#include \"")
                .and_then(|s| s.strip_suffix('"'))
            {
                let target = files
                    .iter()
                    .find(|f| f.file_name().unwrap() == n)
                    .expect("missing shader include");
                expand(target, files, seen)
            } else {
                format!("{l}\n")
            }
        })
        .collect()
}
fn main() {
    println!("cargo:rerun-if-changed=shaders");
    let mut files = Vec::new();
    collect(Path::new("shaders"), &mut files);
    let out = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let mut dispatch = String::from("pub fn shader(name: &str) -> &'static str { match name {\n");
    for p in &files {
        let name = p.file_name().unwrap().to_str().unwrap();
        fs::write(out.join(name), expand(p, &files, &mut HashSet::new())).unwrap();
        dispatch.push_str(&format!(
            "{name:?} => include_str!(concat!(env!(\"OUT_DIR\"), \"/{name}\")),\n"
        ));
    }
    dispatch.push_str("_ => panic!(\"unknown shader {name}\"), }}\n");
    fs::write(out.join("shaders.rs"), dispatch).unwrap();
}
