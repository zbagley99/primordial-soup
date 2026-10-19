//! Regenerate only when deliberately versioning the coefficient contract.
use primordial_soup::network::{self, Bank, Mode};
fn main() {
    let mut fixtures = Vec::new();
    let revision = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "1".into())
        .parse()
        .unwrap();
    for mode in [Mode::Mlp, Mode::Gru] {
        for cohort in [0, 3] {
            let b = Bank {
                mode,
                revision,
                parent: None,
                seed: 0.25,
                scale: 0.1,
            };
            let generated = network::derive(&b, cohort);
            let mutated = network::derive(
                &Bank {
                    parent: Some(generated.clone()),
                    ..b
                },
                cohort,
            );
            fixtures.push(serde_json::json!({"mode":mode,"cohort":cohort,"generated_bits":generated.iter().map(|v|v.to_bits()).collect::<Vec<_>>(),"mutated_bits":mutated.iter().map(|v|v.to_bits()).collect::<Vec<_>>() }));
        }
    }
    println!("{}", serde_json::to_string_pretty(&fixtures).unwrap());
}
