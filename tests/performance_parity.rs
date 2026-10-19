use primordial_soup::performance::{Band, Meter};
use serde::Deserialize;

#[derive(Deserialize)]
struct Frame {
    now: f64,
    ms: f64,
    adjusting: bool,
    active: bool,
    band: String,
    readout: u32,
}
#[derive(Deserialize)]
struct Fixture {
    frames: Vec<Frame>,
}

#[test]
fn frame_windows_and_color_debounce_match_recorded_reference() {
    let fixture: Fixture =
        serde_json::from_str(include_str!("fixtures/performance.fixture.json")).unwrap();
    let mut meter = Meter::default();
    for (n, frame) in fixture.frames.iter().enumerate() {
        meter.sample(frame.now, frame.ms);
        if frame.active {
            meter.update(frame.now, frame.adjusting);
        }
        let band = match meter.band {
            Band::Red => "red",
            Band::Yellow => "yellow",
            Band::Green => "green",
        };
        assert_eq!(band, frame.band, "color at source frame {n}");
        assert_eq!(meter.readout, frame.readout, "readout at source frame {n}");
    }
}
