use anyhow::{Result, ensure};

fn validate(bytes: &[u8], particles: usize, streams: usize) -> Result<()> {
    ensure!(
        particles > 0 && matches!(streams, 2 | 3),
        "Invalid GRU history dimensions"
    );
    let expected = particles
        .checked_mul(streams)
        .and_then(|n| n.checked_mul(16));
    ensure!(
        expected == Some(bytes.len()),
        "GRU history byte length does not match particle/stream counts"
    );
    Ok(())
}

pub fn planar_gru_history(bytes: &[u8], particles: usize, streams: usize) -> Result<Vec<u8>> {
    validate(bytes, particles, streams)?;
    let mut native = vec![0; bytes.len()];
    for particle in 0..particles {
        for stream in 0..streams {
            let original = (particle * streams + stream) * 16;
            let planar = (stream * particles + particle) * 16;
            native[planar..planar + 16].copy_from_slice(&bytes[original..original + 16]);
        }
    }
    Ok(native)
}

pub fn interleave_gru_history(bytes: &[u8], particles: usize, streams: usize) -> Result<Vec<u8>> {
    validate(bytes, particles, streams)?;
    let mut canonical = vec![0; bytes.len()];
    for particle in 0..particles {
        for stream in 0..streams {
            let original = (particle * streams + stream) * 16;
            let planar = (stream * particles + particle) * 16;
            canonical[original..original + 16].copy_from_slice(&bytes[planar..planar + 16]);
        }
    }
    Ok(canonical)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gru_history_preserves_tagged_vec4_bytes_and_signed_zero() {
        for (streams, expected_order) in [
            (2, &[0u32, 2, 4, 1, 3, 5][..]),
            (3, &[0u32, 3, 6, 1, 4, 7, 2, 5, 8][..]),
        ] {
            let tagged = |tag: u32| [tag, 0x8000_0000, 0, tag ^ 0x1234_5678];
            let original: Vec<u8> = (0..3 * streams as u32)
                .flat_map(|tag| tagged(tag).into_iter().flat_map(u32::to_le_bytes))
                .collect();
            let expected: Vec<u8> = expected_order
                .iter()
                .flat_map(|tag| tagged(*tag).into_iter().flat_map(u32::to_le_bytes))
                .collect();
            assert_eq!(planar_gru_history(&original, 3, streams).unwrap(), expected);
            assert_eq!(
                interleave_gru_history(&expected, 3, streams).unwrap(),
                original
            );
            assert!(planar_gru_history(&original[..original.len() - 1], 3, streams).is_err());
            assert!(interleave_gru_history(&expected, 4, streams).is_err());
        }
        assert!(planar_gru_history(&[], 0, 2).is_err());
        assert!(interleave_gru_history(&[0; 16], 1, 1).is_err());
        assert!(planar_gru_history(&[], usize::MAX, 3).is_err());
    }
}
