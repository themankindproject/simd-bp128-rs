//! Bit width calculation utilities.

static BIT_WIDTH_LOOKUP: [u8; 256] = [
    0, 1, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, // 0-15
    5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, // 16-31
    6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, // 32-47
    6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, // 48-63
    7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, // 64-79
    7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, // 80-95
    7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, // 96-111
    7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, // 112-127
    8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, // 128-143
    8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, // 144-159
    8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, // 160-175
    8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, // 176-191
    8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, // 192-207
    8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, // 208-223
    8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, // 224-239
    8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, // 240-255
];

/// Returns the minimum number of bits required to represent a value.
#[inline]
pub(crate) fn required_bit_width(max_value: u32) -> u8 {
    if max_value <= 255 {
        BIT_WIDTH_LOOKUP[max_value as usize]
    } else {
        32 - max_value.leading_zeros() as u8
    }
}

/// Returns the number of bytes needed to store a full 128-value block.
#[inline]
pub(crate) fn packed_block_size(bit_width: u8) -> usize {
    assert!(bit_width <= 32, "bit_width must be 0-32, got {bit_width}");
    if bit_width == 0 {
        0
    } else {
        ((128 * bit_width as usize) + 7) / 8
    }
}

/// Returns the number of bytes needed to store a partial block.
#[inline]
pub(crate) fn packed_partial_block_size(num_values: usize, bit_width: u8) -> usize {
    if bit_width == 0 || num_values == 0 {
        0
    } else {
        ((num_values * bit_width as usize) + 7) / 8
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_required_bit_width() {
        assert_eq!(required_bit_width(0), 0);
        assert_eq!(required_bit_width(1), 1);
        assert_eq!(required_bit_width(2), 2);
        assert_eq!(required_bit_width(3), 2);
        assert_eq!(required_bit_width(4), 3);
        assert_eq!(required_bit_width(7), 3);
        assert_eq!(required_bit_width(8), 4);
        assert_eq!(required_bit_width(15), 4);
        assert_eq!(required_bit_width(16), 5);
        assert_eq!(required_bit_width(255), 8);
        assert_eq!(required_bit_width(256), 9);
        assert_eq!(required_bit_width(65535), 16);
        assert_eq!(required_bit_width(65536), 17);
        assert_eq!(required_bit_width(u32::MAX), 32);
    }

    #[test]
    fn test_packed_block_size() {
        assert_eq!(packed_block_size(0), 0);
        assert_eq!(packed_block_size(1), 16);
        assert_eq!(packed_block_size(2), 32);
        assert_eq!(packed_block_size(4), 64);
        assert_eq!(packed_block_size(8), 128);
        assert_eq!(packed_block_size(16), 256);
        assert_eq!(packed_block_size(32), 512);
    }

    #[test]
    #[should_panic(expected = "bit_width must be 0-32")]
    fn test_packed_block_size_invalid_bit_width() {
        let _ = packed_block_size(33);
    }
}
