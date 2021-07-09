#![no_std]
#![allow(incomplete_features)]
#![feature(repr_simd, platform_intrinsics, simd_ffi, const_generics, stdsimd)]
#![warn(missing_docs)]
//! Portable SIMD module.

#[macro_use]
mod first;
#[macro_use]
mod permute;
#[macro_use]
mod reduction;

mod select;
pub use select::Select;

mod to_bytes;
pub use to_bytes::ToBytes;

mod comparisons;
mod fmt;
mod intrinsics;
mod iter;
mod ops;
mod round;
mod vendor;

mod math;

mod lanes_at_most_32;
pub use lanes_at_most_32::LanesAtMost32;

mod masks;
pub use masks::*;

mod vector;
pub use vector::*;

mod array;
pub use array::SimdArray;
