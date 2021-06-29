#![no_std]
#![allow(incomplete_features)]
#![feature(repr_simd, platform_intrinsics, simd_ffi, const_generics)]
#![warn(missing_docs)]
//! Portable SIMD module.

#[macro_use]
mod permute;
#[macro_use]
mod transmute;
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

mod math;

mod masks;
pub use masks::*;

mod vector;
pub use vector::*;
