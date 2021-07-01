//! Types and traits associated with masking lanes of vectors.
//! Types representing
#![allow(non_camel_case_types)]

#[cfg_attr(
    not(all(target_arch = "x86_64", target_feature = "avx512f")),
    path = "full_masks.rs"
)]
#[cfg_attr(
    all(target_arch = "x86_64", target_feature = "avx512f"),
    path = "bitmask.rs"
)]
mod mask_impl;

use crate::{LanesAtMost32, SimdI16, SimdI32, SimdI64, SimdI8, SimdIsize};

mod sealed {
    pub trait Sealed {}
}

/// Helper trait for mask types.
pub trait Mask: sealed::Sealed {
    /// The bitmask representation of a mask.
    type BitMask: Copy + Default + AsRef<[u8]> + AsMut<[u8]>;

    // TODO remove this when rustc intrinsics are more flexible
    #[doc(hidden)]
    type IntBitMask;
}

macro_rules! define_opaque_mask {
    {
        $(#[$attr:meta])*
        struct $name:ident<const $lanes:ident: usize>($inner_ty:ty);
        @bits $bits_ty:ident
    } => {
        $(#[$attr])*
        #[allow(non_camel_case_types)]
        pub struct $name<const LANES: usize>($inner_ty)
        where
            $bits_ty<LANES>: LanesAtMost32,
            Self: Mask;

        impl<const LANES: usize> sealed::Sealed for $name<LANES>
        where
            $bits_ty<LANES>: LanesAtMost32,
            Self: Mask,
        {}
        impl Mask for $name<1> {
            type BitMask = [u8; 1];
            type IntBitMask = u8;
        }
        impl Mask for $name<2> {
            type BitMask = [u8; 1];
            type IntBitMask = u8;
        }
        impl Mask for $name<4> {
            type BitMask = [u8; 1];
            type IntBitMask = u8;
        }
        impl Mask for $name<8> {
            type BitMask = [u8; 1];
            type IntBitMask = u8;
        }
        impl Mask for $name<16> {
            type BitMask = [u8; 2];
            type IntBitMask = u16;
        }
        impl Mask for $name<32> {
            type BitMask = [u8; 4];
            type IntBitMask = u32;
        }

        impl_opaque_mask_reductions! { $name, $bits_ty }

        impl<const LANES: usize> $name<LANES>
        where
            $bits_ty<LANES>: LanesAtMost32,
            Self: Mask,
        {
            /// Construct a mask by setting all lanes to the given value.
            pub fn splat(value: bool) -> Self {
                Self(<$inner_ty>::splat(value))
            }

            /// Converts an array to a SIMD vector.
            pub fn from_array(array: [bool; LANES]) -> Self {
                let mut vector = Self::splat(false);
                let mut i = 0;
                while i < $lanes {
                    vector.set(i, array[i]);
                    i += 1;
                }
                vector
            }

            /// Converts a SIMD vector to an array.
            pub fn to_array(self) -> [bool; LANES] {
                let mut array = [false; LANES];
                let mut i = 0;
                while i < $lanes {
                    array[i] = self.test(i);
                    i += 1;
                }
                array
            }

            /// Converts a vector of integers to a mask, where 0 represents `false` and -1
            /// represents `true`.
            ///
            /// # Safety
            /// All lanes must be either 0 or -1.
            #[inline]
            pub unsafe fn from_int_unchecked(value: $bits_ty<LANES>) -> Self {
                Self(<$inner_ty>::from_int_unchecked(value))
            }

            /// Converts a vector of integers to a mask, where 0 represents `false` and -1
            /// represents `true`.
            ///
            /// # Panics
            /// Panics if any lane is not 0 or -1.
            #[inline]
            pub fn from_int(value: $bits_ty<LANES>) -> Self {
                assert!(
                    (value.lanes_eq($bits_ty::splat(0)) | value.lanes_eq($bits_ty::splat(-1))).all(),
                    "all values must be either 0 or -1",
                );
                unsafe { Self::from_int_unchecked(value) }
            }

            /// Converts the mask to a vector of integers, where 0 represents `false` and -1
            /// represents `true`.
            #[inline]
            pub fn to_int(self) -> $bits_ty<LANES> {
                self.0.to_int()
            }

            /// Tests the value of the specified lane.
            ///
            /// # Safety
            /// `lane` must be less than `LANES`.
            #[inline]
            pub unsafe fn test_unchecked(&self, lane: usize) -> bool {
                self.0.test_unchecked(lane)
            }

            /// Tests the value of the specified lane.
            ///
            /// # Panics
            /// Panics if `lane` is greater than or equal to the number of lanes in the vector.
            #[inline]
            pub fn test(&self, lane: usize) -> bool {
                assert!(lane < LANES, "lane index out of range");
                unsafe { self.test_unchecked(lane) }
            }

            /// Sets the value of the specified lane.
            ///
            /// # Safety
            /// `lane` must be less than `LANES`.
            #[inline]
            pub unsafe fn set_unchecked(&mut self, lane: usize, value: bool) {
                self.0.set_unchecked(lane, value);
            }

            /// Sets the value of the specified lane.
            ///
            /// # Panics
            /// Panics if `lane` is greater than or equal to the number of lanes in the vector.
            #[inline]
            pub fn set(&mut self, lane: usize, value: bool) {
                assert!(lane < LANES, "lane index out of range");
                unsafe { self.set_unchecked(lane, value); }
            }

            /// Convert this mask to a bitmask, with one bit set per lane.
            pub fn to_bitmask(self) -> <Self as Mask>::BitMask {
                self.0.to_bitmask::<Self>()
            }

            /// Convert a bitmask to a mask.
            pub fn from_bitmask(bitmask: <Self as Mask>::BitMask) -> Self {
                Self(<$inner_ty>::from_bitmask::<Self>(bitmask))
            }
        }

        // vector/array conversion
        impl<const LANES: usize> From<[bool; LANES]> for $name<LANES>
        where
            $bits_ty<LANES>: crate::LanesAtMost32,
            Self: Mask,
        {
            fn from(array: [bool; LANES]) -> Self {
                Self::from_array(array)
            }
        }

        impl <const LANES: usize> From<$name<LANES>> for [bool; LANES]
        where
            $bits_ty<LANES>: crate::LanesAtMost32,
            $name<LANES>: Mask,
        {
            fn from(vector: $name<LANES>) -> Self {
                vector.to_array()
            }
        }

        impl<const LANES: usize> Copy for $name<LANES>
        where
            $bits_ty<LANES>: LanesAtMost32,
            Self: Mask,
        {}

        impl<const LANES: usize> Clone for $name<LANES>
        where
            $bits_ty<LANES>: LanesAtMost32,
            Self: Mask,
        {
            #[inline]
            fn clone(&self) -> Self {
                *self
            }
        }

        impl<const LANES: usize> Default for $name<LANES>
        where
            $bits_ty<LANES>: LanesAtMost32,
            Self: Mask,
        {
            #[inline]
            fn default() -> Self {
                Self::splat(false)
            }
        }

        impl<const LANES: usize> PartialEq for $name<LANES>
        where
            $bits_ty<LANES>: LanesAtMost32,
            Self: Mask,
        {
            #[inline]
            fn eq(&self, other: &Self) -> bool {
                self.0 == other.0
            }
        }

        impl<const LANES: usize> PartialOrd for $name<LANES>
        where
            $bits_ty<LANES>: LanesAtMost32,
            Self: Mask,
        {
            #[inline]
            fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
                self.0.partial_cmp(&other.0)
            }
        }

        impl<const LANES: usize> core::fmt::Debug for $name<LANES>
        where
            $bits_ty<LANES>: crate::LanesAtMost32,
            Self: Mask,
        {
            fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
                f.debug_list()
                    .entries((0..LANES).map(|lane| self.test(lane)))
                    .finish()
            }
        }

        impl<const LANES: usize> core::ops::BitAnd for $name<LANES>
        where
            $bits_ty<LANES>: LanesAtMost32,
            Self: Mask,
        {
            type Output = Self;
            #[inline]
            fn bitand(self, rhs: Self) -> Self {
                Self(self.0 & rhs.0)
            }
        }

        impl<const LANES: usize> core::ops::BitAnd<bool> for $name<LANES>
        where
            $bits_ty<LANES>: LanesAtMost32,
            Self: Mask,
        {
            type Output = Self;
            #[inline]
            fn bitand(self, rhs: bool) -> Self {
                self & Self::splat(rhs)
            }
        }

        impl<const LANES: usize> core::ops::BitAnd<$name<LANES>> for bool
        where
            $bits_ty<LANES>: LanesAtMost32,
            $name<LANES>: Mask,
        {
            type Output = $name<LANES>;
            #[inline]
            fn bitand(self, rhs: $name<LANES>) -> $name<LANES> {
                $name::<LANES>::splat(self) & rhs
            }
        }

        impl<const LANES: usize> core::ops::BitOr for $name<LANES>
        where
            $bits_ty<LANES>: LanesAtMost32,
            Self: Mask,
        {
            type Output = Self;
            #[inline]
            fn bitor(self, rhs: Self) -> Self {
                Self(self.0 | rhs.0)
            }
        }

        impl<const LANES: usize> core::ops::BitOr<bool> for $name<LANES>
        where
            $bits_ty<LANES>: LanesAtMost32,
            Self: Mask,
        {
            type Output = Self;
            #[inline]
            fn bitor(self, rhs: bool) -> Self {
                self | Self::splat(rhs)
            }
        }

        impl<const LANES: usize> core::ops::BitOr<$name<LANES>> for bool
        where
            $bits_ty<LANES>: LanesAtMost32,
            $name<LANES>: Mask,
        {
            type Output = $name<LANES>;
            #[inline]
            fn bitor(self, rhs: $name<LANES>) -> $name<LANES> {
                $name::<LANES>::splat(self) | rhs
            }
        }

        impl<const LANES: usize> core::ops::BitXor for $name<LANES>
        where
            $bits_ty<LANES>: LanesAtMost32,
            Self: Mask,
        {
            type Output = Self;
            #[inline]
            fn bitxor(self, rhs: Self) -> Self::Output {
                Self(self.0 ^ rhs.0)
            }
        }

        impl<const LANES: usize> core::ops::BitXor<bool> for $name<LANES>
        where
            $bits_ty<LANES>: LanesAtMost32,
            Self: Mask,
        {
            type Output = Self;
            #[inline]
            fn bitxor(self, rhs: bool) -> Self::Output {
                self ^ Self::splat(rhs)
            }
        }

        impl<const LANES: usize> core::ops::BitXor<$name<LANES>> for bool
        where
            $bits_ty<LANES>: LanesAtMost32,
            $name<LANES>: Mask,
        {
            type Output = $name<LANES>;
            #[inline]
            fn bitxor(self, rhs: $name<LANES>) -> Self::Output {
                $name::<LANES>::splat(self) ^ rhs
            }
        }

        impl<const LANES: usize> core::ops::Not for $name<LANES>
        where
            $bits_ty<LANES>: LanesAtMost32,
            Self: Mask,
        {
            type Output = $name<LANES>;
            #[inline]
            fn not(self) -> Self::Output {
                Self(!self.0)
            }
        }

        impl<const LANES: usize> core::ops::BitAndAssign for $name<LANES>
        where
            $bits_ty<LANES>: LanesAtMost32,
            Self: Mask,
        {
            #[inline]
            fn bitand_assign(&mut self, rhs: Self) {
                self.0 = self.0 & rhs.0;
            }
        }

        impl<const LANES: usize> core::ops::BitAndAssign<bool> for $name<LANES>
        where
            $bits_ty<LANES>: LanesAtMost32,
            Self: Mask,
        {
            #[inline]
            fn bitand_assign(&mut self, rhs: bool) {
                *self &= Self::splat(rhs);
            }
        }

        impl<const LANES: usize> core::ops::BitOrAssign for $name<LANES>
        where
            $bits_ty<LANES>: LanesAtMost32,
            Self: Mask,
        {
            #[inline]
            fn bitor_assign(&mut self, rhs: Self) {
                self.0 = self.0 | rhs.0;
            }
        }

        impl<const LANES: usize> core::ops::BitOrAssign<bool> for $name<LANES>
        where
            $bits_ty<LANES>: LanesAtMost32,
            Self: Mask,
        {
            #[inline]
            fn bitor_assign(&mut self, rhs: bool) {
                *self |= Self::splat(rhs);
            }
        }

        impl<const LANES: usize> core::ops::BitXorAssign for $name<LANES>
        where
            $bits_ty<LANES>: LanesAtMost32,
            Self: Mask,
        {
            #[inline]
            fn bitxor_assign(&mut self, rhs: Self) {
                self.0 = self.0 ^ rhs.0;
            }
        }

        impl<const LANES: usize> core::ops::BitXorAssign<bool> for $name<LANES>
        where
            $bits_ty<LANES>: LanesAtMost32,
            Self: Mask,
        {
            #[inline]
            fn bitxor_assign(&mut self, rhs: bool) {
                *self ^= Self::splat(rhs);
            }
        }
    };
}

define_opaque_mask! {
    /// Mask for vectors with `LANES` 8-bit elements.
    ///
    /// The layout of this type is unspecified.
    struct Mask8<const LANES: usize>(mask_impl::Mask8<Self, LANES>);
    @bits SimdI8
}

define_opaque_mask! {
    /// Mask for vectors with `LANES` 16-bit elements.
    ///
    /// The layout of this type is unspecified.
    struct Mask16<const LANES: usize>(mask_impl::Mask16<Self, LANES>);
    @bits SimdI16
}

define_opaque_mask! {
    /// Mask for vectors with `LANES` 32-bit elements.
    ///
    /// The layout of this type is unspecified.
    struct Mask32<const LANES: usize>(mask_impl::Mask32<Self, LANES>);
    @bits SimdI32
}

define_opaque_mask! {
    /// Mask for vectors with `LANES` 64-bit elements.
    ///
    /// The layout of this type is unspecified.
    struct Mask64<const LANES: usize>(mask_impl::Mask64<Self, LANES>);
    @bits SimdI64
}

define_opaque_mask! {
    /// Mask for vectors with `LANES` pointer-width elements.
    ///
    /// The layout of this type is unspecified.
    struct MaskSize<const LANES: usize>(mask_impl::MaskSize<Self, LANES>);
    @bits SimdIsize
}

/// Vector of eight 8-bit masks
pub type mask8x8 = Mask8<8>;

/// Vector of 16 8-bit masks
pub type mask8x16 = Mask8<16>;

/// Vector of 32 8-bit masks
pub type mask8x32 = Mask8<32>;

/// Vector of 16 8-bit masks
pub type mask8x64 = Mask8<64>;

/// Vector of four 16-bit masks
pub type mask16x4 = Mask16<4>;

/// Vector of eight 16-bit masks
pub type mask16x8 = Mask16<8>;

/// Vector of 16 16-bit masks
pub type mask16x16 = Mask16<16>;

/// Vector of 32 16-bit masks
pub type mask16x32 = Mask32<32>;

/// Vector of two 32-bit masks
pub type mask32x2 = Mask32<2>;

/// Vector of four 32-bit masks
pub type mask32x4 = Mask32<4>;

/// Vector of eight 32-bit masks
pub type mask32x8 = Mask32<8>;

/// Vector of 16 32-bit masks
pub type mask32x16 = Mask32<16>;

/// Vector of two 64-bit masks
pub type mask64x2 = Mask64<2>;

/// Vector of four 64-bit masks
pub type mask64x4 = Mask64<4>;

/// Vector of eight 64-bit masks
pub type mask64x8 = Mask64<8>;

/// Vector of two pointer-width masks
pub type masksizex2 = MaskSize<2>;

/// Vector of four pointer-width masks
pub type masksizex4 = MaskSize<4>;

/// Vector of eight pointer-width masks
pub type masksizex8 = MaskSize<8>;

macro_rules! impl_from {
    { $from:ident ($from_inner:ident) => $($to:ident ($to_inner:ident)),* } => {
        $(
        impl<const LANES: usize> From<$from<LANES>> for $to<LANES>
        where
            crate::$from_inner<LANES>: crate::LanesAtMost32,
            crate::$to_inner<LANES>: crate::LanesAtMost32,
            $from<LANES>: Mask,
            Self: Mask,
        {
            fn from(value: $from<LANES>) -> Self {
                Self(value.0.into())
            }
        }
        )*
    }
}
impl_from! { Mask8 (SimdI8) => Mask16 (SimdI16), Mask32 (SimdI32), Mask64 (SimdI64), MaskSize (SimdIsize) }
impl_from! { Mask16 (SimdI16) => Mask32 (SimdI32), Mask64 (SimdI64), MaskSize (SimdIsize), Mask8 (SimdI8) }
impl_from! { Mask32 (SimdI32) => Mask64 (SimdI64), MaskSize (SimdIsize), Mask8 (SimdI8), Mask16 (SimdI16) }
impl_from! { Mask64 (SimdI64) => MaskSize (SimdIsize), Mask8 (SimdI8), Mask16 (SimdI16), Mask32 (SimdI32) }
impl_from! { MaskSize (SimdIsize) => Mask8 (SimdI8), Mask16 (SimdI16), Mask32 (SimdI32), Mask64 (SimdI64) }
