macro_rules! impl_integer_reductions {
    { $name:ident, $scalar:ty } => {
        impl<const LANES: usize> crate::$name<LANES>
        where
            Self: crate::LanesAtMost32
        {
            /// Horizontal wrapping add.  Returns the sum of the lanes of the vector, with wrapping addition.
            /// ```
            /// # use core_simd::*;
            /// let a = u32x4::from_array([1, 2, 3, 4]).horizontal_sum();
            /// assert_eq!(10, a);
            /// let b = u32x4::from_array([u32::MAX, 1, 0, 0]).horizontal_sum();
            /// assert_eq!(0, b);
            /// ```
            #[inline]
            pub fn horizontal_sum(self) -> $scalar {
                unsafe { crate::intrinsics::simd_reduce_add_ordered(self, 0) }
            }

            /// Horizontal wrapping multiply.  Returns the product of the lanes of the vector, with wrapping multiplication.
            /// ```
            /// # use core_simd::*;
            /// let a = u32x4::from_array([2, 2, 2, 2]).horizontal_product();
            /// assert_eq!(16, a);
            /// let b = u32x4::from_array([u32::MAX, 2, 1, 1]).horizontal_product();
            /// assert_eq!(u32::MAX, b + 1); // Because 2*u32::MAX wraps, and is 1 off from the u32::MAX
            /// ```
            #[inline]
            pub fn horizontal_product(self) -> $scalar {
                unsafe { crate::intrinsics::simd_reduce_mul_ordered(self, 1) }
            }

            /// Horizontal bitwise "and".  Returns the cumulative bitwise "and" across the lanes of
            /// the vector.
            /// ```
            /// # use core_simd::*;
            /// let a = u32x4::from_array([3, 3, 3, 3]).horizontal_and();
            /// assert_eq!(3, a);
            /// let b = u32x4::from_array([1, 1, 0, 0]).horizontal_and();
            /// assert_eq!(0, b);
            /// ```
            #[inline]
            pub fn horizontal_and(self) -> $scalar {
                unsafe { crate::intrinsics::simd_reduce_and(self) }
            }

            /// Horizontal bitwise "or".  Returns the cumulative bitwise "or" across the lanes of
            /// the vector.
            /// ```
            /// # use core_simd::*;
            /// let a = u32x4::from_array([1, 2, 0, 0]).horizontal_or();
            /// assert_eq!(3, a);
            /// ```
            #[inline]
            pub fn horizontal_or(self) -> $scalar {
                unsafe { crate::intrinsics::simd_reduce_or(self) }
            }

            /// Horizontal bitwise "xor".  Returns the cumulative bitwise "xor" across the lanes of
            /// the vector.
            /// ```
            /// # use core_simd::*;
            /// let a = u32x4::from_array([5, 5, 5, 0]).horizontal_xor();
            /// assert_eq!(5, a);
            /// ```
            #[inline]
            pub fn horizontal_xor(self) -> $scalar {
                unsafe { crate::intrinsics::simd_reduce_xor(self) }
            }

            /// Horizontal maximum.  Returns the maximum lane in the vector.
            /// ```
            /// # use core_simd::*;
            /// let a = u32x4::from_array([1, 2, 42, 0]).horizontal_max();
            /// assert_eq!(42, a);
            /// ```
            #[inline]
            pub fn horizontal_max(self) -> $scalar {
                unsafe { crate::intrinsics::simd_reduce_max(self) }
            }

            /// Horizontal minimum.  Returns the minimum lane in the vector.
            /// ```
            /// # use core_simd::*;
            /// let a = u32x4::from_array([1, 2, 42, 0]).horizontal_min();
            /// assert_eq!(0, a);
            /// ```
            #[inline]
            pub fn horizontal_min(self) -> $scalar {
                unsafe { crate::intrinsics::simd_reduce_min(self) }
            }
        }
    }
}

macro_rules! impl_float_reductions {
    { $name:ident, $scalar:ty } => {
        impl<const LANES: usize> crate::$name<LANES>
        where
            Self: crate::LanesAtMost32
        {

            /// Horizontal add.  Returns the sum of the lanes of the vector, with saturating addition.
            /// ```
            /// # use core_simd::*;
            /// let a = f32x4::from_array([1.0, 2.0, 3.0, 4.0]).horizontal_sum();
            /// assert_eq!(10.0, a);
            /// let b = f32x4::from_array([f32::MAX, 2.0, 0.0, 0.0]).horizontal_sum();
            /// assert_eq!(f32::MAX, b);
            /// ```
            #[inline]
            pub fn horizontal_sum(self) -> $scalar {
                // LLVM sum is inaccurate on i586
                if cfg!(all(target_arch = "x86", not(target_feature = "sse2"))) {
                    self.as_slice().iter().sum()
                } else {
                    unsafe { crate::intrinsics::simd_reduce_add_ordered(self, 0.) }
                }
            }

            /// Horizontal multiply.  Returns the product of the lanes of the vector.
            /// ```
            /// # use core_simd::*;
            /// let a = f32x4::from_array([1.0, 2.0, 3.0, 4.0]).horizontal_product();
            /// assert_eq!(24.0, a);
            /// let b = f32x4::from_array([f32::MAX, 2.0, 1.0, 1.0]).horizontal_product();
            /// assert_eq!(f32::MAX, b);
            /// ```
            #[inline]
            pub fn horizontal_product(self) -> $scalar {
                // LLVM product is inaccurate on i586
                if cfg!(all(target_arch = "x86", not(target_feature = "sse2"))) {
                    self.as_slice().iter().product()
                } else {
                    unsafe { crate::intrinsics::simd_reduce_mul_ordered(self, 1.) }
                }
            }

            /// Horizontal maximum.  Returns the maximum lane in the vector.
            ///
            /// Returns values based on equality, so a vector containing both `0.` and `-0.` may
            /// return either.  This function will not return `NaN` unless all lanes are `NaN`.
            /// ```
            /// # use core_simd::*;
            /// let a = f32x4::from_array([1.0, 2.0, 3.0, 4.0]).horizontal_max();
            /// assert_eq!(4.0, a);
            /// let b = f32x4::from_array([f32::NAN, 2.0, 0.0, 0.0]).horizontal_max();
            /// assert_eq!(2.0, b);
            /// ```
            #[inline]
            pub fn horizontal_max(self) -> $scalar {
                unsafe { crate::intrinsics::simd_reduce_max(self) }
            }

            /// Horizontal minimum.  Returns the minimum lane in the vector.
            ///
            /// Returns values based on equality, so a vector containing both `0.` and `-0.` may
            /// return either.  This function will not return `NaN` unless all lanes are `NaN`.
            /// ```
            /// # use core_simd::*;
            /// let a = f32x4::from_array([1.0, 2.0, 3.0, 4.0]).horizontal_min();
            /// assert_eq!(1.0, a);
            /// let b = f32x4::from_array([f32::NAN, 2.0, 0.0, 0.0]).horizontal_min();
            /// assert_eq!(0.0, b);
            /// ```
            #[inline]
            pub fn horizontal_min(self) -> $scalar {
                unsafe { crate::intrinsics::simd_reduce_min(self) }
            }
        }
    }
}

macro_rules! impl_full_mask_reductions {
    { $name:ident, $bits_ty:ident } => {
        impl<T: crate::Mask, const LANES: usize> $name<T, LANES>
        where
            crate::$bits_ty<LANES>: crate::LanesAtMost32
        {
            #[inline]
            pub fn any(self) -> bool {
                unsafe { crate::intrinsics::simd_reduce_any(self.to_int()) }
            }

            #[inline]
            pub fn all(self) -> bool {
                unsafe { crate::intrinsics::simd_reduce_all(self.to_int()) }
            }
        }
    }
}

macro_rules! impl_opaque_mask_reductions {
    { $name:ident, $bits_ty:ident } => {
        impl<const LANES: usize> $name<LANES>
        where
            crate::$bits_ty<LANES>: crate::LanesAtMost32,
            $name<LANES>: crate::Mask,
        {
            /// Returns true if any lane is set, or false otherwise.
            #[inline]
            pub fn any(self) -> bool {
                self.0.any()
            }

            /// Returns true if all lanes are set, or false otherwise.
            #[inline]
            pub fn all(self) -> bool {
                self.0.all()
            }
        }
    }
}
