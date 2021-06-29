macro_rules! impl_integer_reductions {
    { $name:ident, $scalar:ty } => {
        impl<const LANES: usize> crate::$name<LANES>
        where
            Self: crate::Vector
        {
            /// Horizontal wrapping add.  Returns the sum of the lanes of the vector, with wrapping addition.
            #[inline]
            pub fn horizontal_sum(self) -> $scalar {
                unsafe { crate::intrinsics::simd_reduce_add_ordered(self, 0) }
            }

            /// Horizontal wrapping multiply.  Returns the product of the lanes of the vector, with wrapping multiplication.
            #[inline]
            pub fn horizontal_product(self) -> $scalar {
                unsafe { crate::intrinsics::simd_reduce_mul_ordered(self, 1) }
            }

            /// Horizontal bitwise "and".  Returns the cumulative bitwise "and" across the lanes of
            /// the vector.
            #[inline]
            pub fn horizontal_and(self) -> $scalar {
                unsafe { crate::intrinsics::simd_reduce_and(self) }
            }

            /// Horizontal bitwise "or".  Returns the cumulative bitwise "or" across the lanes of
            /// the vector.
            #[inline]
            pub fn horizontal_or(self) -> $scalar {
                unsafe { crate::intrinsics::simd_reduce_or(self) }
            }

            /// Horizontal bitwise "xor".  Returns the cumulative bitwise "xor" across the lanes of
            /// the vector.
            #[inline]
            pub fn horizontal_xor(self) -> $scalar {
                unsafe { crate::intrinsics::simd_reduce_xor(self) }
            }

            /// Horizontal maximum.  Returns the maximum lane in the vector.
            #[inline]
            pub fn horizontal_max(self) -> $scalar {
                unsafe { crate::intrinsics::simd_reduce_max(self) }
            }

            /// Horizontal minimum.  Returns the minimum lane in the vector.
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
            Self: crate::Vector
        {

            /// Horizontal add.  Returns the sum of the lanes of the vector.
            #[inline]
            pub fn horizontal_sum(self) -> $scalar {
                // LLVM sum is inaccurate on i586
                if cfg!(all(target_arch = "x86", not(target_feature = "sse2"))) {
                    self.as_array().iter().sum()
                } else {
                    unsafe { crate::intrinsics::simd_reduce_add_ordered(self, 0.) }
                }
            }

            /// Horizontal multiply.  Returns the product of the lanes of the vector.
            #[inline]
            pub fn horizontal_product(self) -> $scalar {
                // LLVM product is inaccurate on i586
                if cfg!(all(target_arch = "x86", not(target_feature = "sse2"))) {
                    self.as_array().iter().product()
                } else {
                    unsafe { crate::intrinsics::simd_reduce_mul_ordered(self, 1.) }
                }
            }

            /// Horizontal maximum.  Returns the maximum lane in the vector.
            ///
            /// Returns values based on equality, so a vector containing both `0.` and `-0.` may
            /// return either.  This function will not return `NaN` unless all lanes are `NaN`.
            #[inline]
            pub fn horizontal_max(self) -> $scalar {
                unsafe { crate::intrinsics::simd_reduce_max(self) }
            }

            /// Horizontal minimum.  Returns the minimum lane in the vector.
            ///
            /// Returns values based on equality, so a vector containing both `0.` and `-0.` may
            /// return either.  This function will not return `NaN` unless all lanes are `NaN`.
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
            crate::$bits_ty<LANES>: crate::Vector
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
            crate::$bits_ty<LANES>: crate::Vector,
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
