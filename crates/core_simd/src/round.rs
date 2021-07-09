macro_rules! implement {
    {
        $type:ident, $int_type:ident
    } => {
        #[cfg(feature = "std")]
        impl<const LANES: usize> crate::$type<LANES>
        where
            Self: crate::LanesAtMost32,
        {
            /// Returns the smallest integer greater than or equal to each lane.
            /// ```
            /// # use core_simd::*;
            /// let a = f32x4::from_array([-3.3, -0.7, 2.9, 0.2]);
            /// let b = f32x4::from_array([-3.0, 0.0, 3.0, 1.0]);
            /// let c = f32x4::ceil(a);
            /// assert_eq!(c, b);
            /// ```
            #[must_use = "method returns a new vector and does not mutate the original value"]
            #[inline]
            pub fn ceil(self) -> Self {
                unsafe { crate::intrinsics::simd_ceil(self) }
            }

            /// Returns the largest integer value less than or equal to each lane.
            /// ```
            /// # use core_simd::*;
            /// let a = f32x4::from_array([-3.3, -0.7, 2.3, 1.3]);
            /// let b = f32x4::from_array([-4.0, -1.0, 2.0, 1.0]);
            /// let c = f32x4::floor(a);
            /// assert_eq!(c, b);
            /// ```
            #[must_use = "method returns a new vector and does not mutate the original value"]
            #[inline]
            pub fn floor(self) -> Self {
                unsafe { crate::intrinsics::simd_floor(self) }
            }

            /// Rounds to the nearest integer value. Ties round away from zero.
            /// ```
            /// # use core_simd::*;
            /// let a = f32x4::from_array([-3.6, -0.5, 2.5, 0.6]);
            /// let b = f32x4::from_array([-4.0, -1.0, 3.0, 1.0]);
            /// let c = f32x4::round(a);
            /// assert_eq!(c, b);
            /// ```
            #[must_use = "method returns a new vector and does not mutate the original value"]
            #[inline]
            pub fn round(self) -> Self {
                unsafe { crate::intrinsics::simd_round(self) }
            }

            /// Returns the floating point's integer value, with its fractional part removed.
            /// ```
            /// # use core_simd::*;
            /// let a = f32x4::from_array([-3.47, -0.1234, 2.3, 0.1234]);
            /// let b = f32x4::from_array([-3.0, -0.0, 2.0, 0.0]);
            /// let c = f32x4::trunc(a);
            /// assert_eq!(c, b);
            /// ```
            #[must_use = "method returns a new vector and does not mutate the original value"]
            #[inline]
            pub fn trunc(self) -> Self {
                unsafe { crate::intrinsics::simd_trunc(self) }
            }

            /// Returns the floating point's fractional value, with its integer part removed.
            /// ```
            /// # use core_simd::*;
            /// let a = f32x4::from_array([-3.25, -0.75, 2.5, 10.0]);
            /// let b = f32x4::from_array([-0.25, -0.75, 0.5, 0.0]);
            /// let c = f32x4::fract(a);
            /// assert_eq!(c, b);
            /// ```
            #[must_use = "method returns a new vector and does not mutate the original value"]
            #[inline]
            pub fn fract(self) -> Self {
                self - self.trunc()
            }
        }

        impl<const LANES: usize> crate::$type<LANES>
        where
            Self: crate::LanesAtMost32,
            crate::$int_type<LANES>: crate::LanesAtMost32,
        {
            /// Rounds toward zero and converts to the same-width integer type, assuming that
            /// the value is finite and fits in that type.
            ///
            /// # Safety
            /// The value must:
            ///
            /// * Not be NaN
            /// * Not be infinite
            /// * Be representable in the return type, after truncating off its fractional part
            #[inline]
            pub unsafe fn to_int_unchecked(self) -> crate::$int_type<LANES> {
                crate::intrinsics::simd_cast(self)
            }

            /// Creates a floating-point vector from an integer vector.  Rounds values that are
            /// not exactly representable.
            #[inline]
            pub fn round_from_int(value: crate::$int_type<LANES>) -> Self {
                unsafe { crate::intrinsics::simd_cast(value) }
            }
        }
    }
}

implement! { SimdF32, SimdI32 }
implement! { SimdF64, SimdI64 }
