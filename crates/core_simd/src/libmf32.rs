use crate::SimdF32;

impl<const LANES: usize> SimdF32<LANES>
where
    Self: crate::LanesAtMost32,
{
    // This does not seem to be implemented, yet.
    #[inline]
    fn mul_add(self, mul: Self, add: Self) -> Self {
        self * mul + add
    }

    /// Compute the sine of the angle in radians.
    /// Result is accurate to 0.0000002.
    ///
    /// # Example
    ///
    /// ```
    /// use core_simd::SimdF32;
    /// let x = SimdF32::<8>::splat(1.0);
    /// assert!((x.sin() - SimdF32::<8>::splat((1.0_f32).sin())).abs().horizontal_max() < 0.0000002);
    /// ```
    #[inline]
    pub fn sin(&self) -> Self {
        let x = Self::splat(1.0 / (core::f32::consts::PI * 2.0)) * self;
        let x = x - x.floor() - 0.5;
        Self::splat(12.268859941019306_f32)
            .mul_add(x * x, Self::splat(-41.216241051002875_f32))
            .mul_add(x * x, Self::splat(76.58672703334098_f32))
            .mul_add(x * x, Self::splat(-81.59746095374902_f32))
            .mul_add(x * x, Self::splat(41.34151143437585_f32))
            .mul_add(x * x, Self::splat(-6.283184525811273_f32))
            * x
    }
}
