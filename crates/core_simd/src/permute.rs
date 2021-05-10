macro_rules! impl_shuffle_lane {
    { $name:ident, $fn:ident, $n:literal } => {
        impl $name<$n> {
            /// A const SIMD shuffle that takes 2 SIMD vectors and produces another vector, using
            /// the indices in the const parameter. The first or "self" vector will have its lanes
            /// indexed from 0, and the second vector will have its first lane indexed at $n.
            /// Indices must be in-bounds of either vector at compile time.
            ///
            /// Some SIMD shuffle instructions can be quite slow, so avoiding them by loading data
            /// into the desired patterns in advance is preferred, but shuffles are still faster
            /// than storing and reloading from memory.
            #[inline]
            pub fn shuffle<const IDX: [u32; $n]>(self, second: Self) -> Self {
                unsafe { crate::intrinsics::$fn(self, second, IDX) }
            }

            /// Reverse the order of the lanes in the vector.
            #[inline]
            pub fn reverse(self) -> Self {
                const fn idx() -> [u32; $n] {
                    let mut idx = [0u32; $n];
                    let mut i = 0;
                    while i < $n {
                        idx[i] = ($n - i - 1) as u32;
                        i += 1;
                    }
                    idx
                }
                self.shuffle::<{ idx() }>(self)
            }

            /// Rotate a vector to the left `N` times.
            ///
            /// # Examples
            ///
            /// ```
            /// # use core_simd::SimdU32;
            /// let a = SimdU32::from_array([0, 1, 2, 3, 4]);
            /// let b = SimdU32::from_array([2, 3, 4, 0, 1]);
            /// assert_eq!(a.rotate_left::<{2}>(), b);
            /// ```
            #[inline]
            pub fn rotate_left<const N: u32>(self) -> Self {
                const fn idx() -> [u32; $n] {
                    let mut base = [0u32; $n];
                    let mut i = 0;
                    while i < $n {
                        base[i] = i as u32;
                        i += 1;
                    }
                    let mut i = 0;
                    while i < N {
                        let temp = base[0];
                        let mut j = 0;
                        while j < $n - 1 {
                            base[j] = base[j + 1];
                            j += 1;
                        }
                        base[$n - 1] = temp;
                        i += 1;
                    }
                    base
                }

                self.shuffle::<{ idx() }>(self)
            }

            /// Rotate a vector to the right `N` times.
            ///
            /// # Examples
            ///
            /// ```
            /// # use core_simd::SimdU32;
            /// let a = SimdU32::from_array([0, 1, 2, 3, 4]);
            /// let b = SimdU32::from_array([3, 4, 0, 1, 2]);
            /// assert_eq!(a.rotate_left::<{2}>(), b);
            /// ```
            #[inline]
            pub fn rotate_right<const N: u32>(self) -> Self {
                const fn idx() -> [u32; $n] {
                    let mut base = [0u32; $n];
                    let mut i = 0;
                    while i < $n {
                        base[i] = i as u32;
                        i += 1;
                    }
                    let mut i = 0;
                    while i < N {
                        let last = base[$n - 1];
                        let mut j = ($n - 2) as i32;
                        while j >= 0 {
                            base[(j + 1) as usize] = base[j as usize];
                            j -= 1;
                        }
                        base[0] = last;
                        i += 1;
                    }
                    base
                }

                self.shuffle::<{ idx() }>(self)
            }

            /// Interleave two vectors.
            ///
            /// Produces two vectors with lanes taken alternately from `self` and `other`.
            ///
            /// The first result contains the first `LANES / 2` lanes from `self` and `other`,
            /// alternating, starting with the first lane of `self`.
            ///
            /// The second result contains the last `LANES / 2` lanes from `self` and `other`,
            /// alternating, starting with the lane `LANES / 2` from the start of `self`.
            ///
            /// This particular permutation is efficient on many architectures.
            ///
            /// ```
            /// # use core_simd::SimdU32;
            /// let a = SimdU32::from_array([0, 1, 2, 3]);
            /// let b = SimdU32::from_array([4, 5, 6, 7]);
            /// let (x, y) = a.interleave(b);
            /// assert_eq!(x.to_array(), [0, 4, 1, 5]);
            /// assert_eq!(y.to_array(), [2, 6, 3, 7]);
            /// ```
            #[inline]
            pub fn interleave(self, other: Self) -> (Self, Self) {
                const fn lo() -> [u32; $n] {
                    let mut idx = [0u32; $n];
                    let mut i = 0;
                    while i < $n {
                        let offset = i / 2;
                        idx[i] = if i % 2 == 0 {
                            offset
                        } else {
                            $n + offset
                        } as u32;
                        i += 1;
                    }
                    idx
                }
                const fn hi() -> [u32; $n] {
                    let mut idx = [0u32; $n];
                    let mut i = 0;
                    while i < $n {
                        let offset = ($n + i) / 2;
                        idx[i] = if i % 2 == 0 {
                            offset
                        } else {
                            $n + offset
                        } as u32;
                        i += 1;
                    }
                    idx
                }
                (self.shuffle::<{ lo() }>(other), self.shuffle::<{ hi() }>(other))
            }

            /// Deinterleave two vectors.
            ///
            /// The first result takes every other lane of `self` and then `other`, starting with
            /// the first lane.
            ///
            /// The second result takes every other lane of `self` and then `other`, starting with
            /// the second lane.
            ///
            /// This particular permutation is efficient on many architectures.
            ///
            /// ```
            /// # use core_simd::SimdU32;
            /// let a = SimdU32::from_array([0, 4, 1, 5]);
            /// let b = SimdU32::from_array([2, 6, 3, 7]);
            /// let (x, y) = a.deinterleave(b);
            /// assert_eq!(x.to_array(), [0, 1, 2, 3]);
            /// assert_eq!(y.to_array(), [4, 5, 6, 7]);
            /// ```
            #[inline]
            pub fn deinterleave(self, other: Self) -> (Self, Self) {
                const fn even() -> [u32; $n] {
                    let mut idx = [0u32; $n];
                    let mut i = 0;
                    while i < $n {
                        idx[i] = 2 * i as u32;
                        i += 1;
                    }
                    idx
                }
                const fn odd() -> [u32; $n] {
                    let mut idx = [0u32; $n];
                    let mut i = 0;
                    while i < $n {
                        idx[i] = 1 + 2 * i as u32;
                        i += 1;
                    }
                    idx
                }
                (self.shuffle::<{ even() }>(other), self.shuffle::<{ odd() }>(other))
            }
        }
    }
}

macro_rules! impl_shuffle_2pow_lanes {
    { $name:ident } => {
        impl_shuffle_lane!{ $name, simd_shuffle2, 2 }
        impl_shuffle_lane!{ $name, simd_shuffle4, 4 }
        impl_shuffle_lane!{ $name, simd_shuffle8, 8 }
        impl_shuffle_lane!{ $name, simd_shuffle16, 16 }
        impl_shuffle_lane!{ $name, simd_shuffle32, 32 }
    }
}
