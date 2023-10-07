// The k-means implementations here are based upon the following paper:
//
// Thompson, S., Celebi, M.E. & Buck, K.H. Fast color quantization using MacQueen’s k-means algorithm.
// Journal of Real-Time Image Processing, vol. 17, 1609–1624, 2020.
// https://doi.org/10.1007/s11554-019-00914-6
//
// Accessed from https://faculty.uca.edu/ecelebi/documents/JRTIP_2020a.pdf

mod kdtree;
mod online;

#[cfg(feature = "threads")]
mod minibatch;

pub(crate) use kdtree::*;
pub use minibatch::*;
pub use online::*;

use crate::{AboveMaxLen, MAX_COLORS, MAX_K};

#[derive(Debug, Clone)]
#[repr(transparent)]
pub struct Centroids<Color>(Vec<Color>);

impl<Color> Centroids<Color> {
    #[must_use]
    pub fn into_inner(self) -> Vec<Color> {
        self.0
    }

    #[must_use]
    pub fn from_truncated(mut centroids: Vec<Color>) -> Self {
        centroids.truncate(MAX_K);
        Self(centroids)
    }

    #[allow(clippy::cast_possible_truncation)]
    #[must_use]
    pub fn num_colors(&self) -> u16 {
        self.0.len() as u16
    }
}

impl<Color> From<Centroids<Color>> for Vec<Color> {
    fn from(value: Centroids<Color>) -> Self {
        value.into_inner()
    }
}

impl<Color> TryFrom<Vec<Color>> for Centroids<Color> {
    type Error = AboveMaxLen<u16>;

    fn try_from(colors: Vec<Color>) -> Result<Self, Self::Error> {
        if colors.len() <= MAX_K {
            Ok(Self(colors))
        } else {
            Err(AboveMaxLen(MAX_COLORS))
        }
    }
}
