use crate::{squared_euclidean_distance, MAX_K};

use std::{array, mem::MaybeUninit};

use ordered_float::OrderedFloat;

const B: usize = 24;

// Since we don't collapse empty leaves, there is a very small chance that
// the number of leaves/branches can rise above MAX_COLORS.
// So, we use u16 indices instead of u8 to be safe.
// This doesn't increase the size of Branch anyways.
#[derive(Clone, Copy)]
enum NodeIndex {
    Leaf(u16),
    Branch(u16),
}

struct Branch {
    left_right: [NodeIndex; 2],
    value: f64,
}

#[inline]
fn bounds_contains<const N: usize>(bounds: &[[f64; 2]; N], point: [f64; N]) -> bool {
    bounds
        .iter()
        .zip(point)
        .all(|(&[l, u], p)| (l..u).contains(&p))
}

enum Leaf<const N: usize> {
    Inline {
        count: u8,
        keys: [u8; B],
        points: [[f64; N]; B],
        bounds: [[f64; 2]; N],
    },
    Overflow {
        keys: Vec<u8>,
        points: Vec<[f64; N]>,
        bounds: [[f64; 2]; N],
    },
}

impl<const N: usize> Leaf<N> {
    #[inline]
    fn push_unchecked(&mut self, key: u8, point: [f64; N]) {
        match self {
            Leaf::Inline { count, keys, points, .. } => {
                let c = usize::from(*count);
                points[c] = point;
                keys[c] = key;
                *count += 1;
            }
            Leaf::Overflow { keys, points, .. } => {
                points.push(point);
                keys.push(key);
            }
        }
    }

    #[inline]
    fn push(&mut self, dim: u8, key: u8, point: [f64; N]) -> Option<(Self, f64)> {
        match self {
            Leaf::Inline { count, keys, points, bounds } => {
                let len = usize::from(*count);
                if len < B {
                    self.push_unchecked(key, point);
                    None
                } else {
                    let d = usize::from(dim);
                    assert_eq!(len, B);
                    for i in 1..len {
                        let mut j = i;
                        while j > 0 && OrderedFloat(points[j - 1][d]) > OrderedFloat(points[j][d]) {
                            points.swap(j - 1, j);
                            keys.swap(j - 1, j);
                            j -= 1;
                        }
                    }

                    let mut split = B / 2;
                    let split_value = points[split][d];
                    #[allow(clippy::float_cmp)]
                    if points[split - 1][d] == split_value {
                        for offset in 1..(B / 2) {
                            if points[split - 1 - offset][d] < split_value {
                                split -= offset;
                                break;
                            }
                            if points[split + offset][d] > split_value {
                                split += offset;
                                break;
                            }
                        }
                        if split == B / 2 {
                            // all points in this leaf have the same value along dimension d
                            let mut leaf = Leaf::Overflow {
                                keys: keys.to_vec(),
                                points: points.to_vec(),
                                bounds: *bounds,
                            };
                            leaf.push_unchecked(key, point);
                            *self = leaf;
                            return None;
                        }
                    }

                    let value = (points[split - 1][d] + split_value) / 2.0;

                    let mut right = {
                        let count = B - split;

                        let mut bounds = *bounds;
                        bounds[d][0] = value;

                        let mut p = [[0.0; N]; B];
                        p[..count].copy_from_slice(&points[split..]);

                        let mut k = [0; B];
                        k[..count].copy_from_slice(&keys[split..]);

                        #[allow(clippy::cast_possible_truncation)]
                        let count = count as u8;

                        Leaf::Inline { count, points: p, keys: k, bounds }
                    };

                    bounds[d][1] = value;
                    #[allow(clippy::cast_possible_truncation)]
                    {
                        *count = split as u8;
                    }

                    if point[d] < value {
                        self.push_unchecked(key, point);
                    } else {
                        right.push_unchecked(key, point);
                    }

                    Some((right, value))
                }
            }
            Leaf::Overflow { .. } => {
                self.push_unchecked(key, point);
                None
            }
        }
    }

    #[inline]
    fn update(&mut self, index: u8, point: [f64; N]) -> bool {
        let i = usize::from(index);
        match self {
            Leaf::Inline { count, keys, points, bounds } => {
                if bounds_contains(bounds, point) {
                    points[i] = point;
                    true
                } else {
                    let c = usize::from(*count) - 1;
                    points[i] = points[c];
                    keys[i] = keys[c];
                    *count -= 1;
                    false
                }
            }
            Leaf::Overflow { keys, points, bounds } => {
                if bounds_contains(bounds, point) {
                    points[i] = point;
                    true
                } else {
                    points.swap_remove(i);
                    keys.swap_remove(i);
                    false
                }
            }
        }
    }

    #[inline]
    fn as_slices(&self) -> (&[u8], &[[f64; N]]) {
        match self {
            Leaf::Inline { count, keys, points, .. } => {
                let count = usize::from(*count);
                (&keys[..count], &points[..count])
            }
            Leaf::Overflow { keys, points, .. } => (keys.as_slice(), points.as_slice()),
        }
    }

    #[inline]
    fn iter(&self) -> impl Iterator<Item = (u8, [f64; N])> + '_ {
        let (keys, points) = self.as_slices();
        keys.iter().copied().zip(points.iter().copied())
    }
}

#[derive(Clone, Copy)]
pub struct KdTreeEntry {
    pub key: u8,
    leaf: u16,
    leaf_index: u8,
}

struct NearestNeighborState<const N: usize> {
    point: [f64; N],
    branch_dist: [f64; N],
    min_dist: f64,
    entry: KdTreeEntry,
    nearest: [f64; N],
}

pub struct KdTree<const N: usize> {
    point_count: u16,
    root: NodeIndex,
    branches: Vec<Branch>,
    leaves: Vec<Leaf<N>>,
}

impl<const N: usize> KdTree<N> {
    pub fn new(points: &[[f64; N]]) -> Self {
        let len = points.len();

        #[allow(clippy::cast_possible_truncation)]
        let k = len as u16;

        let capacity = 2 * (len + B - 1) / B;
        let mut tree = Self {
            point_count: k,
            root: NodeIndex::Leaf(0),
            branches: Vec::with_capacity(capacity),
            leaves: Vec::with_capacity(capacity),
        };

        #[allow(unsafe_code)] // OrderedFloat is repr(transparent)
        let points = unsafe {
            std::slice::from_raw_parts(points.as_ptr().cast::<[OrderedFloat<f64>; N]>(), len)
        };

        // points.sort_unstable();
        // assert # of duplicates for a point < B

        #[allow(clippy::cast_possible_truncation)]
        let mut indices = [array::from_fn(|i| i as u8); N];

        for (d, indices) in indices.iter_mut().enumerate() {
            indices[..len].sort_unstable_by_key(|&i| {
                let mut point = points[usize::from(i)];
                point.rotate_left(d);
                point
            });
        }

        let mut bounds = [[f64::NEG_INFINITY, f64::INFINITY]; N];
        let mut buf = [0; MAX_K];

        #[allow(clippy::unwrap_used)] // same length
        let indices: [&mut [u8; MAX_K]; N] =
            indices.iter_mut().collect::<Vec<_>>().try_into().unwrap();
        // the collect above may be eliminated by the compiler
        let indices = indices.map(|i| &mut i[..len]);

        #[allow(unsafe_code)] // OrderedFloat is repr(transparent)
        let points = unsafe { std::slice::from_raw_parts(points.as_ptr().cast::<[f64; N]>(), len) };

        tree.root = tree.build_rec(points, 0, &mut bounds, indices, &mut buf[..len]);

        debug_assert!(tree.leaves.iter().all(|leaf| {
            let bounds = match leaf {
                Leaf::Inline { bounds, .. } | Leaf::Overflow { bounds, .. } => bounds,
            };

            leaf.as_slices()
                .1
                .iter()
                .all(|&point| bounds_contains(bounds, point))
        }));

        tree
    }

    pub fn num_points(&self) -> u16 {
        self.point_count
    }

    pub fn iter(&self) -> impl Iterator<Item = (u8, [f64; N])> + '_ {
        self.leaves.iter().flat_map(Leaf::iter)
    }

    // Russell A. Brown, Building a Balanced k-d Tree in O(kn log n) Time,
    // Journal of Computer Graphics Techniques, vol. 4, no. 1, 50-68, 2015.
    // http://jcgt.org/published/0004/01/03/

    #[allow(clippy::too_many_lines)]
    fn build_rec<'a>(
        &mut self,
        points: &[[f64; N]],
        dim: u8,
        bounds: &mut [[f64; 2]; N],
        indices: [&'a mut [u8]; N],
        buf: &'a mut [u8],
    ) -> NodeIndex {
        let d = usize::from(dim);
        let len = buf.len();

        if len <= B {
            #[allow(clippy::cast_possible_truncation)]
            let count = len as u8;

            let mut k = [0; B];
            k[..len].copy_from_slice(indices[0]);

            let mut p = [[0.0; N]; B];
            for (&i, point) in indices[0].iter().zip(&mut p) {
                *point = points[usize::from(i)];
            }

            let leaf = Leaf::Inline {
                count,
                keys: k,
                points: p,
                bounds: *bounds,
            };

            #[allow(clippy::cast_possible_truncation)]
            let l = self.leaves.len() as u16;
            self.leaves.push(leaf);
            NodeIndex::Leaf(l)
        } else {
            let (split, median) = {
                let indices = &indices[0];
                let half_len = len / 2;

                let mut split = half_len;
                let split_value = points[usize::from(indices[split])][d];
                #[allow(clippy::float_cmp)]
                if points[usize::from(indices[split - 1])][d] == split_value {
                    for offset in 1..half_len {
                        if points[usize::from(indices[split - 1 - offset])][d] < split_value {
                            split -= offset;
                            break;
                        }
                        if points[usize::from(indices[split + offset])][d] > split_value {
                            split += offset;
                            break;
                        }
                    }
                    if split == half_len {
                        // all points from 0..(2 * half_len) have the same value along dimension d
                        if len % 2 == 1 && points[usize::from(indices[len - 1])][d] > split_value {
                            // check last element for odd length slices
                            split = len - 1;
                        } else {
                            // all points in this leaf have the same value along dimension d
                            // create empty left leaf, and continue to next dimension
                            split = 0;
                        }
                    }
                }

                let median = if split == 0 {
                    points[usize::from(indices[split])][d]
                } else {
                    (points[usize::from(indices[split - 1])][d]
                        + points[usize::from(indices[split])][d])
                        / 2.0
                };

                (split, median)
            };

            let mut buf = buf;
            // `MaybeUninit`s do not require initialization, so it is safe to assume that
            // an array of `MaybeUninit`s are initialized. See the `MaybeUninit` doc example.
            #[allow(unsafe_code)]
            let mut indices_l: [MaybeUninit<&mut [u8]>; N] =
                unsafe { MaybeUninit::uninit().assume_init() };
            #[allow(unsafe_code)]
            let mut indices_r: [MaybeUninit<&mut [u8]>; N] =
                unsafe { MaybeUninit::uninit().assume_init() };
            assert_eq!(indices_l.len(), indices.len());
            assert_eq!(indices_r.len(), indices.len());
            for (i, indices) in indices.into_iter().enumerate() {
                let (left, right) = buf.split_at_mut(split);
                let mut l = 0;
                let mut r = 0;
                for &i in &*indices {
                    if points[usize::from(i)][d] < median {
                        left[l] = i;
                        l += 1;
                    } else {
                        right[r] = i;
                        r += 1;
                    }
                }
                indices_l[i] = MaybeUninit::new(left);
                indices_r[i] = MaybeUninit::new(right);
                buf = indices;
            }
            indices_l.rotate_left(1);
            indices_r.rotate_left(1);
            // Each index is written to with an initialized value,
            // since the lengths are the same (see asserts above).
            #[allow(unsafe_code)]
            let indices_l = unsafe { indices_l.map(|x| MaybeUninit::assume_init(x)) };
            #[allow(unsafe_code)]
            let indices_r = unsafe { indices_r.map(|x| MaybeUninit::assume_init(x)) };
            let (buf_l, buf_r) = buf.split_at_mut(split);

            #[allow(clippy::cast_possible_truncation)]
            let dim = (dim + 1) % N as u8;

            let old = bounds[d][1];
            bounds[d][1] = median;
            let left = self.build_rec(points, dim, bounds, indices_l, buf_l);
            bounds[d][1] = old;

            let old = bounds[d][0];
            bounds[d][0] = median;
            let right = self.build_rec(points, dim, bounds, indices_r, buf_r);
            bounds[d][0] = old;

            #[allow(clippy::cast_possible_truncation)]
            let b = self.branches.len() as u16;
            self.branches
                .push(Branch { left_right: [left, right], value: median });
            NodeIndex::Branch(b)
        }
    }

    fn add(&mut self, key: u8, point: [f64; N]) {
        let mut node = self.root;
        let mut dim = 0;
        let mut parent = None;
        let mut is_right = false;

        loop {
            match node {
                NodeIndex::Leaf(i) => {
                    if let Some((right, value)) = self.leaves[usize::from(i)].push(dim, key, point)
                    {
                        #[allow(clippy::cast_possible_truncation)]
                        let branch_count = self.branches.len() as u16;
                        #[allow(clippy::cast_possible_truncation)]
                        let right_leaf_i = self.leaves.len() as u16;

                        self.leaves.push(right);
                        self.branches.push(Branch {
                            left_right: [NodeIndex::Leaf(i), NodeIndex::Leaf(right_leaf_i)],
                            value,
                        });

                        let index = NodeIndex::Branch(branch_count);
                        match parent {
                            Some(i) => {
                                self.branches[usize::from(i)].left_right[usize::from(is_right)] =
                                    index;
                            }
                            None => self.root = index,
                        }
                    }

                    break;
                }
                NodeIndex::Branch(i) => {
                    let Branch { left_right, value } = self.branches[usize::from(i)];
                    is_right = point[usize::from(dim)] >= value;
                    node = left_right[usize::from(is_right)];
                    parent = Some(i);
                    #[allow(clippy::cast_possible_truncation)]
                    {
                        dim = (dim + 1) % N as u8;
                    }
                }
            }
        }
    }

    pub fn update_entry(&mut self, entry: KdTreeEntry, point: [f64; N]) {
        let KdTreeEntry { key, leaf, leaf_index } = entry;
        if !self.leaves[usize::from(leaf)].update(leaf_index, point) {
            self.add(key, point);
        }
    }

    pub fn update_batch(&mut self, batch: &[[f64; N]]) {
        debug_assert_eq!(batch.len(), usize::from(self.point_count));
        let mut buf = [0; MAX_K];
        let mut len = 0;

        for leaf in &mut self.leaves {
            match leaf {
                Leaf::Inline { count, keys, points, bounds } => {
                    let mut index = 0;
                    while index < *count {
                        let i = usize::from(index);
                        let key = keys[i];
                        let point = batch[usize::from(key)];
                        if bounds_contains(bounds, point) {
                            points[i] = point;
                            index += 1;
                        } else {
                            let c = usize::from(*count) - 1;
                            points[i] = points[c];
                            keys[i] = keys[c];
                            *count -= 1;
                            buf[len] = key;
                            len += 1;
                        }
                    }
                }
                Leaf::Overflow { keys, points, bounds } => {
                    let mut i = 0;
                    while i < points.len() {
                        let key = keys[i];
                        let point = batch[usize::from(key)];
                        if bounds_contains(bounds, point) {
                            points[i] = point;
                            i += 1;
                        } else {
                            points.swap_remove(i);
                            keys.swap_remove(i);
                            buf[len] = key;
                            len += 1;
                        }
                    }
                }
            }
        }

        for &key in &buf[..len] {
            self.add(key, batch[usize::from(key)]);
        }
    }

    pub fn nearest_neighbor_entry(&self, point: [f64; N]) -> (KdTreeEntry, [f64; N]) {
        let mut data = NearestNeighborState {
            point,
            branch_dist: [0.0; N],
            min_dist: f64::INFINITY,
            entry: KdTreeEntry { key: 0, leaf: 0, leaf_index: 0 },
            nearest: [0.0; N],
        };

        self.nearest_neighbor_rec(&mut data, self.root, 0, 0.0);

        // found a closest point
        debug_assert!(data.min_dist.is_finite());

        (data.entry, data.nearest)
    }

    fn nearest_neighbor_rec(
        &self,
        state: &mut NearestNeighborState<N>,
        node: NodeIndex,
        dim: u8,
        branch_dist: f64,
    ) {
        match node {
            NodeIndex::Leaf(leaf) => {
                let mut leaf_index = 0;
                let mut key = 0;
                let mut nearest = [0.0; N];
                let mut min_dist = state.min_dist;
                let (keys, points) = self.leaves[usize::from(leaf)].as_slices();
                for (chunk_i, (k, p)) in
                    keys.chunks_exact(4).zip(points.chunks_exact(4)).enumerate()
                {
                    for (i, (&k, &p)) in k.iter().zip(p).enumerate() {
                        let dist = squared_euclidean_distance(p, state.point);
                        #[allow(clippy::cast_possible_truncation)]
                        if dist < min_dist {
                            min_dist = dist;
                            leaf_index = (chunk_i * 4 + i) as u8;
                            key = k;
                            nearest = p;
                        }
                    }
                }
                for (i, (&k, &p)) in keys
                    .chunks_exact(4)
                    .remainder()
                    .iter()
                    .zip(points.chunks_exact(4).remainder())
                    .enumerate()
                {
                    let dist = squared_euclidean_distance(p, state.point);
                    #[allow(clippy::cast_possible_truncation)]
                    if dist < min_dist {
                        min_dist = dist;
                        leaf_index = (keys.len() - keys.len() % 4 + i) as u8;
                        key = k;
                        nearest = p;
                    }
                }

                if min_dist < state.min_dist {
                    state.min_dist = min_dist;
                    state.entry = KdTreeEntry { key, leaf, leaf_index };
                    state.nearest = nearest;
                }
            }
            NodeIndex::Branch(i) => {
                let Branch { left_right, value } = self.branches[usize::from(i)];
                let d = usize::from(dim);
                let diff = state.point[d] - value;
                let direction = diff < 0.0;
                let closer = left_right[usize::from(!direction)];
                let farther = left_right[usize::from(direction)];

                #[allow(clippy::cast_possible_truncation)]
                let dim = (dim + 1) % N as u8;

                self.nearest_neighbor_rec(state, closer, dim, branch_dist);

                let old_diff = state.branch_dist[d];
                let branch_dist = branch_dist + (diff * diff - old_diff * old_diff);
                if branch_dist < state.min_dist {
                    state.branch_dist[d] = diff;
                    self.nearest_neighbor_rec(state, farther, dim, branch_dist);
                    state.branch_dist[d] = old_diff;
                }
            }
        }
    }
}
