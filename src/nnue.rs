// Learned evaluation (NNUE style) trained by tocandira.
//
// One sparse feature per piece on the board, relative to the perspective owner's queen:
//   index = ((cell * 2 + enemy) * 8 + bug) * 4 + level
// with cell the hex offset within `radius` of the anchor, bug in tocandira's order
// (Q A B G S M L P) and level the stack height bucket 0, 1, 2, 3+. Both perspectives are
// accumulated (sum of embedding rows plus a projection of the pieces in hand), clipped to
// [0, 1], then two small layers give a score in the same units as BasicEvaluator.
//
// Weights come from `hive-export`: "TOCA", u32 version, radius, num_features, hidden, l2,
// f32 scale, then little endian f32 arrays ft, hand_w, hand_b, l1_w, l1_b, l2_w, l2_b,
// out_w, out_b.

use crate::{Board, Bug, Color, Hex, Rules, Turn, ROW_SIZE, START_HEX};
use minimax::{Evaluation, Evaluator};
use std::sync::Arc;

const LEVELS: usize = 4;
const FEATURES_PER_CELL: usize = 2 * 8 * LEVELS;
const MAX_PIECES: usize = 28;
const MAX_HIDDEN: usize = 512;
const MAX_L2: usize = 64;
// Pieces of one bug a side can hold in hand.
const MAX_COUNT: usize = 3;
// Keep the score clear of the win/loss values.
const SCORE_LIMIT: f32 = 20000.0;
// Quantization: accumulators and activations carry 127 per unit, layer weights 64 per unit.
const ACT_SCALE: f64 = 127.0;
const WEIGHT_SCALE: f64 = 64.0;
const WEIGHT_SHIFT: u32 = 6;
// tocandira's bug order, for the feature index and the pieces in hand vector.
const BUG_ORDER: [Bug; 8] = [
    Bug::Queen,
    Bug::Ant,
    Bug::Beetle,
    Bug::Grasshopper,
    Bug::Spider,
    Bug::Mosquito,
    Bug::Ladybug,
    Bug::Pillbug,
];

fn bug_index(bug: Bug) -> usize {
    BUG_ORDER.iter().position(|&b| b == bug).unwrap()
}

pub struct Nnue {
    radius: i32,
    pub num_features: usize,
    hidden: usize,
    l2: usize,
    scale: f32,
    // (dq + radius) * (2 * radius + 1) + (dr + radius) -> cell index, same order as tocandira.
    cell_index: Vec<usize>,
    // Feature table rows and biases in accumulator units (ACT_SCALE per unit).
    ft: Vec<i16>,
    // hand_w column times count, for each (slot, count): [16][MAX_COUNT + 1][hidden].
    hand_rows: Vec<i16>,
    hand_b: Vec<i16>,
    // Layer weights in WEIGHT_SCALE units, biases in ACT_SCALE * WEIGHT_SCALE units.
    l1_w: Vec<i8>,
    l1_b: Vec<i32>,
    l2_w: Vec<i8>,
    l2_b: Vec<i32>,
    out_w: Vec<f32>,
    out_b: f32,
}

fn quantize_i16(values: &[f32], scale: f64) -> Vec<i16> {
    values
        .iter()
        .map(|&v| round_half_even(v as f64 * scale).clamp(-32767.0, 32767.0) as i16)
        .collect()
}

fn quantize_i8(values: &[f32], scale: f64) -> Vec<i8> {
    values.iter().map(|&v| round_half_even(v as f64 * scale).clamp(-127.0, 127.0) as i8).collect()
}

fn quantize_i32(values: &[f32], scale: f64) -> Vec<i32> {
    values.iter().map(|&v| round_half_even(v as f64 * scale) as i32).collect()
}

// Python's round(): halves go to the even neighbour.
fn round_half_even(x: f64) -> f64 {
    if (x - x.trunc()).abs() == 0.5 {
        2.0 * (x / 2.0).round()
    } else {
        x.round()
    }
}

fn hex_distance(dq: i32, dr: i32) -> i32 {
    dq.abs().max(dr.abs()).max((dq - dr).abs())
}

fn read_u32(buf: &[u8], pos: &mut usize) -> std::io::Result<u32> {
    let bytes: [u8; 4] = buf
        .get(*pos..*pos + 4)
        .ok_or_else(|| std::io::Error::other("truncated nnue file"))?
        .try_into()
        .unwrap();
    *pos += 4;
    Ok(u32::from_le_bytes(bytes))
}

fn read_f32s(buf: &[u8], pos: &mut usize, n: usize) -> std::io::Result<Vec<f32>> {
    let slice =
        buf.get(*pos..*pos + 4 * n).ok_or_else(|| std::io::Error::other("truncated nnue file"))?;
    *pos += 4 * n;
    Ok(slice.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect())
}

impl Nnue {
    pub fn load(path: &str) -> std::io::Result<Nnue> {
        Nnue::parse(&std::fs::read(path)?)
    }

    pub fn parse(buf: &[u8]) -> std::io::Result<Nnue> {
        if buf.get(0..4) != Some(b"TOCA") {
            return Err(std::io::Error::other("not a tocandira nnue file"));
        }
        let mut pos = 4;
        let version = read_u32(buf, &mut pos)?;
        if version != 1 {
            return Err(std::io::Error::other(format!("unsupported nnue version {version}")));
        }
        let radius = read_u32(buf, &mut pos)? as i32;
        let num_features = read_u32(buf, &mut pos)? as usize;
        let hidden = read_u32(buf, &mut pos)? as usize;
        let l2 = read_u32(buf, &mut pos)? as usize;
        let scale = f32::from_bits(read_u32(buf, &mut pos)?);

        let width = (2 * radius + 1) as usize;
        let mut cell_index = vec![usize::MAX; width * width];
        let mut cells = 0;
        for dq in -radius..=radius {
            for dr in -radius..=radius {
                if hex_distance(dq, dr) <= radius {
                    cell_index[(dq + radius) as usize * width + (dr + radius) as usize] = cells;
                    cells += 1;
                }
            }
        }
        if cells * FEATURES_PER_CELL != num_features {
            return Err(std::io::Error::other("nnue radius and feature count disagree"));
        }

        if hidden > MAX_HIDDEN || l2 > MAX_L2 || !hidden.is_multiple_of(8) {
            return Err(std::io::Error::other("unsupported nnue layer sizes"));
        }
        let ft = read_f32s(buf, &mut pos, num_features * hidden)?;
        let hand_w = read_f32s(buf, &mut pos, hidden * 16)?;
        let mut hand_rows = vec![0.0; 16 * (MAX_COUNT + 1) * hidden];
        for slot in 0..16 {
            for count in 0..=MAX_COUNT {
                for i in 0..hidden {
                    hand_rows[(slot * (MAX_COUNT + 1) + count) * hidden + i] =
                        hand_w[i * 16 + slot] * count as f32;
                }
            }
        }
        let hand_b = read_f32s(buf, &mut pos, hidden)?;
        let l1_w = read_f32s(buf, &mut pos, l2 * 2 * hidden)?;
        let l1_b = read_f32s(buf, &mut pos, l2)?;
        let l2_w = read_f32s(buf, &mut pos, l2 * l2)?;
        let l2_b = read_f32s(buf, &mut pos, l2)?;
        let out_w = read_f32s(buf, &mut pos, l2)?;
        let out_b = read_f32s(buf, &mut pos, 1)?[0];
        if pos != buf.len() {
            return Err(std::io::Error::other("nnue file has trailing bytes"));
        }
        Ok(Nnue {
            radius,
            num_features,
            hidden,
            l2,
            scale,
            cell_index,
            ft: quantize_i16(&ft, ACT_SCALE),
            hand_rows: quantize_i16(&hand_rows, ACT_SCALE),
            hand_b: quantize_i16(&hand_b, ACT_SCALE),
            l1_w: quantize_i8(&l1_w, WEIGHT_SCALE),
            l1_b: quantize_i32(&l1_b, ACT_SCALE * WEIGHT_SCALE),
            l2_w: quantize_i8(&l2_w, WEIGHT_SCALE),
            l2_b: quantize_i32(&l2_b, ACT_SCALE * WEIGHT_SCALE),
            out_w,
            out_b,
        })
    }

    // Offsets beyond the radius are projected onto the ring, rounding in cube coordinates.
    // Same arithmetic as tocandira's encoder (f64, round half to even), so the features match.
    fn clamp(&self, dq: i32, dr: i32) -> (i32, i32) {
        let d = hex_distance(dq, dr);
        if d <= self.radius {
            return (dq, dr);
        }
        let scale = self.radius as f64 / d as f64;
        // nokamute axial -> standard axial (x = q - r) -> cube (x, y, z) with x + y + z = 0.
        let (x, z) = (dq - dr, dr);
        let (fx, fy, fz) = (x as f64 * scale, (-x - z) as f64 * scale, z as f64 * scale);
        let (mut rx, ry, mut rz) = (round_half_even(fx), round_half_even(fy), round_half_even(fz));
        let (ex, ey, ez) = ((rx - fx).abs(), (ry - fy).abs(), (rz - fz).abs());
        if ex > ey && ex > ez {
            rx = -ry - rz;
        } else if ey <= ez {
            rz = -rx - ry;
        }
        ((rx + rz) as i32, rz as i32)
    }

    fn feature(&self, dq: i32, dr: i32, enemy: bool, bug: Bug, level: u8) -> usize {
        let (dq, dr) = self.clamp(dq, dr);
        let width = (2 * self.radius + 1) as usize;
        let cell =
            self.cell_index[(dq + self.radius) as usize * width + (dr + self.radius) as usize];
        let within =
            (enemy as usize * 8 + bug_index(bug)) * LEVELS + (level as usize).min(LEVELS - 1);
        cell * FEATURES_PER_CELL + within
    }

    // The perspective owner's queen, else the other queen, else where the first piece went.
    fn anchor(board: &Board, color: Color) -> Hex {
        if board.remaining[color as usize][Bug::Queen as usize] == 0 {
            board.queens[color as usize]
        } else if board.remaining[color.other()][Bug::Queen as usize] == 0 {
            board.queens[color.other()]
        } else {
            START_HEX
        }
    }

    // Shortest wrapped offset between two hexes of the grid, as (dq, dr).
    fn offset(hex: Hex, anchor: Hex) -> (i32, i32) {
        let row = ROW_SIZE as i32;
        let wrap = |x: i32| (x + row / 2).rem_euclid(row) - row / 2;
        let (q, r) = ((hex % ROW_SIZE) as i32, (hex / ROW_SIZE) as i32);
        let (aq, ar) = ((anchor % ROW_SIZE) as i32, (anchor / ROW_SIZE) as i32);
        (wrap(q - aq), wrap(r - ar))
    }

    // Active features from `color`'s perspective, one per piece on the board.
    pub fn features(&self, board: &Board, color: Color) -> Vec<usize> {
        let mut out = [0; MAX_PIECES];
        let n = self.features_into(board, color, &mut out);
        out[..n].to_vec()
    }

    fn features_into(&self, board: &Board, color: Color, out: &mut [usize; MAX_PIECES]) -> usize {
        let anchor = Nnue::anchor(board, color);
        let mut n = 0;
        for &hex in board.occupied_hexes.iter().flatten() {
            let node = board.node(hex);
            let (dq, dr) = Nnue::offset(hex, anchor);
            let level = board.height(hex) - 1;
            out[n] = self.feature(dq, dr, node.color() != color, node.bug(), level);
            n += 1;
        }
        for under in board.get_underworld() {
            let node = under.node();
            let (dq, dr) = Nnue::offset(under.hex(), anchor);
            out[n] = self.feature(dq, dr, node.color() != color, node.bug(), under.height() - 1);
            n += 1;
        }
        n
    }

    // Pieces in hand in tocandira's bug order, `color` first.
    pub fn hand(board: &Board, color: Color) -> [f32; 16] {
        let mut hand = [0.0; 16];
        for (i, c) in [color as usize, color.other()].into_iter().enumerate() {
            for (j, bug) in BUG_ORDER.iter().enumerate() {
                hand[i * 8 + j] = board.remaining[c][*bug as usize] as f32;
            }
        }
        hand
    }

    // Bias plus one table row per piece, for the perspective `color`, into `acc`.
    fn accumulate_pieces(&self, board: &Board, color: Color, acc: &mut [i16]) {
        let h = self.hidden;
        acc[..h].copy_from_slice(&self.hand_b);
        let mut feats = [0; MAX_PIECES];
        let n = self.features_into(board, color, &mut feats);
        for &f in &feats[..n] {
            add_row(&mut acc[..h], &self.ft[f * h..(f + 1) * h]);
        }
    }

    fn hand_row(&self, slot: usize, count: usize) -> &[i16] {
        let h = self.hidden;
        let row = (slot * (MAX_COUNT + 1) + count.min(MAX_COUNT)) * h;
        &self.hand_rows[row..row + h]
    }

    // Activations of one perspective from scratch (a board without accumulators).
    fn activations(&self, board: &Board, color: Color, act: &mut [u8; MAX_HIDDEN]) {
        let h = self.hidden;
        let mut acc = [0i16; MAX_HIDDEN];
        self.accumulate_pieces(board, color, &mut acc[..h]);
        for (slot, c) in [color as usize, color.other()].into_iter().enumerate() {
            for (j, bug) in BUG_ORDER.iter().enumerate() {
                let count = board.remaining[c][*bug as usize] as usize;
                add_row(&mut acc[..h], self.hand_row(slot * 8 + j, count));
            }
        }
        clip(&acc[..h], &mut act[..h]);
    }

    pub fn evaluate(&self, board: &Board) -> Evaluation {
        let me = board.to_move();
        let them = if me == Color::White { Color::Black } else { Color::White };
        let h = self.hidden;
        let mut own = [0u8; MAX_HIDDEN];
        let mut opp = [0u8; MAX_HIDDEN];
        match board.nnue.as_deref() {
            Some(state) if std::ptr::eq(&*state.net, self) => {
                clip(&state.acc[me as usize][..h], &mut own[..h]);
                clip(&state.acc[them as usize][..h], &mut opp[..h]);
            }
            _ => {
                self.activations(board, me, &mut own);
                self.activations(board, them, &mut opp);
            }
        }
        let mut x = [0u8; MAX_L2];
        for (i, row) in self.l1_w.chunks_exact(2 * h).enumerate() {
            let sum = dot(&own[..h], &row[..h]) + dot(&opp[..h], &row[h..]) + self.l1_b[i];
            x[i] = (sum >> WEIGHT_SHIFT).clamp(0, ACT_SCALE as i32) as u8;
        }
        let mut y = [0u8; MAX_L2];
        for (i, row) in self.l2_w.chunks_exact(self.l2).enumerate() {
            let sum = dot(&x[..self.l2], row) + self.l2_b[i];
            y[i] = (sum >> WEIGHT_SHIFT).clamp(0, ACT_SCALE as i32) as u8;
        }
        let out: f32 =
            self.out_w.iter().zip(&y).map(|(w, &a)| w * a as f32 / ACT_SCALE as f32).sum::<f32>()
                + self.out_b;
        (out * self.scale).clamp(-SCORE_LIMIT, SCORE_LIMIT).round() as Evaluation
    }
}

fn clip(acc: &[i16], act: &mut [u8]) {
    for (a, v) in act.iter_mut().zip(acc) {
        *a = (*v).clamp(0, ACT_SCALE as i16) as u8;
    }
}

// Accumulators kept inside a Board: bias, one table row per piece on the board and one row
// per (bug, count) in hand, for each perspective colour. Pieces are added and removed as
// the board changes; a queen appearing or leaving changes the anchors, so that refreshes
// both perspectives from scratch.
#[derive(Clone)]
pub struct NnueState {
    net: Arc<Nnue>,
    acc: [[i16; MAX_HIDDEN]; 2],
}

impl Board {
    pub fn attach_nnue(&mut self, net: Arc<Nnue>) {
        self.nnue = Some(Box::new(NnueState { net, acc: [[0; MAX_HIDDEN]; 2] }));
        self.nnue_refresh();
    }

    fn nnue_refresh(&mut self) {
        let mut state = self.nnue.take().unwrap();
        let h = state.net.hidden;
        for color in [Color::White, Color::Black] {
            let acc = &mut state.acc[color as usize];
            state.net.accumulate_pieces(self, color, acc);
            for (slot, c) in [color as usize, color.other()].into_iter().enumerate() {
                for (j, bug) in BUG_ORDER.iter().enumerate() {
                    let count = self.remaining[c][*bug as usize] as usize;
                    add_row(&mut acc[..h], state.net.hand_row(slot * 8 + j, count));
                }
            }
        }
        self.nnue = Some(state);
    }

    // After a piece landed on `hex` (placed or moved there).
    pub(crate) fn nnue_on_insert(&mut self, hex: Hex, bug: Bug, color: Color) {
        if bug == Bug::Queen {
            return self.nnue_refresh();
        }
        let level = self.height(hex) - 1;
        let mut state = self.nnue.take().unwrap();
        let h = state.net.hidden;
        for p in [Color::White, Color::Black] {
            let (dq, dr) = Nnue::offset(hex, Nnue::anchor(self, p));
            let f = state.net.feature(dq, dr, color != p, bug, level);
            add_row(&mut state.acc[p as usize][..h], &state.net.ft[f * h..(f + 1) * h]);
        }
        self.nnue = Some(state);
    }

    // Before the top piece of `hex` leaves (moved away or a placement undone). A leaving
    // queen only loses its own row here; the anchors are redone by the insert that completes
    // a move, or by the hand change that completes an undone placement.
    pub(crate) fn nnue_on_remove(&mut self, hex: Hex) {
        let node = self.node(hex);
        let level = self.height(hex) - 1;
        let mut state = self.nnue.take().unwrap();
        let h = state.net.hidden;
        for p in [Color::White, Color::Black] {
            let (dq, dr) = Nnue::offset(hex, Nnue::anchor(self, p));
            let f = state.net.feature(dq, dr, node.color() != p, node.bug(), level);
            sub_row(&mut state.acc[p as usize][..h], &state.net.ft[f * h..(f + 1) * h]);
        }
        self.nnue = Some(state);
    }

    // The count of `bug` in `color`'s hand changed by `delta` (the board already reflects it).
    pub(crate) fn nnue_on_hand(&mut self, bug: Bug, color: Color, delta: i8) {
        if bug == Bug::Queen {
            return self.nnue_refresh();
        }
        let mut state = self.nnue.take().unwrap();
        let h = state.net.hidden;
        let now = self.remaining[color as usize][bug as usize] as usize;
        let before = (now as i8 - delta) as usize;
        for p in [Color::White, Color::Black] {
            let slot = if color == p { 0 } else { 8 } + bug_index(bug);
            sub_row(&mut state.acc[p as usize][..h], state.net.hand_row(slot, before));
            add_row(&mut state.acc[p as usize][..h], state.net.hand_row(slot, now));
        }
        self.nnue = Some(state);
    }
}

fn sub_row(acc: &mut [i16], row: &[i16]) {
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") {
        // SAFETY: the feature was detected at runtime.
        return unsafe { avx2::sub_row(acc, row) };
    }
    for (a, w) in acc.iter_mut().zip(row) {
        *a = a.saturating_sub(*w);
    }
}

fn add_row(acc: &mut [i16], row: &[i16]) {
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") {
        // SAFETY: the feature was detected at runtime.
        return unsafe { avx2::add_row(acc, row) };
    }
    for (a, w) in acc.iter_mut().zip(row) {
        *a = a.saturating_add(*w);
    }
}

// Activations (0..127) against int8 weights, summed in int32.
fn dot(act: &[u8], w: &[i8]) -> i32 {
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") {
        // SAFETY: the feature was detected at runtime.
        return unsafe { avx2::dot(act, w) };
    }
    dot_scalar(act, w)
}

fn dot_scalar(act: &[u8], w: &[i8]) -> i32 {
    let mut lanes = [0i32; 8];
    let mut ca = act.chunks_exact(8);
    let mut cw = w.chunks_exact(8);
    for (x, y) in (&mut ca).zip(&mut cw) {
        for k in 0..8 {
            lanes[k] += x[k] as i32 * y[k] as i32;
        }
    }
    let tail: i32 =
        ca.remainder().iter().zip(cw.remainder()).map(|(x, y)| *x as i32 * *y as i32).sum();
    lanes.iter().sum::<i32>() + tail
}

// The same two loops with AVX2: 16 lane int16 adds, and the u8 x i8 multiply-add pairs that
// NNUE inference is built around.
#[cfg(target_arch = "x86_64")]
mod avx2 {
    use std::arch::x86_64::*;

    #[target_feature(enable = "avx2")]
    pub unsafe fn add_row(acc: &mut [i16], row: &[i16]) {
        let n = acc.len().min(row.len()) / 16 * 16;
        for i in (0..n).step_by(16) {
            let a = _mm256_loadu_si256(acc.as_ptr().add(i) as *const __m256i);
            let r = _mm256_loadu_si256(row.as_ptr().add(i) as *const __m256i);
            _mm256_storeu_si256(acc.as_mut_ptr().add(i) as *mut __m256i, _mm256_adds_epi16(a, r));
        }
        for i in n..acc.len().min(row.len()) {
            acc[i] = acc[i].saturating_add(row[i]);
        }
    }

    #[target_feature(enable = "avx2")]
    pub unsafe fn sub_row(acc: &mut [i16], row: &[i16]) {
        let n = acc.len().min(row.len()) / 16 * 16;
        for i in (0..n).step_by(16) {
            let a = _mm256_loadu_si256(acc.as_ptr().add(i) as *const __m256i);
            let r = _mm256_loadu_si256(row.as_ptr().add(i) as *const __m256i);
            _mm256_storeu_si256(acc.as_mut_ptr().add(i) as *mut __m256i, _mm256_subs_epi16(a, r));
        }
        for i in n..acc.len().min(row.len()) {
            acc[i] = acc[i].saturating_sub(row[i]);
        }
    }

    #[target_feature(enable = "avx2")]
    pub unsafe fn dot(act: &[u8], w: &[i8]) -> i32 {
        let len = act.len().min(w.len());
        let n = len / 32 * 32;
        let ones = _mm256_set1_epi16(1);
        let mut sum = _mm256_setzero_si256();
        for i in (0..n).step_by(32) {
            let a = _mm256_loadu_si256(act.as_ptr().add(i) as *const __m256i);
            let b = _mm256_loadu_si256(w.as_ptr().add(i) as *const __m256i);
            // Pairs of u8 * i8 products summed into i16 (127 * 127 * 2 fits), then into i32.
            sum = _mm256_add_epi32(sum, _mm256_madd_epi16(_mm256_maddubs_epi16(a, b), ones));
        }
        let halves = _mm_add_epi32(_mm256_castsi256_si128(sum), _mm256_extracti128_si256(sum, 1));
        let quads = _mm_add_epi32(halves, _mm_srli_si128(halves, 8));
        let mut total = _mm_cvtsi128_si32(_mm_add_epi32(quads, _mm_srli_si128(quads, 4)));
        for i in n..len {
            total += act[i] as i32 * w[i] as i32;
        }
        total
    }
}

#[derive(Clone)]
pub struct NnueEvaluator(pub Arc<Nnue>);

impl Evaluator for NnueEvaluator {
    type G = Rules;

    fn evaluate(&self, board: &Board) -> Evaluation {
        self.0.evaluate(board)
    }

    fn generate_noisy_moves(&self, board: &Board, moves: &mut Vec<Turn>) {
        crate::noisy_moves(board, moves);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // A radius 6, hidden 4, l2 2 net with all weights set to a constant.
    fn tiny() -> Nnue {
        let (radius, hidden, l2) = (6u32, 8usize, 2usize);
        let num_features = 127 * FEATURES_PER_CELL;
        let mut buf = Vec::new();
        buf.extend_from_slice(b"TOCA");
        for v in [1u32, radius, num_features as u32, hidden as u32, l2 as u32] {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        buf.extend_from_slice(&400.0f32.to_le_bytes());
        let sizes =
            [num_features * hidden, hidden * 16, hidden, l2 * 2 * hidden, l2, l2 * l2, l2, l2, 1];
        for n in sizes {
            for _ in 0..n {
                buf.extend_from_slice(&0.01f32.to_le_bytes());
            }
        }
        Nnue::parse(&buf).unwrap()
    }

    #[test]
    fn features_follow_the_pieces() {
        let nnue = tiny();
        // wA1 (0,0), bA1 (1,0), wQ (-1,0), bQ (2,0).
        let board = Board::from_game_string("Base;;;wA1;bA1 wA1-;wQ -wA1;bQ bA1-").unwrap();
        let own = nnue.features(&board, Color::White);
        assert_eq!(own.len(), 4);
        // White's queen is the anchor: cell (0, 0), own colour, queen, level 0.
        assert!(own.contains(&nnue.feature(0, 0, false, Bug::Queen, 0)));
        // The white ant is one step east of the white queen, the black ant two steps.
        assert!(own.contains(&nnue.feature(1, 0, false, Bug::Ant, 0)));
        assert!(own.contains(&nnue.feature(2, 0, true, Bug::Ant, 0)));
        // From black's queen the white ant is an enemy piece two steps west.
        let opp = nnue.features(&board, Color::Black);
        assert!(opp.contains(&nnue.feature(-2, 0, true, Bug::Ant, 0)));
        assert!(own.iter().all(|&f| f < nnue.num_features));
        let hand = Nnue::hand(&board, Color::White);
        assert_eq!(hand[0], 0.0); // white queen placed
        assert_eq!(hand.iter().sum::<f32>(), 22.0 - 4.0);
        let _ = nnue.evaluate(&board);
    }

    // The accumulators kept in the board must equal a from scratch evaluation after any
    // sequence of moves and undos, queens moving included.
    #[test]
    fn incremental_accumulators_match_from_scratch() {
        use minimax::{Game, Strategy};
        let net = Arc::new(tiny_random());
        let mut board = Board::from_game_type("Base+MLP").unwrap();
        board.attach_nnue(net.clone());
        let mut random = minimax::Random::<Rules>::new();
        let mut history = Vec::new();
        for ply in 0..120 {
            if Rules::get_winner(&board).is_some() {
                break;
            }
            let m = random.choose_move(&board).unwrap();
            board.apply(m);
            history.push(m);
            let mut fresh = board.clone();
            fresh.nnue = None;
            assert_eq!(net.evaluate(&board), net.evaluate(&fresh), "ply {ply}");
            // Undo and redo keep the accumulators in step as well.
            if ply % 7 == 3 {
                board.undo(m);
                let mut fresh = board.clone();
                fresh.nnue = None;
                assert_eq!(net.evaluate(&board), net.evaluate(&fresh), "undo at ply {ply}");
                board.apply(m);
            }
        }
        assert!(history.len() > 20);
    }

    // Like tiny() but with varied weights, so mismatches cannot cancel out.
    fn tiny_random() -> Nnue {
        let (radius, hidden, l2) = (6u32, 8usize, 2usize);
        let num_features = 127 * FEATURES_PER_CELL;
        let mut buf = Vec::new();
        buf.extend_from_slice(b"TOCA");
        for v in [1u32, radius, num_features as u32, hidden as u32, l2 as u32] {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        buf.extend_from_slice(&400.0f32.to_le_bytes());
        let sizes =
            [num_features * hidden, hidden * 16, hidden, l2 * 2 * hidden, l2, l2 * l2, l2, l2, 1];
        let mut state = 7u32;
        for n in sizes {
            for _ in 0..n {
                state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                let w = (state >> 8) as f32 / (1u32 << 24) as f32 * 0.4 - 0.2;
                buf.extend_from_slice(&w.to_le_bytes());
            }
        }
        Nnue::parse(&buf).unwrap()
    }

    #[test]
    fn covered_pieces_keep_their_level() {
        let nnue = tiny();
        // As above, then wB1 (-1,-1), bB1 (3,0), wG1 (-1,1), and the beetle climbs onto bQ (2,0).
        let board = Board::from_game_string(
            "Base;;;wA1;bA1 wA1-;wQ -wA1;bQ bA1-;wB1 wQ/;bB1 bQ-;wG1 /wQ;bB1 bQ",
        )
        .unwrap();
        let own = nnue.features(&board, Color::White);
        assert_eq!(own.len(), 7);
        // The black beetle sits on its queen: level 1 at the queen's hex, (3, 0) from white's queen.
        assert!(own.contains(&nnue.feature(3, 0, true, Bug::Beetle, 1)));
        assert!(own.contains(&nnue.feature(3, 0, true, Bug::Queen, 0)));
    }
}
