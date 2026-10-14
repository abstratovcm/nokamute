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
// Keep the score clear of the win/loss values.
const SCORE_LIMIT: f32 = 20000.0;
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
    ft: Vec<f32>,
    hand_w: Vec<f32>,
    hand_b: Vec<f32>,
    l1_w: Vec<f32>,
    l1_b: Vec<f32>,
    l2_w: Vec<f32>,
    l2_b: Vec<f32>,
    out_w: Vec<f32>,
    out_b: f32,
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

        let ft = read_f32s(buf, &mut pos, num_features * hidden)?;
        let hand_w = read_f32s(buf, &mut pos, hidden * 16)?;
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
            ft,
            hand_w,
            hand_b,
            l1_w,
            l1_b,
            l2_w,
            l2_b,
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
        let anchor = Nnue::anchor(board, color);
        let mut out = Vec::with_capacity(28);
        for &hex in board.occupied_hexes.iter().flatten() {
            let node = board.node(hex);
            let (dq, dr) = Nnue::offset(hex, anchor);
            let level = board.height(hex) - 1;
            out.push(self.feature(dq, dr, node.color() != color, node.bug(), level));
        }
        for under in board.get_underworld() {
            let node = under.node();
            let (dq, dr) = Nnue::offset(under.hex(), anchor);
            out.push(self.feature(dq, dr, node.color() != color, node.bug(), under.height() - 1));
        }
        out
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

    fn accumulate(&self, features: &[usize], hand: &[f32; 16]) -> Vec<f32> {
        let mut acc = self.hand_b.clone();
        for (i, row) in self.hand_w.chunks_exact(16).enumerate() {
            acc[i] += row.iter().zip(hand).map(|(w, h)| w * h).sum::<f32>();
        }
        for &f in features {
            let row = &self.ft[f * self.hidden..(f + 1) * self.hidden];
            for (a, w) in acc.iter_mut().zip(row) {
                *a += w;
            }
        }
        for a in acc.iter_mut() {
            *a = a.clamp(0.0, 1.0);
        }
        acc
    }

    pub fn evaluate(&self, board: &Board) -> Evaluation {
        let me = board.to_move();
        let them = if me == Color::White { Color::Black } else { Color::White };
        let hand_me = Nnue::hand(board, me);
        let hand_them = Nnue::hand(board, them);
        let own = self.accumulate(&self.features(board, me), &hand_me);
        let opp = self.accumulate(&self.features(board, them), &hand_them);
        let mut x = Vec::with_capacity(self.l2);
        for (i, row) in self.l1_w.chunks_exact(2 * self.hidden).enumerate() {
            let dot: f32 = row[..self.hidden].iter().zip(&own).map(|(w, a)| w * a).sum::<f32>()
                + row[self.hidden..].iter().zip(&opp).map(|(w, a)| w * a).sum::<f32>();
            x.push((dot + self.l1_b[i]).clamp(0.0, 1.0));
        }
        let mut y = Vec::with_capacity(self.l2);
        for (i, row) in self.l2_w.chunks_exact(self.l2).enumerate() {
            let dot: f32 = row.iter().zip(&x).map(|(w, a)| w * a).sum();
            y.push((dot + self.l2_b[i]).clamp(0.0, 1.0));
        }
        let out: f32 = self.out_w.iter().zip(&y).map(|(w, a)| w * a).sum::<f32>() + self.out_b;
        (out * self.scale).clamp(-SCORE_LIMIT, SCORE_LIMIT).round() as Evaluation
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
        let (radius, hidden, l2) = (6u32, 4usize, 2usize);
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
