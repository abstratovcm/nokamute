// Strategy scalars: counts a Hive player reads off the whole board, which the learned
// evaluation's piece features cannot compute because they depend on the hive's
// connectivity and on the sliding rules. Side to move first, then the opponent, for each of:
//   bugs that can move, pinned bugs, free hexes around the queen (6 before it is placed),
//   immovable bugs next to the queen, empty hexes the side can place on, queen covered.
// Movability follows BasicEvaluator: a bug is immovable when lifting it would split the
// hive (pinned) or when it crawls and no slide is open; a bug on top of a stack can move.

use crate::board::*;
use crate::bug::Bug;
use crate::eval::crawls;
use crate::hex_grid::*;

pub const NUM_SCALARS: usize = 12;
// Divisors that bring each scalar to about [0, 1] as a net input; larger values clip.
pub const SCALAR_MAX: [u32; NUM_SCALARS] = [14, 14, 14, 14, 6, 6, 6, 6, 24, 24, 1, 1];

pub fn scalars(board: &Board) -> [u8; NUM_SCALARS] {
    let cut = board.find_cut_vertexes();
    let mut buf = [0; 6];
    let mut immovable = HexSet::new();
    let mut movable = [0u8; 2];
    let mut pinned = [0u8; 2];
    for &hex in board.occupied_hexes.iter().flatten() {
        let node = board.node(hex);
        let color = node.color() as usize;
        if node.is_stacked() {
            movable[color] += 1;
            continue;
        }
        if cut.get(hex) {
            pinned[color] += 1;
        }
        if cut.get(hex)
            || (crawls(board, hex) && board.slidable_adjacent(&mut buf, hex, hex).next().is_none())
        {
            immovable.set(hex);
        } else {
            movable[color] += 1;
        }
    }

    // A side can place on the empty hexes next to its bugs and next to no enemy bug.
    let mut near = [HexSet::new(), HexSet::new()];
    for &hex in board.occupied_hexes.iter().flatten() {
        let color = board.node(hex).color() as usize;
        for adj in adjacent(hex) {
            if !board.occupied(adj) {
                near[color].set(adj);
            }
        }
    }
    let territory = [
        near[0].count_without(&near[1]).min(255) as u8,
        near[1].count_without(&near[0]).min(255) as u8,
    ];

    let mut liberties = [6u8; 2];
    let mut stuck = [0u8; 2];
    let mut covered = [0u8; 2];
    for c in 0..2 {
        if board.remaining[c][Bug::Queen as usize] == 0 {
            let queen = board.queens[c];
            liberties[c] =
                adjacent(queen).iter().filter(|&&adj| !board.occupied(adj)).count() as u8;
            stuck[c] = adjacent(queen).iter().filter(|&&adj| immovable.get(adj)).count() as u8;
            covered[c] = board.node(queen).is_stacked() as u8;
        }
    }

    let me = board.to_move() as usize;
    let them = 1 - me;
    [
        movable[me],
        movable[them],
        pinned[me],
        pinned[them],
        liberties[me],
        liberties[them],
        stuck[me],
        stuck[them],
        territory[me],
        territory[them],
        covered[me],
        covered[them],
    ]
}

// A scalar as a net input in activation units (127 per unit), rounded, clipped at 127.
pub(crate) fn scalar_input(value: u8, max: u32) -> u8 {
    ((value as u32 * 127 + max / 2) / max).min(127) as u8
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalars_count_from_the_side_to_move() {
        // A line wQ - wA1 - bA1 - bQ: both ants hold the hive together, both queens are free.
        // White to move.
        let board = Board::from_game_string("Base;;;wA1;bA1 wA1-;wQ -wA1;bQ bA1-").unwrap();
        let s = scalars(&board);
        // Movable: each queen (end of the line); the ants are pinned.
        assert_eq!(&s[0..4], &[1, 1, 1, 1]);
        // Each queen touches one bug: five free hexes, the ant next to it is stuck.
        assert_eq!(&s[4..8], &[5, 5, 1, 1]);
        // Placeable: the empty hexes around a side's bugs that touch no enemy bug, five each.
        assert_eq!(&s[8..10], &[5, 5]);
        assert_eq!(&s[10..12], &[0, 0]);

        // Black to move: the same board seen from the other side swaps the pairs.
        let board = Board::from_game_string("Base;;;wA1;bA1 wA1-;wQ -wA1").unwrap();
        let s = scalars(&board);
        assert_eq!(s[4], 6); // black's queen is not placed
        assert_eq!(s[5], 5);
    }

    #[test]
    fn covered_queen_and_stacked_beetle() {
        // The black beetle climbs onto the black queen: it moves (stacked), the queen is
        // covered and counts no free hexes beyond what is empty around its hex.
        let board = Board::from_game_string(
            "Base;;;wA1;bA1 wA1-;wQ -wA1;bQ bA1-;wB1 wQ/;bB1 bQ-;wG1 /wQ;bB1 bQ",
        )
        .unwrap();
        let s = scalars(&board);
        // White to move: black's queen (them) is covered, white's is not.
        assert_eq!(&s[10..12], &[0, 1]);
        assert_eq!(scalar_input(s[11], SCALAR_MAX[11]), 127);
        assert_eq!(scalar_input(3, 6), 64);
        assert_eq!(scalar_input(30, 24), 127);
    }
}
