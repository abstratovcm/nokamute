extern crate easybench;
extern crate minimax;
extern crate nokamute;

use minimax::{Evaluator, Game, Strategy};
use nokamute::{BasicEvaluator, Board, Nnue, NnueEvaluator, Rules};
use std::sync::Arc;

// A radius 6, hidden 256, l2 32 net with pseudo random weights (the real file is not needed
// to time the forward pass).
fn synthetic_net(radius: u32, hidden: usize) -> Nnue {
    let l2 = 32usize;
    let cells = (3 * radius * radius + 3 * radius + 1) as usize;
    let num_features = cells * 64;
    let mut buf = Vec::new();
    buf.extend_from_slice(b"TOCA");
    for v in [1u32, radius, num_features as u32, hidden as u32, l2 as u32] {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    buf.extend_from_slice(&400.0f32.to_le_bytes());
    let sizes =
        [num_features * hidden, hidden * 16, hidden, l2 * 2 * hidden, l2, l2 * l2, l2, l2, 1];
    let mut state = 12345u32;
    for n in sizes {
        for _ in 0..n {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            let w = (state >> 8) as f32 / (1u32 << 24) as f32 * 0.2 - 0.1;
            buf.extend_from_slice(&w.to_le_bytes());
        }
    }
    Nnue::parse(&buf).unwrap()
}

// A mid game position: 40 random plies (the same approach as the minimax bench).
fn midgame() -> Board {
    let mut board = Board::default();
    let mut rand = minimax::Random::<Rules>::new();
    for _ in 0..40 {
        if Rules::get_winner(&board).is_some() {
            break;
        }
        board.apply(rand.choose_move(&board).unwrap());
    }
    board
}

fn main() {
    let board = midgame();
    let basic = BasicEvaluator::default();
    let nnue = NnueEvaluator(Arc::new(synthetic_net(6, 256)));
    println!("pieces on board: {}", nnue.0.features(&board, board.to_move()).len());
    println!("basic eval:            {}", easybench::bench(|| basic.evaluate(&board)));
    println!("nnue r6 h256:          {}", easybench::bench(|| nnue.evaluate(&board)));
    println!(
        "nnue feats:            {}",
        easybench::bench(|| nnue.0.features(&board, board.to_move()))
    );
    // Same arithmetic, a table that fits in L2: the difference is memory latency.
    let small = NnueEvaluator(Arc::new(synthetic_net(1, 256)));
    println!("nnue r1 h256 (in L2):  {}", easybench::bench(|| small.evaluate(&board)));
    let h128 = NnueEvaluator(Arc::new(synthetic_net(6, 128)));
    println!("nnue r6 h128:          {}", easybench::bench(|| h128.evaluate(&board)));
    // With the accumulators kept in the board: only the small layers per evaluation.
    let mut attached = board.clone();
    attached.attach_nnue(nnue.0.clone());
    println!("nnue r6 h256 incr:     {}", easybench::bench(|| nnue.evaluate(&attached)));
    let mut attached128 = board.clone();
    attached128.attach_nnue(h128.0.clone());
    println!("nnue r6 h128 incr:     {}", easybench::bench(|| h128.evaluate(&attached128)));
    // Cost of keeping them in step: one piece move (remove + insert) with and without.
    let m = minimax::Random::<Rules>::new().choose_move(&attached).unwrap();
    println!(
        "apply+undo plain:      {}",
        easybench::bench(|| {
            let mut b = board.clone();
            b.apply(m);
            b.undo(m);
        })
    );
    println!(
        "apply+undo incr:       {}",
        easybench::bench(|| {
            let mut b = attached.clone();
            b.apply(m);
            b.undo(m);
        })
    );
}
