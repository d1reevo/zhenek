//! Periodic full GPU comparisons with sparse hot ranges between them.

use std::collections::{BTreeMap, HashMap};
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};

/// `VICTORY_CARD_COMPARE_HOT_SPAN` (90, about three frames): for how many
/// submissions after a compare found a piece changed the piece is compared
/// after every submission that declares it.
pub(super) fn hot_span() -> u64 {
    static SPAN: OnceLock<u64> = OnceLock::new();
    *SPAN.get_or_init(|| {
        std::env::var("VICTORY_CARD_COMPARE_HOT_SPAN")
            .ok()
            .and_then(|v| v.trim().parse().ok())
            .unwrap_or(90)
    })
}

/// `VICTORY_CARD_COMPARE_EVERY=<D>` (off by default): a declared range of a copy is read whole by
/// a compare at most once in `D` submissions; the compares between read only
/// its pieces found changed in the last [`hot_span`] submissions ([`Lean`]).
/// A shader write to a cold piece can be picked up by the next whole
/// comparison when the title declares that range again.
pub(super) fn every() -> Option<u64> {
    static EVERY: OnceLock<Option<u64>> = OnceLock::new();
    *EVERY.get_or_init(|| {
        std::env::var("VICTORY_CARD_COMPARE_EVERY")
            .ok()
            .and_then(|v| v.trim().parse::<u64>().ok())
            .filter(|&d| d > 0)
    })
}

/// What [`every`] goes by: for each declared range the submission whose
/// compare last read all of it, and for each copy the pieces found changed
/// and when. A range is a copy, its first piece and its pieces: the title
/// declares the same ranges frame after frame.
#[derive(Default)]
pub(super) struct Lean {
    whole: HashMap<(u64, u32, u32), u64>,
    changed: HashMap<u64, BTreeMap<u32, u64>>,
    last_prune: u64,
}

/// Since the last frame-rate line ([`lean_line`]): declared ranges read
/// whole, ranges read in their hot pieces only, the pieces read of those,
/// the pieces left.
pub(super) static LEAN_STATS: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];

impl Lean {
    /// Hot subranges of the declaration, in order, or `None` for a full compare.
    pub(super) fn plan(
        &mut self,
        number: u64,
        every: u64,
        copy: u64,
        first: u32,
        pieces: u32,
    ) -> Option<Vec<std::ops::Range<u32>>> {
        let span = hot_span();
        if number.saturating_sub(self.last_prune) >= every.max(span) {
            self.last_prune = number;
            let oldest = number.saturating_sub(every.max(span.saturating_mul(4)));
            self.whole.retain(|_, last| *last >= oldest);
            self.changed.retain(|_, changed| {
                changed.retain(|_, last| *last >= oldest);
                !changed.is_empty()
            });
        }
        let whole = self.whole.entry((copy, first, pieces)).or_default();
        if *whole == 0 || number.saturating_sub(*whole) >= every {
            *whole = number;
            LEAN_STATS[0].fetch_add(1, Ordering::Relaxed);
            return None;
        }
        let mut hot: Vec<std::ops::Range<u32>> = Vec::new();
        let mut count = 0u64;
        if let Some(changed) = self.changed.get(&copy) {
            for (&piece, &last) in changed.range(first..first + pieces) {
                if number.saturating_sub(last) > span {
                    continue;
                }
                match hot.last_mut() {
                    Some(run) if run.end == piece => run.end += 1,
                    _ => hot.push(piece..piece + 1),
                }
                count += 1;
            }
        }
        LEAN_STATS[1].fetch_add(1, Ordering::Relaxed);
        LEAN_STATS[2].fetch_add(count, Ordering::Relaxed);
        LEAN_STATS[3].fetch_add(u64::from(pieces) - count, Ordering::Relaxed);
        Some(hot)
    }

    /// Submission `number`'s compare found `piece` of `copy` changed.
    pub(super) fn found(&mut self, copy: u64, piece: u32, number: u64) {
        let at = self
            .changed
            .entry(copy)
            .or_default()
            .entry(piece)
            .or_insert(number);
        *at = (*at).max(number);
    }
}

/// The frame-rate line's part ([`LEAN_STATS`]), if the switch is on.
pub(super) fn lean_line(frames: f64) {
    let Some(every) = every() else {
        return;
    };
    let s = LEAN_STATS
        .each_ref()
        .map(|n| n.swap(0, Ordering::Relaxed) as f64 / frames);
    tracing::info!(
        "declared ranges read whole once in {every} submissions (VICTORY_CARD_COMPARE_EVERY, hot for {} submissions) per frame: {:.1} read whole, \
         {:.1} read in their hot pieces only, {:.0} pieces read of those, {:.0} left",
        hot_span(),
        s[0],
        s[1],
        s[2],
        s[3],
    );
}

#[cfg(test)]
mod tests {
    use super::Lean;

    #[test]
    fn compares_hot_runs_between_full_reads_of_each_range() {
        let mut heat = Lean::default();
        assert_eq!(heat.plan(10, 30, 5, 0, 8), None);
        assert_eq!(heat.plan(11, 30, 5, 0, 8), Some(vec![]));
        heat.found(5, 2, 11);
        heat.found(5, 3, 11);
        heat.found(5, 6, 11);
        heat.found(5, 8, 11);
        assert_eq!(heat.plan(12, 30, 5, 0, 8), Some(vec![2..4, 6..7]));
        assert_eq!(heat.plan(12, 30, 5, 1, 7), None);
        assert_eq!(heat.plan(39, 30, 5, 0, 8), Some(vec![2..4, 6..7]));
        assert_eq!(heat.plan(40, 30, 5, 0, 8), None);
    }
}
