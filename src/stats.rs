use crate::Pilot;

#[derive(Default, Clone)]
struct Row {
    buckets: usize,
    elements: usize,
    pilot_sum: Pilot,
    pilot_max: Pilot,
}

impl Row {
    fn add(&mut self, bucket_len: usize, pilot: Pilot) {
        self.buckets += 1;
        self.elements += bucket_len;
        self.pilot_sum += pilot;
        self.pilot_max = self.pilot_max.max(pilot);
    }
}

pub struct BucketStats {
    by_pct: Vec<Row>,
    by_bucket_len: Vec<Row>,
}

impl BucketStats {
    pub fn new() -> Self {
        Self {
            by_pct: vec![Row::default(); 100],
            by_bucket_len: vec![Row::default(); 100],
        }
    }

    pub fn add(&mut self, bucket_id: usize, buckets_total: usize, bucket_len: usize, pilot: Pilot) {
        let pct = bucket_id * 100 / buckets_total;
        self.by_pct[pct].add(bucket_len, pilot);
        if self.by_bucket_len.len() <= bucket_len {
            self.by_bucket_len.resize(bucket_len + 1, Row::default());
        }
        self.by_bucket_len[bucket_len].add(bucket_len, pilot);
    }

    pub fn print(&self) {
        eprintln!();
        Self::print_rows(&self.by_pct, false);
        eprintln!();
        Self::print_rows(&self.by_bucket_len, true);
        eprintln!();
    }

    fn print_rows(rows: &[Row], reverse: bool) {
        let b_total = rows.iter().map(|r| r.buckets).sum::<usize>();
        let n = rows.iter().map(|r| r.elements).sum::<usize>();

        eprintln!(
            "{:>3}  {:>11} {:>7} {:>6} {:>6} {:>6} {:>10} {:>10}",
            "sz", "cnt", "bucket%", "cuml%", "elem%", "cuml%", "avg p", "max p"
        );
        let mut bucket_cuml = 0;
        let mut elem_cuml = 0;
        let process_row = |row: &Row| {
            if row.buckets == 0 {
                return;
            }
            bucket_cuml += row.buckets;
            elem_cuml += row.elements;
            eprintln!(
                "{:>3}: {:>11} {:>7.2} {:>6.2} {:>6.2} {:>6.2} {:>10.1} {:>10}",
                row.elements / row.buckets,
                row.buckets,
                row.buckets as f32 / b_total as f32 * 100.,
                bucket_cuml as f32 / b_total as f32 * 100.,
                row.elements as f32 / n as f32 * 100.,
                elem_cuml as f32 / n as f32 * 100.,
                row.pilot_sum as f32 / row.buckets as f32,
                row.pilot_max,
            );
        };
        if reverse {
            rows.iter().rev().for_each(process_row);
        } else {
            rows.iter().for_each(process_row);
        }
        let sum_pilots = rows.iter().map(|r| r.pilot_sum).sum::<Pilot>();
        let max_pilot = rows.iter().map(|r| r.pilot_max).max().unwrap();
        eprintln!(
            "{:>3}: {:>11} {:>7.2} {:>6.2} {:>6.2} {:>6.2} {:>10.1} {:>10}",
            "",
            b_total,
            100.,
            100.,
            100.,
            100.,
            sum_pilots as f32 / b_total as f32,
            max_pilot
        );
    }
}
