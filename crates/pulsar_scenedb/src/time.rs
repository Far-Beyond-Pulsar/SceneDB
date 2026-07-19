use std::time::{Duration, Instant};

#[derive(Clone, Copy, Debug)]
pub struct GameTime {
    pub elapsed: Duration,
    pub delta: Duration,
    pub tick: u64,
}

impl GameTime {
    #[inline]
    pub fn delta_secs(&self) -> f32 {
        self.delta.as_secs_f32()
    }

    #[inline]
    pub fn delta_secs_f64(&self) -> f64 {
        self.delta.as_secs_f64()
    }

    #[inline]
    pub fn elapsed_secs(&self) -> f32 {
        self.elapsed.as_secs_f32()
    }
}
