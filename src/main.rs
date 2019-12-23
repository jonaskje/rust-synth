extern crate anyhow;
extern crate cpal;
use itertools::Itertools;

use cpal::traits::{DeviceTrait, EventLoopTrait, HostTrait};

const MAX_CAPACITY: usize = 65536;

#[derive(Debug, Copy, Clone)]
struct Trigger {
    velocity: f32,  // [0, 1]
    frequency: f32, // hertz
    length: f32,    // in seconds
}

impl Trigger {
    fn new(velocity: f32, frequency: f32, length: f32) -> Self {
        Self {
            velocity: velocity,
            frequency: frequency,
            length: length,
        }
    }
}

#[derive(Debug, Copy, Clone)]
enum BiquadFilterType {
    LowPass,
    HighPass,
    BandPass,
}

#[derive(Default)]
struct BiquadFilter {
    in1: f32,
    in2: f32,
    out1: f32,
    out2: f32,
    a1: f32,
    a2: f32,
    b0: f32,
    b1: f32,
    b2: f32,
}

impl BiquadFilter {
    fn new(filter_type: BiquadFilterType, frequency: f32, sample_dt: f32, resonance: f32) -> Self {
        let q = 1.0 / (2.0 * resonance);
        let omega = 2.0 * std::f32::consts::PI * frequency * sample_dt; // radians per sample
        let alpha = q * omega.sin();
        let a1 = -2.0 * omega.cos() / (1.0 + alpha);
        let a2 = (1.0 - alpha) / (1.0 + alpha);
        let (b0, b1, b2) = match filter_type {
            BiquadFilterType::LowPass => {
                let b0 = (1.0 + a1 + a2) / 4.0;
                (b0, 2.0 * b0, b0)
            }
            BiquadFilterType::HighPass => {
                let b0 = (1.0 - a1 + a2) / 4.0;
                (b0, -2.0 * b0, b0)
            }
            BiquadFilterType::BandPass => {
                let b0 = (1.0 - a2) / 2.0;
                (b0, 0.0, -b0)
            }
        };

        Self {
            in1: 0.0,
            in2: 0.0,
            out1: 0.0,
            out2: 0.0,
            a1: a1,
            a2: a2,
            b0: b0,
            b1: b1,
            b2: b2,
        }
    }

    fn sample(&mut self, in_value: f32) -> f32 {
        let out = self.b0 * in_value + self.b1 * self.in1 + self.b2 * self.in2
            - self.a1 * self.out1
            - self.a2 * self.out2;
        self.in2 = self.in1;
        self.in1 = in_value;
        self.out2 = self.out1;
        self.out1 = out;
        out
    }

    fn reset(&mut self) {
        self.in1 = 0.0;
        self.in2 = 0.0;
        self.out1 = 0.0;
        self.out2 = 0.0;
    }
}

trait Track {
    fn trigger(&mut self, trigger: Trigger);
    fn render(&mut self, buf: &mut [f32]);
}

#[derive(Debug, Copy, Clone)]
enum SynthPreset {
    BasicSaw,
    Kick,
    Snare,
    HiHat,
    OpenHiHat,
    Temp,
    // OpenHiHat,
}

struct Synth {
    trigger: Option<Trigger>,
    t: f32,
    sample_dt: f32,

    filter0: BiquadFilter,
    filter1: BiquadFilter,
    filter2: BiquadFilter,
    filter3: BiquadFilter,

    preset: SynthPreset,
}

impl Synth {
    fn new(preset: SynthPreset, sample_dt: f32) -> Self {
        let mut result = Self {
            trigger: None,
            t: 0.0,
            sample_dt: sample_dt,
            filter0: Default::default(),
            filter1: Default::default(),
            filter2: Default::default(),
            filter3: Default::default(),
            preset: preset,
        };

        match preset {
            SynthPreset::BasicSaw => (),
            SynthPreset::Kick => {
                result.filter0 =
                    BiquadFilter::new(BiquadFilterType::LowPass, 200.0 + 1000.0, sample_dt, 0.5);
            }
            SynthPreset::Snare => {
                result.filter0 =
                    BiquadFilter::new(BiquadFilterType::HighPass, 800.0, sample_dt, 0.707);
                result.filter1 =
                    BiquadFilter::new(BiquadFilterType::LowPass, 8000.0, sample_dt, 0.707);
            }
            SynthPreset::HiHat | SynthPreset::OpenHiHat => {
                result.filter0 =
                    BiquadFilter::new(BiquadFilterType::BandPass, 12000.0, sample_dt, 0.707);
                result.filter1 =
                    BiquadFilter::new(BiquadFilterType::HighPass, 7000.0, sample_dt, 0.707);
            }
            SynthPreset::Temp => {
                result.filter0 =
                    BiquadFilter::new(BiquadFilterType::LowPass, 200.0 + 1000.0, sample_dt, 0.5);
            }
        }

        result
    }

    fn exp_ramp(t: f32, t0: f32, t1: f32, v0: f32, v1: f32) -> f32 {
        v0 * (v1 / v0).powf((t - t0) / (t1 - t0))
    }
    fn exp_decay(t: f32, decay_time: f32) -> f32 {
        Self::exp_ramp(t, 0.0, decay_time, 1.0, 0.00001)
    }
    fn exp_attack(t: f32, attack_time: f32) -> f32 {
        Self::exp_ramp(t, 0.0, attack_time, 0.00001, 1.0)
    }

    fn ramp(t: f32, t0: f32, t1: f32, v0: f32, v1: f32) -> f32 {
        (t - t0) / (t1 - t0) * (v1 - v0) + v0
    }
    fn decay(t: f32, decay_time: f32) -> f32 {
        Self::ramp(t, 0.0, decay_time, 1.0, 0.0)
    }
    fn attack(t: f32, attack_time: f32) -> f32 {
        Self::ramp(t, 0.0, attack_time, 0.0, 1.0)
    }

    fn exp_ad(t: f32, attack: f32, decay: f32) -> f32 {
        if t < attack {
            Self::attack(t, attack)
        } else if t < attack + decay {
            Self::exp_decay(t - attack, decay)
        } else {
            0.0
        }
    }

    fn ad(t: f32, attack: f32, decay: f32) -> f32 {
        if t < attack {
            Self::attack(t, attack)
        } else if t < attack + decay {
            Self::decay(t - attack, decay)
        } else {
            0.0
        }
    }

    fn adsr(
        t: f32,
        gate_length: f32,
        attack: f32,
        attack_value: f32,
        decay: f32,
        decay_value: f32,
        release: f32,
    ) -> f32 {
        if t < gate_length {
            if t < attack && attack > 0.0 {
                Self::exp_ramp(t, 0.0, attack, 0.00001, attack_value)
            } else if t < (attack + decay) && decay > 0.0 {
                Self::exp_ramp(t, attack, attack + decay, attack_value, decay_value)
            } else {
                decay_value
            }
        } else {
            if release > 0.0 {
                Self::exp_ramp(t, gate_length, gate_length + release, decay_value, 0.00001)
            } else {
                0.0
            }
        }
    }

    fn sine(t: f32, freq: f32) -> f32 {
        (t * freq * 2.0 * std::f32::consts::PI).sin()
    }
    fn triangle(t: f32, freq: f32) -> f32 {
        let a = (t * freq) % 1.0;
        if a < 0.5 {
            -1.0 + 2.0 * 2.0 * a
        } else {
            1.0 - 2.0 * 2.0 * (a - 0.5)
        }
    }
    fn square(t: f32, freq: f32) -> f32 {
        let a = (t * freq) % 1.0;
        if a < 0.5 {
            -1.0
        } else {
            1.0
        }
    }

    fn noise() -> f32 {
        unsafe {
            static mut STATE: u64 = 0x123456789abcdef0u64;
            let tmp = STATE.wrapping_mul(2862933555777941757u64) + 3037000493u64;
            STATE = tmp;
            (f32::from_bits(0x3F800000u32 | (tmp as u32) & 0x007fffff) - 1.5) * 2.0
        }
    }
}

impl Track for Synth {
    fn trigger(&mut self, trigger: Trigger) {
        self.trigger = Some(trigger);
        self.t = 0.0;
        self.filter0.reset();
        self.filter1.reset();
        self.filter2.reset();
        self.filter3.reset();
    }

    fn render(&mut self, buf: &mut [f32]) {
        if let Some(trigger) = &self.trigger {
            let mut t = self.t;
            let dt = self.sample_dt;
            let velocity = trigger.velocity;
            let length = trigger.length;
            let frequency = trigger.frequency;
            let out_iter = buf.iter_mut();
            match self.preset {
                SynthPreset::BasicSaw => {
                    for out in out_iter {
                        *out = velocity
                            * (Self::adsr(t, length * 20.0, 0.03, 1.0, 0.2, 0.9, 2.0)
                                * Self::triangle(t, frequency)
                                + Self::adsr(t, length, 0.03, 1.0, 0.2, 0.9, 2.0)
                                    * Self::triangle(
                                        t,
                                        frequency * (Self::sine(t, 1.0) * 0.005 + 1.0),
                                    ));
                        t += dt;
                    }
                }
                SynthPreset::Kick => {
                    for out in out_iter {
                        let decay = 0.200;
                        let osc = Self::exp_ad(t, 0.00011, decay) * 300.0 + 48.0;
                        let amp = Self::ad(t, 0.002, decay);
                        *out = velocity * self.filter0.sample(Self::sine(t, osc) * amp);
                        t += dt;
                    }
                }
                SynthPreset::Snare => {
                    for out in out_iter {
                        *out = velocity
                            * 0.5
                            * self.filter1.sample(
                                (Self::exp_decay(t, 0.2)
                                    * (0.5 * Self::sine(t, 180.0)
                                        + 0.4 * Self::sine(t, 330.0)
                                        + 0.5 * Self::triangle(t, 111.0))
                                    + Self::exp_decay(t, 0.3) * self.filter0.sample(Self::noise())),
                            );
                        t += dt;
                    }
                }
                SynthPreset::HiHat => {
                    //http://joesul.li/van/synthesizing-hi-hats/
                    for out in out_iter {
                        let b = 40.0;
                        let v = Self::square(t, b * 2.0)
                            + Self::square(t, b * 3.0)
                            + Self::square(t, b * 4.16)
                            + Self::square(t, b * 5.43)
                            + Self::square(t, b * 6.79)
                            + Self::square(t, b * 8.32);
                        *out = velocity
                            * self.filter1.sample(self.filter0.sample(v))
                            * Self::adsr(t, 0.005 + 0.01, 0.005, 1.0, 0.01, 0.333, 0.27);
                        t += dt;
                    }
                }
                SynthPreset::OpenHiHat => {
                    //http://joesul.li/van/synthesizing-hi-hats/
                    for out in out_iter {
                        let b = 40.0;
                        let v = Self::square(t, b * 2.0)
                            + Self::square(t, b * 3.0)
                            + Self::square(t, b * 4.16)
                            + Self::square(t, b * 5.43)
                            + Self::square(t, b * 6.79)
                            + Self::square(t, b * 8.32);
                        *out = velocity
                            * self.filter1.sample(self.filter0.sample(v))
                            * Self::adsr(t, 0.005 + 0.05, 0.005, 1.0, 0.05, 0.333, 1.5);
                        t += dt;
                    }
                }
                SynthPreset::Temp => {
                    for out in out_iter {
                        let decay = 0.200;
                        let osc = Self::exp_ad(t, 0.00011, decay) * 300.0 + 48.0;
                        let amp = Self::ad(t, 0.002, decay);
                        *out = velocity * self.filter0.sample(Self::sine(t, osc) * amp);
                        t += dt;
                    }
                }
            };
            self.t = t;
            if self.t >= trigger.length {
                //self.trigger = None; // TODO: would like to know when track is silent...
            }
        } else {
            buf.iter_mut().for_each(|it| *it = 0.0);
        }
    }
}

fn beats_to_seconds(bpm: f32, beat_count: f32) -> f32 {
    beat_count * 60.0 / bpm
}

fn tone_to_frequency(tone: char, octave: i32) -> f32 {
    let tone_nr = 12 * (octave - 4)
        + match tone {
            'a' => 0,
            'A' => 1,
            'b' => 2,
            'c' => 3,
            'C' => 4,
            'd' => 5,
            'D' => 6,
            'e' => 7,
            'f' => 8,
            'F' => 9,
            'g' => 10,
            'G' => 11,
            _ => panic!("Invalid tone"),
        };

    // tone_nr(-1) == G#4
    // tone_nr(0) == A4
    // tone_nr(1) == A#4, etc
    // A4 = 440Hz

    (2.0f32).powf(tone_nr as f32 / 12.0) * 440.0
}

fn parse_sequence(bpm: f32, s: &str) -> Vec<Option<Trigger>> {
    let result = s
        .split_whitespace()
        .map(|it| {
            if it.len() != 3 {
                panic!("Invalid sequence");
            }

            if it == "---" {
                None
            } else {
                let mut ch = it.chars();
                let tone_char = ch.next().unwrap();
                let octave = ch.next().unwrap().to_digit(10).unwrap() as i32;
                let velocity = ch.next().unwrap().to_digit(10).unwrap();
                Some(Trigger::new(
                    velocity as f32 / 9.0,
                    tone_to_frequency(tone_char, octave),
                    beats_to_seconds(bpm, 1.0 / 2.0),
                ))
            }
        })
        .collect::<Vec<Option<Trigger>>>();
    if result.len() != 32 {
        panic!("Expected sequence length == 32");
    }
    result
}

fn main() -> Result<(), anyhow::Error> {
    let host = cpal::default_host();
    let device = host
        .default_output_device()
        .expect("failed to find a default output device");
    let format = device.default_output_format()?;
    let event_loop = host.event_loop();
    let stream_id = event_loop.build_output_stream(&device, &format)?;
    event_loop.play_stream(stream_id.clone())?;

    let mut sample_count: usize = 0;
    let mut buf_storage = Box::new([0.0; MAX_CAPACITY]);

    let bpm = 160.0;
    let sample_rate = format.sample_rate.0 as usize;
    const SEGMENTS_PER_BEAT: usize = 8;
    let samples_per_segment: usize =
        ((sample_rate * 60) as f32 / (bpm * SEGMENTS_PER_BEAT as f32)) as usize;
    let sample_dt = 1.0 / sample_rate as f32;

    assert_eq!(format.channels, 2); // need to implement mixing the synth's stereo channels to outputs with more/fewer channels

    let mut left = Box::new([0.0; MAX_CAPACITY]);
    let mut right = Box::new([0.0; MAX_CAPACITY]);

    let mut tracks = vec![
        Synth::new(SynthPreset::Kick, sample_dt),
        Synth::new(SynthPreset::Snare, sample_dt),
        Synth::new(SynthPreset::HiHat, sample_dt),
        Synth::new(SynthPreset::OpenHiHat, sample_dt),
        Synth::new(SynthPreset::BasicSaw, sample_dt),
    ];

    let kick_beat_0 = parse_sequence(bpm, "c49 --- --- --- --- --- --- --- c49 --- --- --- --- --- --- --- c49 --- --- --- --- --- --- --- c49 --- --- --- --- --- --- ---");
    let kick_beat_1 = parse_sequence(bpm, "c49 --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- c49 --- --- --- --- --- --- --- --- --- --- ---");
    let snare_beat_0 = parse_sequence(bpm, "--- --- --- --- --- --- --- --- c49 --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- c49 --- --- --- --- --- --- ---");
    let snare_beat_1 = parse_sequence(bpm, "--- --- --- --- --- --- --- --- c49 --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- c49 --- --- --- c47 --- --- ---");
    let silence     = parse_sequence(bpm, "--- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---");
    let hihat_0 =     parse_sequence(bpm, "c49 --- --- --- c46 --- --- --- c49 --- --- --- c46 --- --- --- c49 --- --- --- c46 --- --- --- c49 --- --- --- c46 --- --- ---");
    let hihat_1 =     parse_sequence(bpm, "--- --- --- --- c49 --- --- --- --- --- --- --- c46 --- c49 --- --- --- c49 --- c46 --- --- --- --- --- --- --- c49 --- --- ---");
    let patterns = vec![
        // 00 (silent)
        vec![
            silence.clone(),
            silence.clone(),
            silence.clone(),
            silence.clone(),
            silence.clone(),
        ],
        // 01
        vec![
            kick_beat_1.clone(),
            snare_beat_0.clone(),
            hihat_1.clone(),
            kick_beat_0.clone(),
            //silence.clone(),
            parse_sequence(bpm, "c11 --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- D11 --- --- ---"),
        ],
        // 02
        vec![
            kick_beat_1.clone(),
            snare_beat_1.clone(),
            hihat_1.clone(),
            kick_beat_0.clone(),
            //silence.clone(),
            parse_sequence(bpm, "b01 --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---"),
        ],
        // 03
        vec![
            silence.clone(),
            silence.clone(),
            silence.clone(),
            kick_beat_0.clone(),
            silence.clone(),
        ],
    ];

    let song = vec![1, 1, 1, 2, 1, 1, 1, 2];
    //let song = vec![3];
    let mut song_pattern_index: i32 = -1;

    event_loop.run(move |id, result| {
        let data = match result {
            Ok(data) => data,
            Err(err) => {
                eprintln!("an error occurred on stream {:?}: {}", id, err);
                return;
            }
        };

        match data {
            cpal::StreamData::Output {
                buffer: cpal::UnknownTypeOutputBuffer::F32(mut buffer),
            } => {
                let samples_per_channel = buffer.len() / 2;
                let lbuf = &mut left[0..samples_per_channel];
                let rbuf = &mut right[0..samples_per_channel];
                for it in lbuf.iter_mut() {
                    *it = 0.0;
                }
                for it in rbuf.iter_mut() {
                    *it = 0.0;
                }
                {
                    let total_samples_to_render = lbuf.len();
                    let mut remaining_samples = total_samples_to_render;

                    let mut pan =
                        |buf: &mut [f32], left_pan: f32, right_pan: f32, offset: usize| {
                            for (src, dst) in buf.iter().zip(lbuf[offset..].iter_mut()) {
                                *dst += src * left_pan;
                            }
                            for (src, dst) in buf.iter().zip(rbuf[offset..].iter_mut()) {
                                *dst += src * right_pan;
                            }
                        };

                    while remaining_samples > 0 {
                        let segment = (sample_count / samples_per_segment) % 32 as usize;
                        let samples_to_render = (samples_per_segment
                            - (sample_count % samples_per_segment))
                            .min(remaining_samples);
                        let begin_segment = sample_count % samples_per_segment == 0;

                        if segment == 0 && begin_segment == true {
                            song_pattern_index = (song_pattern_index + 1) % song.len() as i32;
                        }

                        if begin_segment == true {
                            for (seq_index, seq) in patterns[song[song_pattern_index as usize]]
                                .iter()
                                .enumerate()
                            {
                                if let Some(trigger) = seq[segment] {
                                    tracks[seq_index].trigger(trigger);
                                }
                            }
                        }

                        let buf = &mut buf_storage[0..samples_to_render];

                        for track in tracks.iter_mut() {
                            track.render(buf);
                            pan(buf, 1.0, 1.0, total_samples_to_render - remaining_samples);
                        }

                        remaining_samples -= samples_to_render;
                        sample_count += samples_to_render;
                    }
                }

                let mut src = left.iter().interleave(right.iter());
                for dst in buffer.iter_mut() {
                    *dst = *src.next().unwrap();
                }
            }
            _ => (),
        }
    });
}
