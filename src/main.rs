extern crate anyhow;
extern crate cpal;
use itertools::Itertools;

use cpal::traits::{DeviceTrait, EventLoopTrait, HostTrait};



const MAX_CAPACITY: usize = 65536;

fn noise() -> f32 {
    unsafe {
        static mut STATE: u64 = 0x123456789abcdef0u64;
        let tmp = STATE.wrapping_mul(2862933555777941757u64) + 3037000493u64;
        STATE = tmp;
        (f32::from_bits(0x3F800000u32 | (tmp as u32) & 0x007fffff) - 1.5) * 2.0
    }
}

fn exponential_ramp(t: f32, t0: f32, t1: f32, v0: f32, v1: f32) -> f32 {
    v0 * (v1/v0).powf((t-t0)/(t1-t0))
}

fn decay(t: f32, decay_time: f32) -> f32 {
    exponential_ramp(t, 0.0, decay_time, 1.0, 0.00001)
}

fn attack(t: f32, attack_time: f32) -> f32 {
    exponential_ramp(t, 0.0, attack_time, 0.00001, 1.0)
}

fn adr(t: f32, attack: f32, attack_value: f32, decay: f32, decay_value: f32, release: f32) -> f32 {
    if t < attack && attack > 0.0 {
        exponential_ramp(t, 0.0, attack, 0.00001, attack_value)
    } else if t < decay && decay > 0.0 {
        exponential_ramp(t, attack, attack + decay, attack_value, decay_value)
    } else if t < release && release > 0.0 {
        exponential_ramp(t, attack + decay, attack + decay + release, decay_value, 0.00001)
    } else {
        0.0
    }
}


fn sine(t: f32, freq: f32) -> f32 {
    (t * freq * 2.0 * std::f32::consts::PI).sin()
}

fn cose(t: f32, freq: f32) -> f32 {
    (t * freq * 2.0 * std::f32::consts::PI).cos()
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

fn make_bp_filter_biquad(freq: f32, dt: f32) -> impl FnMut(f32) -> f32 {
    // https://www.vicanek.de/articles/BiquadFits.pdf
    let mut in1 = 0.0;
    let mut in2 = 0.0;
    let mut out1 = 0.0;
    let mut out2 = 0.0;
    let resonance = 0.707; // Q
    let q = 1.0/(2.0 * resonance);
    let omega = 2.0 * std::f32::consts::PI * freq * dt;  // radians per sample
    let alpha = q * omega.sin();
    let a1 = -2.0 * omega.cos() / (1.0 + alpha);
    let a2 = (1.0-alpha)/(1.0+alpha);
    let b0 = (1.0 - a2)/2.0;
    let b1 = 0.0;
    let b2 = -b0;
    move |in_value: f32| -> f32 {
        let out = b0*in_value + b1*in1 + b2*in2 - a1*out1 - a2*out2;
        in2 = in1;
        in1 = in_value;
        out2 = out1;
        out1 = out;
        out
    }
}

fn make_hp_filter_biquad(freq: f32, dt: f32) -> impl FnMut(f32) -> f32 {
    // https://www.vicanek.de/articles/BiquadFits.pdf
    let mut in1 = 0.0;
    let mut in2 = 0.0;
    let mut out1 = 0.0;
    let mut out2 = 0.0;
    let resonance = 0.707; // Q
    let q = 1.0/(2.0 * resonance);
    let omega = 2.0 * std::f32::consts::PI * freq * dt;  // radians per sample
    let alpha = q * omega.sin();
    let a1 = -2.0 * omega.cos() / (1.0 + alpha);
    let a2 = (1.0-alpha)/(1.0+alpha);
    let b0 = (1.0 - a1 + a2)/4.0;
    let b1 = -2.0 * b0;
    let b2 = b0;
    move |in_value: f32| -> f32 {
        let out = b0*in_value + b1*in1 + b2*in2 - a1*out1 - a2*out2;
        in2 = in1;
        in1 = in_value;
        out2 = out1;
        out1 = out;
        out
    }
}

fn make_lp_filter_biquad(freq: f32, dt: f32) -> impl FnMut(f32) -> f32 {
    // https://www.vicanek.de/articles/BiquadFits.pdf
    let mut in1 = 0.0;
    let mut in2 = 0.0;
    let mut out1 = 0.0;
    let mut out2 = 0.0;
    let resonance = 0.707; // Q
    let q = 1.0/(2.0 * resonance);
    let omega = 2.0 * std::f32::consts::PI * freq * dt;  // radians per sample
    let alpha = q * omega.sin();
    let a1 = -2.0 * omega.cos() / (1.0 + alpha);
    let a2 = (1.0-alpha)/(1.0+alpha);
    let b0 = (1.0 + a1 + a2)/4.0;
    let b1 = 2.0 * b0;
    let b2 = b0;
    move |in_value: f32| -> f32 {
        let out = b0*in_value + b1*in1 + b2*in2 - a1*out1 - a2*out2;
        in2 = in1;
        in1 = in_value;
        out2 = out1;
        out1 = out;
        out
    }
}

fn kick1(t: f32) -> f32 {
    sine(t, 65.41 * (1.0 + (4.0 - 1.0) * decay(t, 0.015))) * decay(t, 0.30)
}

fn make_kick2(dt: f32) -> impl FnMut(f32) -> f32 {
    let mut lp2 = make_lp_filter_biquad(300.0, dt);
    let mut lp_tones = make_lp_filter_biquad(200.0, dt);
    move |t: f32| {
        if t == 0.0 {
            // TODO: better way to reset filters (to avoid clicks)
            lp2 = make_lp_filter_biquad(300.0, dt);
            lp_tones = make_lp_filter_biquad(200.0, dt);
        }
        let n = lp2(noise());
        let pitch = decay(t, 0.1) * 0.1 + 1.0;
        let v = 
            1.00 * sine(t, 50.0 * pitch) +
            0.80 * sine(t, 86.0 * pitch) +
            0.70 * sine(t, 93.0 * pitch) +
            0.60 * sine(t, 118.0 * pitch) +
            0.40 * sine(t, 182.0 * pitch) +
            0.30 * sine(t, 225.0 * pitch) +
            0.20 * sine(t, 273.0 * pitch);
        decay(t, 0.6) * (lp_tones(v/4.0) + 0.5*n)
    }
}


fn make_snare1(dt: f32) -> impl FnMut(f32) -> f32 {
    let mut filter2 = make_hp_filter_biquad(100.0, dt);
    move |t: f32| {
        0.5 * (decay(t, 0.2) * (0.5*sine(t, 180.0) + 0.4*sine(t, 330.0) + 0.5*triangle(t, 111.0)) + decay(t, 0.5) * filter2(noise()))
    }
}

fn make_snare2(dt: f32) -> impl FnMut(f32) -> f32 {
    let mut filter2 = make_hp_filter_biquad(100.0, dt);
    move |t: f32| {
        0.5 * (decay(t, 0.3) * (0.5*sine(t, 180.0) + 0.4*sine(t, 330.0) + 0.5*triangle(t, 111.0) + filter2(noise())))
    }
}

fn make_closed_hat1(dt: f32) -> impl FnMut(f32) -> f32 {
    //http://joesul.li/van/synthesizing-hi-hats/
    let mut bandpass = make_bp_filter_biquad(10000.0, dt);
    let mut hp = make_hp_filter_biquad(7000.0, dt);
    move |t: f32| {
        let b = 40.0;
        let v = square(t, b * 2.0) 
        + square(t, b * 3.0) 
        + square(t, b * 4.16) 
        + square(t, b * 5.43) 
        + square(t, b * 6.79) 
        + square(t, b * 8.32);
        hp(bandpass(v)) * adr(t, 0.005, 1.0, 0.01, 0.333, 0.27)
    }
}

fn make_open_hat1(dt: f32) -> impl FnMut(f32) -> f32 {
    //http://joesul.li/van/synthesizing-hi-hats/
    let mut bandpass = make_bp_filter_biquad(10000.0, dt);
    let mut hp = make_hp_filter_biquad(7000.0, dt);
    move |t: f32| {
        let b = 40.0;
        let v = square(t, b * 2.0) 
        + square(t, b * 3.0) 
        + square(t, b * 4.16) 
        + square(t, b * 5.43) 
        + square(t, b * 6.79) 
        + square(t, b * 8.32);
        hp(bandpass(v)) * adr(t, 0.005, 1.0, 0.05, 0.333, 0.5)
    }
}


// fn make_bass1(dt: f32) -> impl FnMut(f32) -> f32 {
//     move |t: f32| {
//         let b = 40.0;
//         let v = square(t, b * 2.0) 
//         + square(t, b * 3.0) 
//         + square(t, b * 4.16) 
//         + square(t, b * 5.43) 
//         + square(t, b * 6.79) 
//         + square(t, b * 8.32);
//         hp(bandpass(v)) * adr(t, 0.02, 1.0, 0.01, 0.333, 0.27)
//     }
// }


struct Track<F> {
    instrument: F,
    clock: f32,
    sample_dt: f32, // time in seconds per sample
}

impl<F> Track<F> 
where
    F: FnMut(f32)->f32 
{
    pub fn new(instrument: F, sample_dt: f32) -> Self {
        Self {
            instrument: instrument,
            clock: -1.0,
            sample_dt: sample_dt,
        }
    }

    pub fn trigger(&mut self) {
        self.clock = 0.0;
    }

    pub fn render(&mut self, buf: &mut [f32]) -> bool {
        if self.clock < 0.0 {
            false
        } else {
            let mut t = self.clock;
            let dt = self.sample_dt;
            for it in buf {
                *it = (self.instrument)(t);
                t += dt;
            }
            self.clock = t;
            true
        }
    }
}

fn make_synth(sample_rate: usize) -> impl FnMut(&mut [f32], &mut [f32]) -> () {
    let mut sample_count: usize = 0;
    let mut buf_storage = Box::new([0.0; MAX_CAPACITY]);
    const BPM: usize = 135;
    const SEGMENTS_PER_BEAT: usize = 8;
    let samples_per_segment: usize = (sample_rate * 60) / (BPM * SEGMENTS_PER_BEAT);
    let sample_dt = 1.0 / sample_rate as f32;

    // one 4/4 patch is split in 32 (1/8ths) segments and each segment is rendered in sequence for each instruments in turn
    //"9-------9-------9-------9-------"

    let mut kick = Track::new(make_kick2(sample_dt), sample_dt);
    let mut snare = Track::new(make_snare1(sample_dt), sample_dt);
    let mut closed_hat = Track::new(make_closed_hat1(sample_dt), sample_dt);
    let mut open_hat = Track::new(make_open_hat1(sample_dt), sample_dt);

    let kick_track  = "9-------9-----------9---9-------";
    let kick_track2 = "9-------9-------9-------9-------";
    //let kick_track  = "c29,---,---,---,---,---,---,---,c29,---,---,---,---,---,---,---,---,---,---,---,c29,---,---,---,c29,---,---,---,---,---,---,---,";
    let snare_track = "--------9---------------9-----9-";
    let closed_hat_track = "9-9---9-9-9---9-9-9---9-9-9---9-";
    let closed_hat_track2= "9-9-9-9-9-9-9-9-9-9-9-9-9-9-9-9-";
    let open_hat_track   = "----9-------9-------9-------9---";
    let once_track       = "9-------------------------------";

    move |left: &mut [f32], right: &mut [f32]| {
        let total_samples_to_render = left.len();
        let mut remaining_samples = total_samples_to_render;

        let mut pan = |buf: &mut[f32], left_pan: f32, right_pan: f32, offset: usize| {
            for (src, dst) in buf.iter().zip(left[offset..].iter_mut()) {
                *dst += src * left_pan;
            }
            for (src, dst) in buf.iter().zip(right[offset..].iter_mut()) {
                *dst += src * right_pan;
            }
        };

        while remaining_samples > 0 {
            let segment = sample_count / samples_per_segment;
            let samples_to_render = (samples_per_segment - (sample_count % samples_per_segment)).min(remaining_samples);
            let begin_segment = sample_count % samples_per_segment == 0;

            if begin_segment == true {
                let index = segment % 32 as usize;
                if kick_track2.as_bytes()[index] != '-' as u8 {
                    kick.trigger();
                }
                if snare_track.as_bytes()[index] != '-' as u8 {
                    snare.trigger();
                }
                if closed_hat_track.as_bytes()[index] != '-' as u8 {
                    closed_hat.trigger();
                }
                if open_hat_track.as_bytes()[index] != '-' as u8 {
                    open_hat.trigger();
                }
            }

            let buf = &mut buf_storage[0..samples_to_render];

            if kick.render(buf) {
                pan(buf, 1.0, 1.0, total_samples_to_render - remaining_samples);
            }
            
            if snare.render(buf) {
                pan(buf, 0.3, 0.4, total_samples_to_render - remaining_samples);
            }

            if closed_hat.render(buf) {
                pan(buf, 0.34, 0.3, total_samples_to_render - remaining_samples);
            }

            if open_hat.render(buf) {
                pan(buf, 0.34, 0.3, total_samples_to_render - remaining_samples);
            }
            remaining_samples -= samples_to_render;
            sample_count += samples_to_render;
            //panic!("dfdf");

        }
    }
}

fn main() -> Result<(), anyhow::Error> {

    let host = cpal::default_host();
    let device = host.default_output_device().expect("failed to find a default output device");
    let format = device.default_output_format()?;
    let event_loop = host.event_loop();
    let stream_id = event_loop.build_output_stream(&device, &format)?;
    event_loop.play_stream(stream_id.clone())?;

    assert_eq!(format.channels, 2); // need to implement mixing the synth's stereo channels to outputs with more/fewer channels

    let mut synth = make_synth(format.sample_rate.0 as usize);

    let mut left = Box::new([0.0; MAX_CAPACITY]);
    let mut right = Box::new([0.0; MAX_CAPACITY]);

    event_loop.run(move |id, result| {
        let data = match result {
            Ok(data) => data,
            Err(err) => {
                eprintln!("an error occurred on stream {:?}: {}", id, err);
                return;
            }
        };

        match data {
            cpal::StreamData::Output { buffer: cpal::UnknownTypeOutputBuffer::F32(mut buffer) } => {
                let samples_per_channel = buffer.len() / 2;
                let lbuf = &mut left[0..samples_per_channel];
                let rbuf = &mut right[0..samples_per_channel];
                for it in lbuf.iter_mut() { *it = 0.0; }
                for it in rbuf.iter_mut() { *it = 0.0; }
                synth(lbuf, rbuf);
                let mut src = left.iter().interleave(right.iter());
                for dst in buffer.iter_mut() {
                    *dst = *src.next().unwrap();
                }
            },
            _ => (),
        }
    });
}

#[cfg(test)]
mod tests {
    // Note this useful idiom: importing names from outer (for mod tests) scope.
    use super::*;
    use std::fs::File;
    use std::io::prelude::*;

    #[test]
    fn test_kick() -> std::io::Result<()> {
        let mut file = File::create("foo.csv")?;
        let sample_rate = 48000;
        let sample_dt = 1.0 / sample_rate as f32;
        let mut buf: [f32; 12000] = [0.0; 12000];

        // let mut instr = Track::new(make_kick2(sample_dt), sample_dt);
        //let mut instr = Track::new(make_snare1(sample_dt), sample_dt);
        let mut instr = Track::new(make_closed_hat1(sample_dt), sample_dt);
        instr.trigger();
        instr.render(&mut buf);

        for it in buf.iter() {
            file.write(format!("{}\n", *it).as_bytes());
        }
        Ok(())
    }

}
 