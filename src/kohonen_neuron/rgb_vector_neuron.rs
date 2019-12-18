extern crate rand;
use kohonen_neuron::*;

impl KohonenNeuron for [f64; 3] {
    fn distance(&self, x: &[f64; 3]) -> f64 {
        let mut dist = 0.0;
        for i in 0..3 {
            let part = self[i] - x[i];
            dist = dist + (part * part);
        };
        //println!("got base dist {}", dist.sqrt());
        let rv = (dist / 3.0).sqrt();
        assert!(rv <= 1.0);
        rv
    }

    fn shift(&mut self, x: &Self, weight: f64) -> () {
        //let o_weight = 1.0 - weight;
        //assert_eq!(weight.is_normal(), true);
        assert_eq!(weight.is_nan(), false);
        assert!(weight >= 0.0);
        assert!(weight <= 1.0);
        /*use palette::{Hsv, LinSrgb, Mix};
        let self_hsv = Hsv::from(LinSrgb::new(self[0], self[1], self[2]));
        let x_hsv = Hsv::from(LinSrgb::new(x[0], x[1], x[2]));*/
        for i in 0..3 {
            self[i] = ((1.0 - weight) * self[i]) + (weight * x[i]);
        }
        /*let self_srgb = LinSrgb::from(self_hsv.mix(&x_hsv, weight));
        self[0] = self_srgb.red;
        self[1] = self_srgb.green;
        self[2] = self_srgb.blue;*/
        ()
    }

    fn get_rgb(&self) -> (u8, u8, u8) {
        let rf = self[0] * 255.0;
        let rg = self[1] * 255.0;
        let rb = self[2] * 255.0;
        (rf as u8, rg as u8, rb as u8)
    }

    fn new() -> Self {
        [rand::random::<f64>(), rand::random::<f64>(), rand::random::<f64>()]
    }

    fn combine(xs: Vec<Self>) -> Self {
        let mut acc = [0.0; 3];
        let n = xs.len() as f64;
        for [r, g, b] in xs.into_iter() {
            acc[0] += r * r;
            acc[1] += g * g;
            acc[2] += b * b;
        }

        acc[0] /= n;
        acc[1] /= n;
        acc[2] /= n;

        acc[0] = acc[0].sqrt();
        acc[1] = acc[1].sqrt();
        acc[2] = acc[2].sqrt();

        acc
    }
}