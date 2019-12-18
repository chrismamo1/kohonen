extern crate rand;
use kohonen_neuron::*;
use palette::Hsl;

impl KohonenNeuron for Hsl {
    ///
    /// Returns the distance as the average difference of each component (H, S, and L)
    /// with the Hue counting twice as much as the other two.
    fn distance(&self, x: &Self) -> f64 {
        let hue_diff: f32 = (self.hue - x.hue).into();
        let hue_diff = hue_diff.abs() / 360.0;
        //println!("hue_diff: {}", hue_diff);
        if !(hue_diff >= 0.0 && hue_diff <= 1.0) {
            println!("bad hue_diff: {}", hue_diff);
        }
        assert!(hue_diff >= 0.0 && hue_diff <= 1.0);
        let saturation_diff = (self.saturation - x.saturation).abs();
        let lightness_diff = (self.lightness - x.lightness).abs();
        (hue_diff as f64 * 2.0 + saturation_diff as f64 + lightness_diff as f64) / 4.0
    }

    fn shift(&mut self, x: &Self, weight: f64) -> () {
        //let o_weight = 1.0 - weight;
        let hue: f32 = self.hue.into();
        let hue = hue / 360.0;
        let x_hue: f32 = x.hue.into();
        let x_hue = x_hue / 360.0;
        let hue = (hue + x_hue) / 2.0;
        let saturation = (self.saturation + x.saturation) / 2.0;
        let lightness = (self.lightness + x.lightness) / 2.0;
        assert_eq!(weight.is_normal(), true);
        //use palette::{Mix};
        //let nv = self.mix(x, weight as f32);
        self.hue = Hsl::new(hue * 360.0, 0.0, 0.0).hue;
        self.saturation = saturation;
        self.lightness = lightness;
        ()
    }

    fn get_rgb(&self) -> (u8, u8, u8) {
        use palette::{Srgb};
        let as_rgb = Srgb::from(*self);
        let rf = as_rgb.red * 255.0;
        let rg = as_rgb.green * 255.0;
        let rb = as_rgb.blue * 255.0;
        (rf as u8, rg as u8, rb as u8)
    }

    fn new() -> Self {
        use palette::LinSrgb;
        Hsl::from(LinSrgb::new(rand::random::<f32>(), rand::random::<f32>(), rand::random::<f32>()))
    }

    fn combine(xs: Vec<Self>) -> Self {
        let n = xs.len() as f32;
        let mut hue = 0.0;
        let mut saturation = 0.0;
        let mut lightness = 0.0;
        for x in xs.into_iter() {
            let hue2: f32 = x.hue.into();
            let saturation2: f32 = x.saturation.into();
            let lightness2: f32 = x.lightness.into();
            hue += hue2;
            saturation += saturation2;
            lightness += lightness2;
        }
        hue /= n;
        saturation /= n;
        lightness /= n;
        Hsl::new(hue, saturation, lightness)
    }

    /*fn clone(&self) -> Self {
        use palette::{Hsl};
        Hsl::new(self.hue, self.saturation, self.lightness)
    }*/
}