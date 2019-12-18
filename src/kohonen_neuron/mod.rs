use std;

pub trait KohonenNeuron: Clone + std::marker::Send {
    fn shift(&mut self, x: &Self, weight: f64) -> ();
    fn distance(&self, x: &Self) -> f64;
    fn get_rgb(&self) -> (u8, u8, u8);
    fn new() -> Self;
    fn combine(Vec<Self>) -> Self;
    //fn clone(&self) -> Self;
}

pub mod rgb_vector_neuron;
pub mod palette_hsl_neuron;
