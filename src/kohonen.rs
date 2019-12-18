use std::ops::{Index, IndexMut};
use kohonen_neuron;

#[derive(Clone)]
pub struct Kohonen<T: kohonen_neuron::KohonenNeuron> {
    data: Vec<Vec<T>>,
    pub rows: usize,
    pub cols: usize,
}

impl<T: kohonen_neuron::KohonenNeuron> Index<usize> for Kohonen<T> {
    type Output = Vec<T>;

    fn index(&self, r: usize) -> &Vec<T> {
        &self.data[r]
    }
}

impl<T: kohonen_neuron::KohonenNeuron> IndexMut<usize> for Kohonen<T> {
    fn index_mut(&mut self, r: usize) -> &mut Vec<T> {
        &mut self.data[r]
    }
}

/// Constructs a new square 2D Kohonen SOFM of side length `sz`
pub fn new<T: kohonen_neuron::KohonenNeuron>(sz: usize) -> Kohonen<T> {
    //let mut net: Kohonen<palette::Hsl> = vec![vec![palette::Hsl::new(0.5, 0.5, 0.5); sz]; sz];
    //let mut net: Kohonen<T> = vec![vec![eerst; sz]; sz];
    let mut net = Vec::new();
    for _i in 0..sz {
        let mut tmp = Vec::new();
        for _j in 0..sz {
            /*for k in 0..3 {
                net[i][j][k] = 0.5 * rand::random::<f32>() + 0.25;
            }*/
            tmp.push(T::new());
        }
        net.push(tmp)
    }
    return Kohonen {data: net, rows: sz, cols: sz}
}

pub fn shape<T>(net: &Kohonen<T>)
-> Vec<usize>
where T: kohonen_neuron::KohonenNeuron {
    let mut rv = Vec::new();
    rv.push(net.rows);
    rv.push(net.cols);
    rv
}

///
/// Get the position of the neuron within `net` that is closest to `x` and its distance
/// from `x`. If there are multiple neurons within some ε of the same distance from `x`
/// then `get_bmu` will return the one closest to the geometric "center" of them.
///
pub fn get_bmu<T>(net: &Kohonen<T>, x: &T)
-> (usize, usize, f64)
where T: kohonen_neuron::KohonenNeuron {
    let shape = shape(net);
    let epsilon = 1e-9;
    let mut rv = (0, 0, net[0][0].distance(x));
    let mut bmus = Vec::new();
    for r in 0..shape[0] {
        for c in 0..shape[1] {
            let (_, _, rd) = rv;
            let new_dist = net[r][c].distance(x);
            if new_dist < rd - epsilon {
                rv = (r, c, new_dist);
                let new_bmus = bmus.into_iter().filter(|(_, _, b_dist)| {
                    *b_dist <= new_dist + epsilon
                });
                bmus = Vec::new();
                bmus.push((r, c, new_dist));
                bmus.extend(new_bmus)
            } else if new_dist >= rd - epsilon && new_dist <= rd + epsilon {
                bmus.push((r, c, new_dist));
            }
        }
    };
    if bmus.len() > 1 {
        let mut cr = 0.0;
        let mut cc = 0.0;
        for i in 0..bmus.len() {
            let (r, c, _) = bmus[i];
            cr += (r * r) as f64;
            cc += (c * c) as f64;
        }
        cr /= bmus.len() as f64;
        cc /= bmus.len() as f64;

        cr = cr.sqrt();
        cc = cc.sqrt();

        fn max(a: usize, b: usize) -> usize {
            if a > b {
                a
            } else {
                b
            }
        }
        fn min(a: usize, b: usize) -> usize {
            if a < b {
                a
            } else {
                b
            }
        }
        fn eu_dist((r1, c1): (usize, usize), (r2, c2): (usize, usize)) -> f64 {
            let a = (max(r1, r2) - min(r1, r2)) as f64;
            let b = (max(c1, c2) - min(c1, c2)) as f64;
            // taking the sqrt of the distance because we'd like to favour placing the "center"
            // within the largest cluster of fits, to minimize secondary clusters.
            (a * a + b * b).sqrt().sqrt()
        }
        fn bmu_to_pos((r, c, _dist): (usize, usize, f64)) -> (usize, usize) {
            (r, c)
        }

        let center = (cr as usize, cc as usize);
        let mut best_fit = (bmu_to_pos(bmus[0]), eu_dist(bmu_to_pos(bmus[0]), center));
        for i in 1..bmus.len() {
            let (_, best_dist) = best_fit;
            let dist = eu_dist(bmu_to_pos(bmus[i]), center);
            if dist < best_dist {
                best_fit = (bmu_to_pos(bmus[i]), dist);
            }
        }
        let ((r, c), _) = best_fit;
        let dist = net[r][c].distance(x);
        println!("\t\tMoved the BMU to the centroid of {} very close neurons.", bmus.len());
        rv = (r, c, dist);
    }
    rv
}

pub fn disp<T: kohonen_neuron::KohonenNeuron>(a: &Kohonen<T>, b: &Kohonen<T>) -> f64 {
    let mut acc = 0.0;
    let rows = a.rows;
    let cols = a.cols;
    for r in 0..rows {
        for c in 0..cols {
            acc = acc + a[r][c].distance(&b[r][c]);
        }
    }
    acc
}

pub fn combine<T: kohonen_neuron::KohonenNeuron>(base: Kohonen<T>, parts: Vec<Kohonen<T>>) -> Kohonen<T> {
    let rows = parts[0].rows;
    let cols = parts[0].cols;
    let mut rv = new(rows);
    for r in 0..rows {
        for c in 0..cols {
            let mut neurons = Vec::new();
            for i in 0..parts.len() {
                if parts[i][r][c].distance(&base[r][c]) > 0.000001 {
                    neurons.push(parts[i][r][c].clone());
                }
            }
            if neurons.len() > 0 {
                rv[r][c] = T::combine(neurons);
            } else {
                rv[r][c] = base[r][c].clone();
            }
        }
    };
    rv
}