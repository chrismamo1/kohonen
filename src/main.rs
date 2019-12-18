extern crate rand;
extern crate palette;
use std::fs::File;
use std::io::prelude::*;
use std::path::Path;
extern crate clap;
use std::thread;
use std::sync::mpsc;
extern crate rayon;
use rayon::prelude::*;

mod kohonen_neuron;
//use kohonen_neuron::rgb_vector_neuron;
mod kohonen;
use kohonen::Kohonen;
mod sphere_of_influence;

/** note: the energy coefficient should be from [0, 1] and should take into account both
 * distance from the BMU and color disparity
 */
pub fn get_within_radius<T>(net: &Kohonen<T>, pos: (usize, usize), radius: i32) ->
        std::vec::Vec<(usize, usize, f64)>
    where T: kohonen_neuron::KohonenNeuron {
    let mut rv = Vec::new();
    let (r, c) = pos;
    let bmu = &net[r][c];
    for r2 in 0..net.rows {
        for c2 in 0..net.cols {
            let comp1 = (r as f64) - (r2 as f64);
            let comp2 = (c as f64) - (c2 as f64);
            let distance = ((comp1 * comp1) + (comp2 * comp2)).sqrt();
            if distance < (radius as f64) {
                let color_dist = bmu.distance(&net[r2][c2]);
                let energy = (distance / radius as f64) * (1.0 - color_dist);
                rv.push((r2, c2, energy))
            }
        }
    }
    rv
}

pub fn get_neuron_neighbors<T>(net: &Kohonen<T>, pos: (usize, usize)) -> [(usize, usize); 8]
where T: kohonen_neuron::KohonenNeuron {
    let (r, c) = pos;
    let rows = net.rows;
    let cols = net.cols;
    assert_eq!(rows, cols);
    let prev = |x| {
        if x as i32 - 1 < 0 {
            rows - 1
        } else {
            x - 1
        }
    };
    let next = |x| (x + 1) % rows;
    [   (prev(r), prev(c)),   (prev(r), c),   (prev(r), next(c)),
        (r, prev(c)),                         (r, next(c)),
        (next(r), prev(c)),   (next(r), c),   (next(r), next(c))
    ]
}

/**
 * @returns a vector of triples consisting of (row, col, energy coefficient from [0, 1])
 */
pub fn get_within_radius_fluid<T>(
    net: &Kohonen<T>,
    pos: (usize, usize),
    radius: i32,
    bucket_decay: f64)
-> std::vec::Vec<(usize, usize, f64)>
where T: kohonen_neuron::KohonenNeuron {
    use std::collections::{HashSet, HashMap};
    fn fluid_collect<T: kohonen_neuron::KohonenNeuron>(
            net: &Kohonen<T>,
            pos: (usize, usize),
            range: i32,
            pow_exp: f64)
                -> Vec<(usize, usize, f64)> {
        let (ro, co) = pos;
        // use variant of Dijkstra's algorithm to produce the shortest-path tree, then
        // prune that tree
        let mut unvisited_nodes = HashSet::new();
        for r in 0..net.rows {
            for c in 0..net.cols {
                unvisited_nodes.insert((r, c));
            }
        }
        let inf = 0.0;
        let mut energies: HashMap<(usize, usize), f64> = unvisited_nodes.clone().into_iter()
            .map(|cur_pos| if cur_pos != pos {
                    (cur_pos, inf)
                } else {
                    let (cur_r, cur_c) = cur_pos;
                    (pos, (1.0 - net[ro][co].distance(&net[cur_r][cur_c]).powf(pow_exp)))
                })
            .collect();
        let mut current = pos;
        while unvisited_nodes.len() > 0 {
            let neighbours = get_neuron_neighbors(net, current);
            let unvisited_neighbours: Vec<(usize, usize)> =
                neighbours.iter()
                    .filter(|neighbour| unvisited_nodes.contains(*neighbour))
                    .map(|pos| *pos)
                    .collect();
            let current_dist = *energies.get(&current).unwrap();
            {
                let _res: Vec<(usize, usize)> =
                    unvisited_neighbours.clone().into_iter().map(
                        |(r, c)| {
                            let decay = 1.0 - 1.0 / range as f64;
                            let new_dist = (1.0 - net[ro][co].distance(&net[r][c]).powf(pow_exp)) * current_dist * decay;
                            let old_dist = *energies.get(&(r, c)).unwrap();
                            if new_dist > old_dist {
                                energies.remove(&(r, c));
                                energies.insert((r, c), new_dist);
                            };
                            (r, c)
                        })
                        .collect();
            };
            let old_len = unvisited_nodes.len();
            unvisited_nodes.remove(&current);
            assert!(old_len > unvisited_nodes.len());
            if unvisited_nodes.len() > 0 {
                let old_cur = current;
                current =
                    unvisited_nodes.clone().into_iter().fold(
                        None,
                        |acc, cand|
                            match acc {
                            None        =>
                                Some(cand),
                            Some(pos)   =>
                                if energies.get(&cand) > energies.get(&pos) {
                                    Some(cand)
                                } else {
                                    acc
                                },
                            })
                        .unwrap();
                assert!(old_cur != current);
            };
        }
        energies.into_iter()
            .filter(|(_pos, energy)| range as f64 * energy >= 1.0)
            .map(|((r, c), energy)| (r, c, /*range as f64 * */energy))
            .collect()
    }
    let collected = fluid_collect(net, pos, radius, bucket_decay);
    collected
        .into_iter()
        /*.map(|(r, c, local_range)| {
            (r, c, radius - local_range)
        })*/
        .collect()
}

pub fn feed_sample<T>(
    net: &mut Kohonen<T>,
    sample: &T,
    rate: f64,
    radius: i32,
    associate: sphere_of_influence::AssociationKind)
-> ()
where T: kohonen_neuron::KohonenNeuron {
    let (r, c, bmu_dist) = kohonen::get_bmu(net, sample);
    let bmu_pos = (r, c);
    let items =
        match associate {
            sphere_of_influence::AssociationKind::Bucket(bucket_decay) =>
                get_within_radius_fluid(net, (r, c), radius, bucket_decay),
            sphere_of_influence::AssociationKind::Euclidean =>
                get_within_radius(net, (r, c), radius),
        };
    let mut displaced = 0.0;
    for i in 0..items.len() {
        let (r, c, item_dist) = items[i];
        let dist = item_dist as f64 / radius as f64;
        let weight = (1.0 - dist).sqrt() * rate;

        let old = &net[r][c].clone();
        let _ = (&mut net[r][c]).shift(&sample, weight);
        displaced = displaced + old.distance(&net[r][c]);
        if (r, c) == bmu_pos {
            //println!("\tweighting with {} at the BMU.", weight);
        } else {
            //println!("\tweighting with {} as {:?}.", weight, (r, c));
        }
    }
    println!("\tDisplaced total of {} from {} items on a BMU of distance {}.",
        displaced,
        items.len(),
        bmu_dist);
    std::io::stdout().flush().unwrap();
    thread::yield_now();
    ()
}

pub fn train<T>(
    net: Kohonen<T>,
    samples: &Vec<T>,
    rate: f64,
    radius: i32,
    associate: sphere_of_influence::AssociationKind)
-> Kohonen<T>
where   T: kohonen_neuron::KohonenNeuron + Send + Sync + Clone + 'static,
        Kohonen<T>: Send + Sync {
    let mut descs = Vec::new();
    for i in 0..samples.len() {
        descs.push((net.clone(), samples[i].clone()));
        //feed_sample(net, &samples[i], rate, radius);
    }
    let nets: Vec<Kohonen<T>> =
        descs
            .par_iter()
            .map(|(my_net, sample)| {
                let associate = associate.clone();
                let mut net = my_net.clone();
                feed_sample(&mut net, &sample, rate, radius, associate);
                net
            })
            .collect();
    std::io::stdout().flush().unwrap();
    kohonen::combine(net, nets)
}

pub fn iter_train<T>(
    net: &Kohonen<T>,
    samples: &std::vec::Vec<T>,
    its: u32,
    associate: sphere_of_influence::AssociationKind)
-> Kohonen<T>
where T: kohonen_neuron::KohonenNeuron + Send + Sync + 'static {
    let mut rv = net.clone();
    let width = net.cols as f64;
    // training with a large fixed radius for a bit should help things get into
    // the right general places
    /*for _i in 0..(its / 2) {
        let radius = width / 2.0;
        let rate = 0.5;
        rv = train(rv.clone(), samples, rate, radius as i32, associate.clone());
    }
    let its = its / 2 + (its % 2);*/
    let time_constant = (its + 1) as f64 / width.ln();
    for i in 0..its {
        let radius = width * (0.0 - (i as f64 + 1.0) / time_constant).exp();
        //let radius = width / 2.0;
        let rate = (0.0 - (i as f64 + 1.0) / time_constant).exp().sqrt();
        //let rate = 0.75;
        println!("Radius: {radius}, rate: {rate}", radius=radius, rate=rate);
        std::io::stdout().flush().unwrap();
        let net2 = rv.clone();
        rv = train(net2, samples, rate, radius.ceil() as i32, associate.clone())
    }
    rv
}

pub fn show<T: kohonen_neuron::KohonenNeuron>(net: &Kohonen<T>, path: &str) {
    let rows = net.rows;
    let cols = net.cols;
    let path = Path::new(path);
    let mut os = match File::create(&path) {
        Err(why) => panic!("couldn't make file pls halp: {}", why),
        Ok(file) => file,
    };
    let _ = os.write_all("P6\n".as_bytes());
    let _ = os.write_all((cols as u64).to_string().as_bytes());
    let _ = os.write_all(" ".as_bytes());
    let _ = os.write_all((rows as u64).to_string().as_bytes());
    let _ = os.write_all("\n255\n".as_bytes());
    for r in 0..rows {
        for c in 0..cols {
            let (r, g, b) = net[r][c].get_rgb();
            let _ = os.write_all(&[r, g, b]);
        }
    }
}

/*pub fn show_csv<T: kohonen_neuron::KohonenNeuron>(net: &kohonen<T>, path: &str) {
    
}*/

fn main() {
    use clap::{Arg, App};
    let matches =
        App::new("kohonen")
            .version("0.1.0")
            .about("A Kohonen SOFM")
            .author("Fuck off")
            .arg(Arg::with_name("iterations")
                .short("its")
                .long("iterations")
                .help("How many training iterations to do")
                .takes_value(true))
            .arg(Arg::with_name("dim")
                .short("dim")
                .long("dimension")
                .help("The size of the network")
                .takes_value(true))
            .arg(Arg::with_name("associate")
                .short("a")
                .long("associate")
                .help("The association method")
                .default_value("bucket")
                .possible_values(&["bucket", "euclidean"])
                .takes_value(true))
            .arg(Arg::with_name("bucket decay")
                .long("bucket-decay")
                .help( "Exponentially affects how much energy it takes to overcome a higher \
                        difference. Lower values will keep spheres of influence small and \
                        tight, while higher ones (above 1.0) will allow greater spread.")
                .default_value("0.7")
                .takes_value(true))
            .arg(Arg::with_name("colour model")
                .long("colour-model")
                .help("The color model to use.")
                .default_value("hsl")
                .possible_values(&["hsl", "rgb"])
                .takes_value(true))
            .arg(Arg::with_name("centroids")
                .long("centroids")
                .help("A list of centroids")
                .takes_value(true))
            .get_matches();
    let net_dim = str::parse::<u32>(matches.value_of("dim").unwrap()).unwrap();
    let train_its = str::parse::<u32>(matches.value_of("iterations").unwrap()).unwrap();
    let associate = sphere_of_influence::from_str(matches.value_of("associate").unwrap()).unwrap();
    let bucket_decay = str::parse::<f64>(matches.value_of("bucket decay").unwrap()).unwrap();
    let associate = match associate {
        sphere_of_influence::AssociationKind::Bucket(_) =>
            sphere_of_influence::AssociationKind::Bucket(bucket_decay),
        _ => associate
    };
    println!(
        "Building a Kohonen net of {dim}x{dim} and training it for {its} iterations.",
        dim=net_dim, its=train_its);
    let colors: Vec<[f64; 3]> = vec![
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.8, 0.8, 0.0],
        [0.0, 0.8, 0.8],
        [0.8, 0.0, 0.8],
        [0.4, 0.4, 0.4],
        [1.0, 1.0, 1.0],
        [0.0, 0.0, 0.0],
        [1.0, 0.66, 0.75],
    ];
    match matches.value_of("colour model").unwrap() {
        "hsl" => {
            let mut net = kohonen::new(net_dim as usize);
            let old_net = net.clone();
            let colors = colors
                .into_iter()
                .map(|[r, g, b]|
                    palette::Hsl::from(palette::Srgb::new(r as f32, g as f32, b as f32)))
                .rev()
                .collect();
            net = iter_train(&net, &colors, train_its, associate);
            println!("Overall displacement: {}", kohonen::disp(&old_net, &net));
            let file = format!("./map_{its}its.ppm", its=train_its);
            show(&net, &file)
        },
        "rgb" => {
            let mut net = kohonen::new(net_dim as usize);
            let old_net = net.clone();
            net = iter_train(&net, &colors, train_its, associate);
            println!("Overall displacement: {}", kohonen::disp(&old_net, &net));
            let file = format!("./map_{its}its.ppm", its=train_its);
            show(&net, &file)
        },
        _ => ()
    };
}
