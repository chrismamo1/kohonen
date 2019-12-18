use std::f64;

#[derive(Clone)]
pub enum AssociationKind {
    Bucket(f64),
    Euclidean,
}

pub fn from_str(s: &str) -> Option<AssociationKind> {
    match s {
        "bucket" => Some(AssociationKind::Bucket(f64::NAN)),
        "euclidean" => Some(AssociationKind::Euclidean),
        _ => None,
    }
}