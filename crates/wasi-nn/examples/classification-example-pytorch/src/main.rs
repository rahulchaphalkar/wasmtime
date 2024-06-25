//use anyhow::Result;
use std::{fs, path::PathBuf};
//use nn::{classify, sort_results};
use wasi_nn::{ExecutionTarget, GraphBuilder, GraphEncoding};

pub fn main() {
    let model = fs::read("fixture/resnet18.ot").unwrap();
    //println!("Read graph weights, size in bytes: {}", &model[..50.min(model.len())]);
    println!("Read graph weights, size in bytes: {}", model.len());
    let graph = GraphBuilder::new(GraphEncoding::Pytorch, ExecutionTarget::CPU)
    .build_from_bytes(&[model]).unwrap();
    println!("value of graph is {:?}", graph);

}
