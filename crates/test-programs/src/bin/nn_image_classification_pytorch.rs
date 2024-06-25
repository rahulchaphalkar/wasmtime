use anyhow::Result;
use std::{fs, path::PathBuf};
use test_programs::nn::{classify, sort_results};
use wasi_nn::{ExecutionTarget, GraphBuilder, GraphEncoding};

pub fn main() -> Result<()> {
     //let xml = fs::read("fixture/resnet.pt")
     //    .expect("the model file to be mapped to the fixture directory");
    //let xml: PathBuf = PathBuf::from("fixture/resnet18.ot");
    // let weights = fs::read("fixture/model.bin")
    //     .expect("the weights file to be mapped to the fixture directory");
    // let _graph = GraphBuilder::new(GraphEncoding::Pytorch, ExecutionTarget::CPU)
    // .build_from_files(["./fixture/resnet.pt"])?;
    let graph = GraphBuilder::new(GraphEncoding::Pytorch, ExecutionTarget::CPU)
    .build_from_bytes(["fixture/resnet.pt"])?;
    println!("value of graph is {:?}", graph);


        //.build_from_bytes(&[xml])?;
    // let tensor = fs::read("fixture/tensor.bgr")
    //     .expect("the tensor file to be mapped to the fixture directory");
    // let results = classify(graph, tensor)?;
    // let top_five = &sort_results(&results)[..5];
    // println!("found results, sorted top 5: {:?}", top_five);
    Ok(())
}
