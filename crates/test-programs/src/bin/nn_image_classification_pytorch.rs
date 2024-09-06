use anyhow::Result;
use std::fs;
use test_programs::nn::sort_results;
use wasi_nn::{ExecutionTarget, GraphBuilder, GraphEncoding};

pub fn main() -> Result<()> {
    let model =
        fs::read("fixture/model.pt").expect("the model file to be mapped to the fixture directory");
    let graph = GraphBuilder::new(GraphEncoding::Pytorch, ExecutionTarget::CPU)
        .build_from_bytes(&[&model])?;

    let mut context = graph.init_execution_context()?;

    let tensor_data = fs::read("fixture/kitten.tensor").unwrap();

    let precision = wasi_nn::TensorType::F32;

    // Resnet18 pytorch input is NCHW
    let shape = &[1, 3, 224, 224];

    context.set_input(0, precision, shape, &tensor_data)?;

    context.compute()?;

    let mut output_buffer = vec![0f32; 1000];
    context.get_output(0, &mut output_buffer[..])?;
    let result = softmax(output_buffer);
    let top_five = &sort_results(&result)[..5];
    println!("Found results, sorted top 5: {:?}", top_five);
    assert_eq!(top_five[0].class_id(), 281); // 281 is tabby cat
    Ok(())
}

fn softmax(output_tensor: Vec<f32>) -> Vec<f32> {
    let max_val = output_tensor
        .iter()
        .cloned()
        .fold(f32::NEG_INFINITY, f32::max);

    // Compute the exponential of each element subtracted by max_val for numerical stability.
    let exps: Vec<f32> = output_tensor.iter().map(|&x| (x - max_val).exp()).collect();

    // Compute the sum of the exponentials.
    let sum_exps: f32 = exps.iter().sum();

    // Normalize each element to get the probabilities.
    exps.iter().map(|&exp| exp / sum_exps).collect()
}
