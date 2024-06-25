use super::{
    read, BackendError, BackendExecutionContext, BackendFromDir, BackendGraph, BackendInner,
};
use crate::wit::types::{ExecutionTarget, GraphEncoding, Tensor, TensorType};
use crate::{ExecutionContext, Graph};
//use anyhow::Ok;
use tch::nn::VarStore;
use tch::CModule;
use tch::{nn::ModuleT, Device, Tensor as TchTensor, Kind};
use tch::vision::{ resnet, imagenet };
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::io::Cursor;

#[derive(Default)]
pub struct PytorchBackend();
unsafe impl Send for PytorchBackend {}
unsafe impl Sync for PytorchBackend {}

impl BackendInner for PytorchBackend {
    fn encoding(&self) -> GraphEncoding {
        GraphEncoding::Pytorch
    }
    
    fn load(&mut self, builders: &[&[u8]], target: ExecutionTarget) -> Result<Graph, BackendError> {
        if builders.len() != 1 {
            return Err(BackendError::InvalidNumberOfBuilders(1, builders.len()).into());
        }
        // Load the torchscript saved model.
        let mut saved_model = builders[0];

        // Load the saved model on the device.
        let mut compiled_module = CModule::load_data_on_device(&mut saved_model, map_execution_target_to_string(target))?;

        // Set the model to be used for inference (eval), default is training.
        compiled_module.f_set_eval()?;

        let graph = PytorchGraph(
            Arc::new(Mutex::new(compiled_module),
            ),
        );
        let box_: Box<dyn BackendGraph> = Box::new(graph);
        Ok(box_.into())
    }
    
    fn as_dir_loadable<'a>(&'a mut self) -> Option<&'a mut dyn BackendFromDir> {
        Some(self)
    }
}

impl BackendFromDir for PytorchBackend {
    fn load_from_dir(
        &mut self,
        path: &Path,
        target: ExecutionTarget,
    ) -> Result<Graph, BackendError> {
        // Load the model from the file.
        let compiled_model = CModule::load_on_device(path, map_execution_target_to_string(target)).unwrap();
        let box_: Box<dyn BackendGraph> = Box::new(PytorchGraph(
            Arc::new(Mutex::new(compiled_model)),
        ));
        Ok(box_.into())

        //todo!()
    }
}

struct PytorchGraph(
    Arc<Mutex<tch::CModule>>,
);

unsafe impl Send for PytorchGraph {}
unsafe impl Sync for PytorchGraph {}

impl BackendGraph for PytorchGraph {
    fn init_execution_context(&self) -> Result<ExecutionContext, BackendError> {


        todo!()
    }
}

unsafe impl Sync for PytorchExecutionContext {}
struct PytorchExecutionContext(Arc<tch::CModule>, tch::Tensor);

impl BackendExecutionContext for PytorchExecutionContext {
    fn set_input(&mut self, index: u32, input_tensor: &Tensor) -> std::result::Result<(), BackendError>{
        //let input = TchTensor::clone_from_ptr(c_tensor)
        let vec_i64: Vec<i64> = input_tensor.dimensions.iter().map(|x| *x as i64).collect();
        //let input = TchTensor::from_data_size(input_tensor.data.as_slice(), vec_i64.as_slice(), Kind::Uint8);
        self.1 = TchTensor::from_data_size(input_tensor.data.as_slice(), vec_i64.as_slice(), Kind::Uint8);
        //let input_slice = vec![input];
        //input_slice
        //let output = self.0.forward_ts(&input_slice).unwrap();
        Ok(())
        //todo!()
    }
    
    fn compute(&mut self) -> Result<(), BackendError> {
        let input = self.1.shallow_clone();
        let input_slice = vec![input];
        self.1 = self.0.forward_ts(&input_slice).unwrap();
        Ok(())
    }
    
    fn get_output(&mut self, index: u32, destination: &mut [u8]) -> Result<u32, BackendError> {
        todo!()
    }
}

/// Return the execution target string expected by OpenVINO from the
/// `ExecutionTarget` enum provided by wasi-nn.
fn map_execution_target_to_string(target: ExecutionTarget) -> Device {
    match target {
        ExecutionTarget::Cpu => Device::Cpu,
        ExecutionTarget::Gpu => unimplemented!("Pytorch does not yet support GPU execution targets"),
        ExecutionTarget::Tpu => unimplemented!("Pytorch does not yet support TPU execution targets"),
    }
}

impl From<tch::TchError> for BackendError {
    fn from(e: tch::TchError) -> Self {
        BackendError::BackendAccess(e.into())
    }
}