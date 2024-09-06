//! Implements a `wasi-nn` [`BackendInner`] using PyTorch.
//!
use super::{BackendError, BackendExecutionContext, BackendFromDir, BackendGraph, BackendInner};
use crate::wit::types::{ExecutionTarget, GraphEncoding, Tensor, TensorType};
use crate::{ExecutionContext, Graph};
use std::path::Path;
use std::sync::{Arc, Mutex};
use tch::CModule;
use tch::{Device, Kind, TchError, Tensor as TchTensor};

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
        // Load the torchscript saved module.
        let mut saved_module = builders[0];

        // Load the saved model on the device.
        let mut compiled_module = CModule::load_data_on_device(
            &mut saved_module,
            map_execution_target_to_string(target),
        )?;

        // Set the model to be used for inference (eval), default is training.
        compiled_module.f_set_eval()?;

        let graph = PytorchGraph(Arc::new(Mutex::new(compiled_module)));
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
        // Load the model from the file path
        let compiled_module =
            CModule::load_on_device(path, map_execution_target_to_string(target)).unwrap();
        let graph = PytorchGraph(Arc::new(Mutex::new(compiled_module)));
        let box_: Box<dyn BackendGraph> = Box::new(graph);
        Ok(box_.into())
    }
}

struct PytorchGraph(Arc<Mutex<tch::CModule>>);

unsafe impl Send for PytorchGraph {}
unsafe impl Sync for PytorchGraph {}

impl BackendGraph for PytorchGraph {
    fn init_execution_context(&self) -> Result<ExecutionContext, BackendError> {
        let tensor = TchTensor::new();
        let box_: Box<dyn BackendExecutionContext> =
            Box::new(PytorchExecutionContext(self.0.clone(), tensor));
        Ok(box_.into())
    }
}

unsafe impl Sync for PytorchExecutionContext {}
struct PytorchExecutionContext(Arc<Mutex<tch::CModule>>, tch::Tensor);

impl BackendExecutionContext for PytorchExecutionContext {
    fn set_input(&mut self, _index: u32, input_tensor: &Tensor) -> Result<(), BackendError> {
        // Input index is not used in pytorch models. The forward method to a model passes the tensor/data to
        // the appropriate layer of the model.
        let kind = map_tensor_type_to_kind(input_tensor.tensor_type);
        let dimensions = input_tensor
            .dimensions
            .iter()
            .map(|&dim| dim as i64)
            .collect::<Vec<_>>();
        self.1 = TchTensor::from_data_size(&input_tensor.data, &dimensions, kind?);
        Ok(())
    }

    fn compute(&mut self) -> Result<(), BackendError> {
        // Use forward method on the compiled module/model after locking the mutex, and pass the input tensor to it
        self.1 = self.0.lock().unwrap().forward_ts(&[&self.1]).unwrap();
        Ok(())
    }

    fn get_output(&mut self, _index: u32, destination: &mut [u8]) -> Result<u32, BackendError> {
        // Output index is not used in pytorch models. The forward method to a model returns the output tensor.
        let numel = self.1.numel();
        if destination.len() < numel {
            return Err(BackendError::NotEnoughMemory(destination.len()));
        }
        self.1.copy_data_u8(destination, numel);
        Ok(numel as u32)
    }
}

fn map_execution_target_to_string(target: ExecutionTarget) -> Device {
    match target {
        ExecutionTarget::Cpu => Device::Cpu,
        ExecutionTarget::Gpu => {
            unimplemented!("Pytorch does not yet support GPU execution targets")
        }
        ExecutionTarget::Tpu => {
            unimplemented!("Pytorch does not yet support TPU execution targets")
        }
    }
}

/// Return PyTorch's precision type for the `TensorType` enum provided by
/// wasi-nn.
fn map_tensor_type_to_kind(tensor_type: TensorType) -> Result<Kind, TchError> {
    match tensor_type {
        TensorType::Fp16 => Ok(Kind::Half),
        TensorType::Fp32 => Ok(Kind::Float),
        TensorType::Fp64 => Ok(Kind::Double),
        TensorType::U8 => Ok(Kind::Uint8),
        TensorType::I32 => Ok(Kind::Int),
        TensorType::I64 => Ok(Kind::Int64),
        _ => Err(TchError::Convert(format!(
            "Tensor type {:?} is not supported",
            tensor_type
        ))),
    }
}

impl From<TchError> for BackendError {
    fn from(e: TchError) -> Self {
        BackendError::BackendAccess(anyhow::Error::new(e))
    }
}
