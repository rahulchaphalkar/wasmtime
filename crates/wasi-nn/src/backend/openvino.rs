//! Implements a `wasi-nn` [`BackendInner`] using OpenVINO.

use super::{
    read, BackendError, BackendExecutionContext, BackendFromDir, BackendGraph, BackendInner, Id,
};
use crate::wit::{ExecutionTarget, GraphEncoding, Tensor, TensorType};
use crate::{ExecutionContext, Graph};
use openvino::{DeviceType, ElementType, InferenceError, SetupError, Shape, Tensor as OvTensor};
use std::path::Path;
use std::sync::{Arc, Mutex};

#[derive(Default)]
pub struct OpenvinoBackend(Option<openvino::Core>);
unsafe impl Send for OpenvinoBackend {}
unsafe impl Sync for OpenvinoBackend {}

impl BackendInner for OpenvinoBackend {
    fn encoding(&self) -> GraphEncoding {
        GraphEncoding::Openvino
    }

    fn load(&mut self, builders: &[&[u8]], target: ExecutionTarget) -> Result<Graph, BackendError> {
        // (2x LLM) + (2x Tokenizer) + (2x Detokenizer) + (1x SharedObj Extension) = 7 expected for LLM
        if builders.len() < 2 || builders.len() > 3{
            return Err(BackendError::InvalidNumberOfBuilders(3, builders.len()).into());
        }
        // Construct the context if none is present; this is done lazily (i.e.
        // upon actually loading a model) because it may fail to find and load
        // the OpenVINO libraries. The laziness limits the extent of the error
        // only to wasi-nn users, not all WASI users.
        if self.0.is_none() {
            self.0.replace(openvino::Core::new()?);
        }
        let core = self
        .0
        .as_mut()
        .expect("openvino::Core was previously constructed");
        // Add extension
        if builders.len() > 2 {
            core.add_extension(std::str::from_utf8(builders[2]).unwrap())?;
        }
        // Read the guest array.
        let llm_xml = builders[0];
        let llm_weights = builders[1];
        // Construct a new tensor for the model weights.
        let shape = Shape::new(&[1, llm_weights.len() as i64])?;
        let mut weights_tensor = OvTensor::new(ElementType::U8, &shape)?;
        let buffer = weights_tensor.get_raw_data_mut()?;
        buffer.copy_from_slice(&llm_weights);
        // Construct OpenVINO graph structures: `model` contains the graph
        // structure, `compiled_model` can perform inference.
        let llm_model = core.read_model_from_buffer(&llm_xml, Some(&weights_tensor))?;
        let llm_compiled_model = core.compile_model(&llm_model, target.into())?;
        let box_: Box<dyn BackendGraph> =
            Box::new(OpenvinoGraph{
                llm_compiled_model: Arc::new(Mutex::new(llm_compiled_model)),
            });
        Ok(box_.into())
    }

    fn as_dir_loadable(&mut self) -> Option<&mut dyn BackendFromDir> {
        Some(self)
    }
}

impl BackendFromDir for OpenvinoBackend {
    fn load_from_dir(
        &mut self,
        path: &Path,
        target: ExecutionTarget,
    ) -> Result<Graph, BackendError> {
        let model = read(&path.join("model.xml"))?;
        let weights = read(&path.join("model.bin"))?;
        self.load(&[&model, &weights], target)
    }
}

struct OpenvinoGraph {
    llm_compiled_model: Arc<Mutex<openvino::CompiledModel>>,
}

unsafe impl Send for OpenvinoGraph {}
unsafe impl Sync for OpenvinoGraph {}

impl BackendGraph for OpenvinoGraph {
    fn init_execution_context(&self) -> Result<ExecutionContext, BackendError> {
        let mut compiled_model = self.llm_compiled_model.lock().unwrap();
        let infer_request = compiled_model.create_infer_request()?;

        let box_: Box<dyn BackendExecutionContext> =
            Box::new(OpenvinoExecutionContext(
                infer_request
            ));
        Ok(box_.into())
    }
}

struct OpenvinoExecutionContext(openvino::InferRequest);

impl BackendExecutionContext for OpenvinoExecutionContext {
    fn set_input(&mut self, id: Id, tensor: &Tensor) -> Result<(), BackendError> {
        // Construct the tensor.
        let precision = tensor.ty.into();
        let dimensions = tensor
            .dimensions
            .iter()
            .map(|&d| d as i64)
            .collect::<Vec<_>>();
        let shape = Shape::new(&dimensions)?;

        let mut new_tensor = OvTensor::new(precision, &shape)?;
        if precision == ElementType::String {
            new_tensor.set_string_data(&std::str::from_utf8(&tensor.data).unwrap()).unwrap();
            // let string_data = std::str::from_utf8(&tensor.data)
            //     .unwrap()
            //     .split('\0')
            //     .map(|s| s.to_string())
            //     .collect::<Vec<String>>();
            // new_tensor.set_string_data(string_data).unwrap();
        }
        else {
            let buffer = new_tensor.get_raw_data_mut()?;
            buffer.copy_from_slice(&tensor.data);
        }
        match id {
            Id::Index(i) => {if i>99 {let name = convert_index_to_name(i);self.0.set_tensor(&name, &new_tensor)?} else {self.0.set_input_tensor_by_index(i as usize, &new_tensor)?}},//self.0.set_input_tensor_by_index(i as usize, &new_tensor)?,
            Id::Name(name) => self.0.set_tensor(&name, &new_tensor)?,
        };
        Ok(())
    }

    fn compute(&mut self) -> Result<(), BackendError> {
        self.0.infer()?;
        Ok(())
    }

    fn get_output(&mut self, id: Id) -> Result<Tensor, BackendError> {
        let output_name = match id {
            Id::Index(i) => { if i>99 {let name = convert_index_to_name(i);self.0.get_tensor(name.as_str())?} else {self.0.get_output_tensor_by_index(i as usize)?}},//self.0.get_output_tensor_by_index(i as usize)?,
            Id::Name(name) => self.0.get_tensor(&name)?,
        };
        let dimensions = output_name
            .get_shape()?
            .get_dimensions()
            .iter()
            .map(|&dim| dim as u32)
            .collect::<Vec<u32>>();
        let ty = output_name.get_element_type()?.try_into()?;
        if ty == TensorType::Fp16 {
            let data = output_name.get_string_data()?;
            Ok(Tensor {
                dimensions,
                ty,
                data: data.into(),
            })
        }
        else {
            let data = output_name.get_raw_data()?.to_vec();
            Ok(Tensor {
                dimensions,
                ty,
                data,
            })
        }
    }
}

fn convert_index_to_name(i: u32) -> String {
    let name = match i {
        100 => "input_ids",
        101 => "attention_mask",
        102 => "position_ids",
        103 => "beam_idx",
        104 => "logits",
        _ => unimplemented!("error"),
    };
    name.to_string()
}

impl From<InferenceError> for BackendError {
    fn from(e: InferenceError) -> Self {
        BackendError::BackendAccess(anyhow::Error::new(e))
    }
}

impl From<SetupError> for BackendError {
    fn from(e: SetupError) -> Self {
        BackendError::BackendAccess(anyhow::Error::new(e))
    }
}

/// Return the execution target string expected by OpenVINO from the
/// `ExecutionTarget` enum provided by wasi-nn.
impl From<ExecutionTarget> for DeviceType<'static> {
    fn from(target: ExecutionTarget) -> Self {
        match target {
            ExecutionTarget::Cpu => DeviceType::CPU,
            ExecutionTarget::Gpu => DeviceType::GPU,
            ExecutionTarget::Tpu => {
                unimplemented!("OpenVINO does not support TPU execution targets")
            }
        }
    }
}

// Todo - Add string tensor type to wit/witx. For now, replace Fp16 with string
/// Return OpenVINO's precision type for the `TensorType` enum provided by
/// wasi-nn.
impl From<TensorType> for ElementType {
    fn from(tensor_type: TensorType) -> Self {
        match tensor_type {
            TensorType::Fp16 => ElementType::String,
            TensorType::Fp32 => ElementType::F32,
            TensorType::Fp64 => ElementType::F64,
            TensorType::U8 => ElementType::U8,
            TensorType::I32 => ElementType::I32,
            TensorType::I64 => ElementType::I64,
            TensorType::Bf16 => ElementType::Bf16,
        }
    }
}

/// Return the `TensorType` enum provided by wasi-nn for OpenVINO's precision type
impl TryFrom<ElementType> for TensorType {
    type Error = BackendError;
    fn try_from(element_type: ElementType) -> Result<Self, Self::Error> {
        match element_type {
            ElementType::String => Ok(TensorType::Fp16),
            ElementType::F16 => Ok(TensorType::Fp16),
            ElementType::F32 => Ok(TensorType::Fp32),
            ElementType::F64 => Ok(TensorType::Fp64),
            ElementType::U8 => Ok(TensorType::U8),
            ElementType::I32 => Ok(TensorType::I32),
            ElementType::I64 => Ok(TensorType::I64),
            ElementType::Bf16 => Ok(TensorType::Bf16),
            _ => Err(BackendError::UnsupportedTensorType(
                element_type.to_string(),
            )),
        }
    }
}
