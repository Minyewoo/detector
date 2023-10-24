use std::rc::Rc;
use opencv::{prelude::Mat, core::lut};
use super::layer::Layer;

pub struct GammaCorrection {
    layer: Box<dyn Layer>,
    look_up_table: Rc<Mat>
}

impl GammaCorrection {
    pub fn new(layer: Box<dyn Layer>, look_up_table: Rc<Mat>) -> Self {
        GammaCorrection { layer, look_up_table }
    }
}

impl Layer for GammaCorrection {
    fn process(&mut self) -> Result<Rc<Mat>, String> {
        match self.layer.process() {
            Ok(frame) => {
                let mut contrast_frame = Mat::default();
                lut(frame.as_ref(), self.look_up_table.as_ref(), &mut contrast_frame)
                    .map_err(|err: opencv::Error| format!("[LutLayer] lut: {err}"))?;
                Ok(Rc::new(contrast_frame))
            },
            Err(error) => Err(error),
        }
    }
}